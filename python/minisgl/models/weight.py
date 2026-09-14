"""权重加载：把 HF 的 safetensors 流式读进来，边读边做三件事 ——

    切分   按 TP 规则切出本 rank 的那一份（规则必须与 layers/linear.py 里的
           各 Linear 子类**完全一致**，否则形状断言会当场报错）
    合并   权重文件里是分着的 q_proj / k_proj / v_proj（以及 gate/up），
           而运行时是一个 LinearQKVMerged / LinearColParallelMerged 的大矩阵
    堆叠   MoE 的每个专家在文件里是独立的一份，运行时打包成 (num_experts, …) 的三维张量

之所以做成生成器（yield），是为了"流式"：一次只在内存里留一个完整张量 + 一个小的
合并缓冲（见 load_weight 的 docstring），而不是把整个模型先全读进 CPU 再拷到 GPU。
"""

from __future__ import annotations

import glob
import re
from typing import Dict, Iterator, Tuple

import safetensors
import torch
from minisgl.distributed import get_tp_info
from minisgl.utils import cached_load_hf_config, div_ceil, download_hf_weight
from tqdm import tqdm

# 两类切法的关键词：与 Linear 子类的选择一一对应
_SPLIT_DIM_0 = [".q_proj", ".k_proj", ".v_proj", ".gate_proj", ".up_proj"]
_SPLIT_DIM_1 = [".o_proj", ".down_proj"]

# Merge groups: individual projections -> fused projection
# 合并后的键 → 各分量在拼接结果里的名字与顺序（顺序必须和运行时 split/cat 的顺序一致：
# AttentionLayer 按 [q, k, v] 切，silu_and_mul 按 [gate, up] 切）
_MERGE_GROUPS = {
    ".q_proj": (".qkv_proj", ("q", "k", "v")),
    ".k_proj": (".qkv_proj", ("q", "k", "v")),
    ".v_proj": (".qkv_proj", ("q", "k", "v")),
    ".gate_proj": (".gate_up_proj", ("gate", "up")),
    ".up_proj": (".gate_up_proj", ("gate", "up")),
}
_SLOT_NAMES = {
    ".q_proj": "q",
    ".k_proj": "k",
    ".v_proj": "v",
    ".gate_proj": "gate",
    ".up_proj": "up",
}
# 匹配 "...experts.5.gate_proj.weight" 这类专家专属键
_EXPERT_PATTERN = re.compile(r"^(?P<prefix>.+\.experts)\.(?P<idx>\d+)\.(?P<name>.+)$")


def _shard_tensor(key: str, value: torch.Tensor, r: int, n: int, num_kv_heads: int):
    """Extract rank r's shard from a single tensor. Returns a contiguous copy."""
    if any(key.count(sub) for sub in _SPLIT_DIM_0):
        # 列并行：按输出维（dim 0）切
        is_kv_proj = any(key.count(sub) for sub in (".k_proj", ".v_proj"))
        if is_kv_proj and num_kv_heads is not None and num_kv_heads < n:
            # kv 头比 TP 还少 → 头要被复制（与 div_even(..., allow_replicate=True) 对应）：
            # 连续若干个 rank 共用同一个头，每个 rank 只拿 1 个头。
            # 能走到这里说明 kv 头数一定整除 tp（否则层那边的 div_even 会先报错）
            head_dim = value.shape[0] // num_kv_heads
            head_idx = r * num_kv_heads // n
            return value[head_idx * head_dim : (head_idx + 1) * head_dim].clone()
        # 常规情形：平均切成 n 份取第 r 份（能整除由层里的 div_even 保证）
        return value.chunk(n, dim=0)[r].clone()
    elif any(key.count(sub) for sub in _SPLIT_DIM_1):
        # 行并行：按输入维（dim 1）切
        return value.chunk(n, dim=1)[r].clone()
    elif key.count("lm_head") or key.count("embed_tokens"):
        # 词表并行：公式必须与 VocabParallelEmbedding.__init__ 里的完全一致
        # （向上取整 → 最后一段可能短一点，所以末尾用 min 夹住）
        num_embeddings = value.shape[0]
        num_embeddings_per_partition = div_ceil(num_embeddings, n)
        vocab_start_idx = r * num_embeddings_per_partition
        vocab_end_idx = min((r + 1) * num_embeddings_per_partition, num_embeddings)
        return value[vocab_start_idx:vocab_end_idx, :].clone()
    else:
        # 复制式权重（norm 的 weight、bias、MoE 的 gate、路由相关的小矩阵……）：
        # 原样交给每个 rank。注意这里是**唯一不 clone 的分支** —— 上面几个分支
        # 切片出来的是视图，会拖住整个原张量不放，必须 clone 才能让 del raw 真正释放显存
        return value


def _get_merge_info(key: str):
    """If key belongs to a merge group, return (merged_key, slot, all_slots). Else None."""
    for suffix, (fused_suffix, slots) in _MERGE_GROUPS.items():
        if key.count(suffix):
            return key.replace(suffix, fused_suffix), _SLOT_NAMES[suffix], slots
    return None


def _get_expert_stack_info(key: str) -> tuple[str, int] | None:
    """Map an expert-scoped checkpoint key to the packed runtime key."""
    # 例：`…mlp.experts.5.gate_up_proj.weight` → ("…mlp.experts.gate_up_proj", 5)。
    # 去掉 ".weight" 是因为运行时的 MoELayer 把权重存在 gate_up_proj / down_proj
    # 这两个属性上（键里本来就没有 .weight）。
    match = _EXPERT_PATTERN.match(key)
    if match is None:
        return None

    packed_name = match.group("name")
    if packed_name.endswith(".weight"):
        packed_name = packed_name.removesuffix(".weight")
    return f"{match.group('prefix')}.{packed_name}", int(match.group("idx"))


def load_weight(model_path: str, device: torch.device) -> Iterator[Tuple[str, torch.Tensor]]:
    """Streaming weight loader. Yields (name, tensor) pairs already sharded, merged,
    and on device. Peak CPU memory: one full tensor + a small merge buffer."""
    from .config import ModelConfig

    model_folder = download_hf_weight(model_path)
    # 需要 num_kv_heads（决定 kv 是否复制）和 num_experts（决定要不要堆叠专家）
    config = ModelConfig.from_hf(cached_load_hf_config(model_path))
    files = glob.glob(f"{model_folder}/*.safetensors")
    # 有的仓库同时放了分片文件和一份合并文件，优先用分片（避免同一份权重读两遍）
    files = [f for f in files if not f.endswith("consolidated.safetensors")] or files
    tp_info = get_tp_info()

    # Buffer for merge groups: merged_key -> {slot: tensor}
    # 合并缓冲：q/k/v（gate/up）是分散在文件里的，要攒齐了才能 cat 成一个大矩阵
    merge_buf: Dict[str, Dict[str, torch.Tensor]] = {}
    # 专家缓冲：每个专家的张量单独到达，攒够 num_experts 个才能 stack 成三维权重
    expert_buf: Dict[str, Dict[int, torch.Tensor]] = {}
    for file in tqdm(files, desc="Loading weights", disable=not tp_info.is_primary()):
        # 直接在目标设备上打开：省掉 CPU 内存中转和一次 H2D 拷贝
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for name in f.keys():
                # Strip multimodal wrapper prefix, skip vision/projector weights
                # 多模态模型只跑文本部分：视觉塔和投影层直接丢
                if name.startswith(("vision_tower.", "multi_modal_projector.")):
                    continue
                raw = f.get_tensor(name)
                # 多模态的文本权重带这个前缀，去掉才能对上模型的键名
                name = name.removeprefix("language_model.")
                tensor = _shard_tensor(name, raw, tp_info.rank, tp_info.size, config.num_kv_heads)
                # 切完之后原始张量就没用了，赶紧放掉（流式加载的关键一步）
                del raw

                if (info := _get_merge_info(name)) is None:
                    out = (name, tensor)
                else:
                    # 攒合并组：三份到齐才 cat 出一个 qkv_proj
                    merged_key, slot, all_slots = info
                    merge_buf.setdefault(merged_key, {})[slot] = tensor
                    if not all(s in merge_buf[merged_key] for s in all_slots):
                        continue  # 还差分量，继续攒
                    # 按 all_slots 的顺序拼（q、k、v 或 gate、up），与运行时切分顺序对应
                    parts = [merge_buf[merged_key][s] for s in all_slots]
                    del merge_buf[merged_key]
                    out = (merged_key, torch.cat(parts, dim=0))

                if config.is_moe and (expert_info := _get_expert_stack_info(out[0])) is not None:
                    # 专家权重：先按 (打包键, 专家号) 攒起来，攒齐 num_experts 个再 stack
                    packed_key, expert_idx = expert_info
                    slots = expert_buf.setdefault(packed_key, {})
                    slots[expert_idx] = out[1]
                    if len(slots) != config.num_experts:
                        continue  # 还有专家没到
                    experts = [slots[idx] for idx in range(config.num_experts)]
                    del expert_buf[packed_key]
                    # (num_experts, out, in) —— 就是 MoELayer 里那两个三维张量的形状
                    yield packed_key, torch.stack(experts, dim=0)
                else:  # Normal dense model
                    yield out[0], out[1]

    # 收尾自检"攒了一半"的情况：权重文件缺分量/缺专家，宁可在这里报错也不要静默少权重
    # （漏掉普通权重则会在另一头暴露：BaseOP.load_state_dict 的 pop 会 KeyError）
    assert not merge_buf, f"Incomplete merge groups in checkpoint: {list(merge_buf.keys())}"
    assert not expert_buf, f"Incomplete expert tensors in checkpoint: {list(expert_buf.keys())}"
