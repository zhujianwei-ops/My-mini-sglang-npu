"""Engine：一个 TP rank 上的"模型执行器"——把 GPU 侧的模型、KV 池、页表、采样器
组装起来，对外只暴露两个动作：

    forward_batch(batch, args)   跑一次 forward（prefill 或 decode），返回采样出的 token
    shutdown()                   拆 CUDA graph 和通信资源

它不管调度（谁上谁下、什么时候淘汰前缀，全在 scheduler/），只管"给定这一批请求，
在这张卡上把 forward 跑出来"。它建立的、全局可见的东西有四样：

    model       权重（meta 设备上建骨架 → 再灌真权重，见 _load_weight_state_dict）
    kv_cache    物理 KV 池（num_pages + 1 页，多出来的那页是 dummy 页）
    page_table  (max_running_req + 1, aligned_max_seq_len) 的 int32，行 = 请求，列 = token 位置
    ctx         运行时上下文，注册成全局单例（set_global_ctx），模型各层用 get_global_ctx() 取

__init__ 里各段的先后顺序是有依赖的，不能随意调换，详见段内注释。两条：
先量显存再建模型（KV 容量要靠差值算），_adjust_config 必须在建 Context 之前
（它可能改掉 page_size，而 Context 和各 attention 后端都要读这个值）。

每次 forward 都走的热路径只有 forward_batch 一个，其余方法都只在启动时跑一次。
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict, NamedTuple, Tuple

import torch
from minisgl.attention import create_attention_backend
from minisgl.core import Batch, Context, Req, set_global_ctx
from minisgl.distributed import destroy_distributed, enable_pynccl_distributed, set_tp_info
from minisgl.kvcache import create_kvcache_pool
from minisgl.layers import set_rope_device
from minisgl.models import create_model, load_weight
from minisgl.moe import create_moe_backend
from minisgl.utils import div_even, init_logger, is_sm90_supported, is_sm100_supported, torch_dtype

from .config import EngineConfig
from .graph import GraphRunner, get_free_memory, mem_GB
from .sample import BatchSamplingArgs, Sampler

logger = init_logger(__name__)


class ForwardOutput(NamedTuple):
    """一次 forward 的产物：采样出的下一个 token（device 版 + CPU 版）+ 一个完成事件。

    为什么 token 要一式两份（产生过程见 forward_batch 结尾）：
      - gpu 版：调度器直接 scatter 回 token_pool（纯 device 操作，不产生同步）；
      - cpu 版：等 copy_done_event 之后用来 detokenize、判 EOS / 长度上限。
    事件是给调度器的"这份 CPU 副本已经拷好了"的信号。
    """

    next_tokens_gpu: torch.Tensor
    next_tokens_cpu: torch.Tensor
    copy_done_event: torch.cuda.Event


class Engine:
    def __init__(self, config: EngineConfig):
        # 必须在任何 CUDA 初始化之前：KV 页数是拿"还没建模型时的空闲显存"量出来的
        # （见 _determine_num_pages），显存先被别人占了就会算少
        assert not torch.cuda.is_initialized()
        # 先定 TP 身份，再改配置：_adjust_config 可能覆盖 attention_backend / page_size，
        # 而它们马上会被 Context 和各后端读走，所以必须排在最前面
        set_tp_info(rank=config.tp_info.rank, size=config.tp_info.size)
        _adjust_config(config)

        self.device = torch.device(f"cuda:{config.tp_info.rank}")
        torch.cuda.set_device(self.device)
        # 固定种子：--use-dummy-weight 的随机权重可复现（见 _load_weight_state_dict）
        torch.manual_seed(42)
        # 自建一条 stream 并设为当前 stream：权重加载、KV 分配、抓图都排在这条流上；
        # 调度器与本类跑在同一条线程、同一条流，forward_batch 里会再 assert 一次
        self.stream = torch.cuda.Stream()
        torch.cuda.set_stream(self.stream)
        self.dtype = config.dtype
        # Context 先只填 page_size，其余字段（kv_cache / page_table / 各后端）按依赖顺序补
        self.ctx = Context(config.page_size)
        set_global_ctx(self.ctx)

        self.tp_cpu_group = self._init_communication(config)
        # 量"建模型之前"的空闲显存（各 rank 取最大），建完模型再量一次，两者之差就是模型占用
        init_free_memory = self._sync_get_memory()[1]
        logger.info_rank0(f"Free memory before loading model: {mem_GB(init_free_memory)}")

        # ======================= Model initialization ========================
        set_rope_device(self.device)
        # meta 设备上建"空骨架"：只有形状和 dtype，不占显存，省掉"先分配再覆盖"的峰值；
        # 紧接着 load 真权重 —— 注意模型不是 nn.Module，而是 layers/base.py 里那套
        # 自定义 BaseOP 树（张量直接存在 __dict__），它的 load_state_dict 是 setattr
        # 直接换张量，而不是往 meta 张量里 copy_，所以骨架能被整块替换掉
        with torch.device("meta"), torch_dtype(config.dtype):
            self.model = create_model(config.model_config)
        self.model.load_state_dict(self._load_weight_state_dict(config))

        # ======================= KV cache initialization ========================
        self.num_pages = self._determine_num_pages(init_free_memory, config)
        num_tokens = self.num_pages * config.page_size
        # 池子按 num_pages + 1 页分配：多出来的一页是 dummy 页，供补位请求和抓图读写，
        # 免得占位请求踩到真实数据（页表最后一行整行都指向它，见下面的 dummy_req）
        self.ctx.kv_cache = self.kv_cache = create_kvcache_pool(
            model_config=config.model_config,
            num_pages=self.num_pages + 1,  # +1 for dummy page
            page_size=config.page_size,
            device=self.device,
            dtype=self.dtype,
        )

        # ======================= Page table initialization ========================
        # NOTE: 1. aligned to 128 bytes; 2. store raw locations instead of pages
        # max_seq_len 不能超过 KV 池的容量（一共就 num_tokens 个槽位）；
        # _align_up_32：int32 下 32 个元素正好 128 字节，对齐到 cache line
        self.max_seq_len = min(config.max_seq_len, num_tokens)
        aligned_max_seq_len = _align_up_32(self.max_seq_len)
        self.ctx.page_table = self.page_table = torch.zeros(  # + 1 for dummy request
            (config.max_running_req + 1, aligned_max_seq_len),
            dtype=torch.int32,
            device=self.device,
        )

        # ======================= Attention & MoE backend initialization ========================
        # 后端要读 ctx（页表 / KV 池 / page_size 都在里面），所以要排在它们之后建
        self.ctx.attn_backend = self.attn_backend = create_attention_backend(
            config.attention_backend, config.model_config
        )
        # 非 MoE 模型不建，ctx.moe_backend 保持未设置
        if config.model_config.is_moe:
            self.ctx.moe_backend = self.moe_backend = create_moe_backend(config.moe_backend)

        # ======================= Sampler initialization ========================
        self.sampler = Sampler(self.device, config.model_config.vocab_size)

        post_free_memory = self._sync_get_memory()[0]
        logger.info_rank0(f"Free memory after initialization: {mem_GB(post_free_memory)}")

        # ======================= Graph capture initialization ========================
        # dummy 请求：CUDA graph 要求固定的 batch 形状，真请求不够就用它补位（graph.pad_batch）。
        # 它的 table_idx 是最后一行（TableManager 不分配这一行），uid = -1 是个不会被真实
        # 请求用到的哨兵值；采样参数和缓存句柄都用不上，所以是 None
        self.dummy_req = Req(
            input_ids=torch.tensor([0], dtype=torch.int32, device="cpu"),
            table_idx=config.max_running_req,
            cached_len=0,
            output_len=1,
            uid=-1,
            sampling_params=None,  # type: ignore
            cache_handle=None,  # type: ignore
        )
        # 整个 dummy 行都指向 dummy 页：补位请求的 attention 会真按这行去读 KV，
        # 让它落在那张专设的页上（池子多分配的那一页），不会踩到别人的数据
        self.page_table[self.dummy_req.table_idx].fill_(num_tokens)  # point to dummy page
        self.graph_runner = GraphRunner(
            stream=self.stream,
            device=self.device,
            model=self.model,
            attn_backend=self.attn_backend,
            cuda_graph_bs=config.cuda_graph_bs,
            cuda_graph_max_bs=config.cuda_graph_max_bs,
            free_memory=init_free_memory,
            max_seq_len=aligned_max_seq_len,
            vocab_size=config.model_config.vocab_size,
            dummy_req=self.dummy_req,
        )

    def _init_communication(self, config: EngineConfig) -> torch.distributed.ProcessGroup:
        """建进程组、选定 TP 通信实现，返回**给 CPU 侧协调用的那个 gloo 组**。

        两种分支：
          - tp=1，或开了 pynccl：进程组用 gloo，allreduce 交给 pynccl（自研实现）。
            max_bytes 是它的通信缓冲要覆盖的最大单次通信量 —— 一次 forward 里最大的一笔
            是 [max_forward_len, hidden_size] 的残差/lm_head allreduce；
          - 否则：进程组用 nccl（模型内部的通信走它），另外单独建一个 gloo 组。
        返回值一律是 gloo 组：_sync_get_memory 要在 CPU 张量上做 all_reduce，
        nccl 不能直接 reduce CPU 张量，所以哪怕主干走 nccl 也得留一个 gloo 组。
        """
        if config.tp_info.size == 1 or config.use_pynccl:
            torch.distributed.init_process_group(
                backend="gloo",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            tp_cpu_group = torch.distributed.group.WORLD
            assert tp_cpu_group is not None
            max_bytes = (
                config.max_forward_len * config.model_config.hidden_size * self.dtype.itemsize
            )
            enable_pynccl_distributed(config.tp_info, tp_cpu_group, max_bytes)
        else:
            torch.distributed.init_process_group(
                backend="nccl",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            tp_cpu_group = torch.distributed.new_group(backend="gloo")
            assert tp_cpu_group is not None
        return tp_cpu_group

    def _load_weight_state_dict(self, config: EngineConfig) -> Dict[str, torch.Tensor]:
        """权重的真正来源。

        use_dummy_weight 时用随机数顶替（跑吞吐/显存基准用，省掉下载和读盘）；
        否则从 safetensors 流式加载（load_weight 会边读边切分 TP 分片），顺带转成目标 dtype。
        两种来源的键都取自模型的 state_dict，所以形状一定对得上。
        """
        if config.use_dummy_weight:
            return {
                k: torch.randn_like(v, device=self.device)
                for k, v in self.model.state_dict().items()
            }
        else:
            return {k: v.to(self.dtype) for k, v in load_weight(config.model_path, self.device)}

    def _determine_num_pages(self, old_free_memory: int, config: EngineConfig) -> int:
        """算 KV 池能开几页。

        单页开销 = K、V 两份 × head_dim × 本 rank 分到的 kv 头数 × page_size × dtype 字节
        × 层数。可用显存 = memory_ratio × 建模型前的空闲显存 - 模型占用，其余全给 KV。
        """
        new_free_memory = self._sync_get_memory()[1]
        cache_per_page = (
            2  # key + value
            * config.model_config.head_dim
            * div_even(config.model_config.num_kv_heads, config.tp_info.size, allow_replicate=True)
            * config.page_size
            * self.dtype.itemsize
            * config.model_config.num_layers
        )
        # --num-pages 直接指定了页数就用它（调试/复现时固定容量）
        num_pages = config.num_page_override
        if num_pages is None:
            # 模型占了多少 = 建模型前的空闲 - 现在的空闲
            model_memory = old_free_memory - new_free_memory
            available_memory = int(config.memory_ratio * old_free_memory) - model_memory
            num_pages = available_memory // cache_per_page

        assert num_pages > 1, "Not enough memory for KV cache, try reducing --num-pages"
        num_tokens = num_pages * config.page_size
        real_kv_size = num_pages * cache_per_page
        logger.info(f"Allocating {num_tokens} tokens for KV cache, K + V = {mem_GB(real_kv_size)}")
        return num_pages

    def _sync_get_memory(self) -> Tuple[int, int]:
        """Get the min and max free memory across TP ranks."""
        # 中文要点：先同步 + empty_cache，把分配器缓存里没用的块还给驱动，量到的才是真空闲；
        # 然后一次 all_reduce 同时求出 min 和 max（把 free 和 -free 拼起来取 MIN 的巧劲）。
        # 各 rank 空闲显存相差超过 2 GiB 直接报错：后面的 KV 容量是按"最宽裕的卡"估的，
        # 差太多会让最紧的卡先 OOM，不如早点失败。
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        free_memory = get_free_memory(self.device)
        free_mem_tensor = torch.tensor([free_memory, -free_memory], device="cpu", dtype=torch.int64)
        torch.distributed.all_reduce(
            free_mem_tensor, op=torch.distributed.ReduceOp.MIN, group=self.tp_cpu_group
        )
        min_free_memory = int(free_mem_tensor[0].item())
        max_free_memory = -int(free_mem_tensor[1].item())
        if max_free_memory - min_free_memory > 2 * 1024 * 1024 * 1024:
            logger.error(
                f"Memory across TP ranks are imbalanced:"
                f" min {mem_GB(min_free_memory)}, max {mem_GB(max_free_memory)}"
            )
            raise RuntimeError("Memory across TP ranks are imbalanced")

        return min_free_memory, max_free_memory

    def forward_batch(self, batch: Batch, args: BatchSamplingArgs) -> ForwardOutput:
        """跑一次 forward：进 batch，出采样到的 token。调度器每一轮都会调它。

        没走 CUDA graph 时这是唯一真正"跑模型"的地方，注意它不返回 logits 本身，
        只返回采样结果 —— 全量 logits 太大，采样完就没用了。
        """
        # 调度器和 Engine 在同一条线程、同一条流上；换流会让 graph replay 的时序乱掉
        assert torch.cuda.current_stream() == self.stream
        # 把 batch 挂到全局上下文上（core.py 的 Context.forward_batch），此后模型各层
        # 通过 get_global_ctx().batch 就能拿到"当前这一批"；退出即清空，不允许嵌套
        with self.ctx.forward_batch(batch):
            # 形状命中已抓的图就 replay（只重放 kernel，省掉 Python 侧的元数据拼装和启动开销）；
            # 否则退回 eager 执行（prefill 及各后端的特殊形状走这里）
            if self.graph_runner.can_use_cuda_graph(batch):
                logits = self.graph_runner.replay(batch)
            else:
                logits = self.model.forward()

        # 把每个请求往前推一格（cached_len 追平 device_len，device_len 再 +1）。
        # 必须排在 forward 之后：此时 [0, device_len) 的 KV 才真的写进了 cache。
        # 只遍历 batch.reqs —— 补位的 dummy_req 在 padded_reqs 里，不算真请求
        for req in batch.reqs:
            req.complete_one()

        # 只取前 batch.size 行：padding 补出来的那些行算的是 dummy 请求，logits 是垃圾，丢掉；
        # token id 统一成 int32（后面要 scatter 进 token_pool，那张表就是 int32）
        next_tokens_gpu = self.sampler.sample(logits[: batch.size], args).to(torch.int32)
        # 拷回主机：non_blocking 是因为目标是 pinned 内存，这份拷贝挂在流上异步进行；
        # 调度器要用 CPU 上的 token id 去 detokenize、判 EOS/长度上限，所以必须落回主机
        next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)
        # 记一个事件，告诉调度器"这次 D2H 拷完了"，由它决定什么时候 synchronize
        # （见 Scheduler._process_last_data 开头的 copy_done.synchronize()）
        copy_done_event = torch.cuda.Event()
        copy_done_event.record(self.stream)
        return ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)

    def shutdown(self) -> None:
        """先拆 CUDA graph 再拆进程组 —— 反过来会挂住（graph 里带着通信资源的引用，
        见 graph.py 里 destroy_cuda_graphs 的 NOTE）。"""
        self.graph_runner.destroy_cuda_graphs()
        torch.distributed.destroy_process_group()
        destroy_distributed()


def _align_up_32(num: int) -> int:
    """向上对齐到 32 的倍数：页表是 int32，32 个元素正好 128 字节，对齐到 cache line。"""
    return (num + 31) // 32 * 32


def _adjust_config(config: EngineConfig):
    """把 "auto" 解析成具体后端名，并做必要的强制覆盖。

    必须在建 Context 之前调用 —— 它可能改掉 page_size，而 page_size 会被 Context
    和各个 attention 后端读走，改晚了就不生效（或前后不一致）。
    """

    def override(attr: str, value: Any):  # this is dangerous, use with caution
        # 绕过 dataclass 的常规赋值（不做校验、不触发 __setattr__），所以才叫 dangerous
        object.__setattr__(config, attr, value)

    if config.attention_backend == "auto":
        # 按卡选：sm100 → trtllm；sm90 → "fa,fi"（逗号是 P/D 混合：prefill 用 fa、
        # decode 用 fi，由 create_attention_backend 拆成 HybridBackend）；其余 → fi
        backend = "trtllm" if is_sm100_supported() else ("fa,fi" if is_sm90_supported() else "fi")
        override("attention_backend", backend)
        logger.info_rank0(f"Auto-selected attention backend: {config.attention_backend}")

    # trtllm 的 kernel 只认这几种页大小，用户设了别的会被强行拨到 64
    # （注意这会连带改变 KV 的物理布局：页表里每 page_size 个 token 共用一个连续块）
    if "trtllm" in config.attention_backend and config.page_size not in [16, 32, 64]:
        override("page_size", 64)
        logger.warning_rank0("Page size is overridden to 64 for TRTLLM backend")

    if config.model_config.is_moe and config.moe_backend == "auto":
        override("moe_backend", "fused")
        logger.info_rank0(f"Auto-selected MoE backend: {config.moe_backend}")
