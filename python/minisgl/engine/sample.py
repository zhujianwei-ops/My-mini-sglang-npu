"""采样：把 logits 变成下一个 token id。

两条路径：
    greedy      整批都是贪心 → torch.argmax，不依赖 flashinfer
    随机采样    交给 flashinfer.sampling 的 softmax + top-k / top-p kernel

分工切成两半，是为了配合 overlap scheduling：

    Sampler.prepare()  调度器侧（CPU）每步调一次：把各请求的采样参数拼成张量并
                       异步 H2D 拷到 device —— 这些 CPU 开销被藏在上一批的计算里
    Sampler.sample()   引擎侧、接在 forward 之后：跟着 logits 一起算，必须等 logits

参数张量的顺序与 batch.reqs 一致（补位的 dummy_req 不参与），与 logits 的行一一对应。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
from minisgl.utils import is_sm90_supported, nvtx_annotate

if TYPE_CHECKING:
    from minisgl.core import Batch


@dataclass
class BatchSamplingArgs:
    """一批的采样参数（顺序与 batch.reqs 一致）。

    temperatures 为 None 是"**整批** greedy"的标志位，而不是"没有温度"：
    此时一个采样 kernel 都不用发，直接 argmax（见 Sampler.sample）。
    """

    temperatures: torch.Tensor | None
    top_k: torch.Tensor | None = None
    top_p: torch.Tensor | None = None


def make_device_tensor(data: List, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """CPU 列表 → pinned host 张量 → 异步拷到 device。

    pin_memory 是让 non_blocking 真的生效的前提：没有 pinned 内存时 .to() 会退化成
    同步拷贝（要先在 host 上等完），那就没法跟上一批的 GPU 计算重叠了。
    """
    return torch.tensor(data, dtype=dtype, pin_memory=True).to(device, non_blocking=True)


def sample_impl(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor | int | None,
    top_p: torch.Tensor | float | None,
) -> torch.Tensor:
    """随机采样的实现：先算 probs，再按有没有 top_k / top_p 选一个 kernel。

    flashinfer 把四种组合拆成了四个 kernel（都不给 / 只 top_k / 只 top_p / 两个都给），
    所以这里是一组 if：少传一个参数，就少一段 GPU 上的工作。
    """
    # 局部导入：只有真要随机采样时才需要 flashinfer（纯 greedy 的部署可以不装它）
    import flashinfer.sampling as sampling

    # 温度就是在这一步生效的：p = softmax(logits / T)。
    # enable_pdl：sm90 及以上打开 PDL（程序化依赖启动），让相邻 kernel 叠着跑
    probs = sampling.softmax(logits, temperatures, enable_pdl=is_sm90_supported())
    if top_k is None and top_p is None:
        # 不截断：直接按概率分布采一个
        return sampling.sampling_from_probs(probs)

    if top_p is None:
        assert top_k is not None
        return sampling.top_k_sampling_from_probs(probs, top_k)

    if top_k is None:
        assert top_p is not None
        return sampling.top_p_sampling_from_probs(probs, top_p)

    assert top_k is not None and top_p is not None
    return sampling.top_k_top_p_sampling_from_probs(probs, top_k, top_p)


@dataclass
class Sampler:
    """无状态：只记 device 和词表大小，参数都由调用方按批传进来。

    Engine 建一次（engine.py 里 `Sampler(self.device, vocab_size)`），
    调度器每步调 prepare()、引擎每步调 sample()。
    """

    device: torch.device
    vocab_size: int

    def prepare(self, batch: Batch) -> BatchSamplingArgs:
        # 只取**真实请求**（batch.reqs，不含补位的 dummy_req）；顺序与 logits
        # 的行一一对应 —— sample() 那边拿到的是已切掉 dummy 行的 logits
        params = [r.sampling_params for r in batch.reqs]
        if all(p.is_greedy for p in params):
            # 整批 greedy：连参数张量都不用建，sample() 直接走 argmax 分支
            return BatchSamplingArgs(temperatures=None)

        # 温度/概率的下限。对温度它还有第二个用处 —— 见下一行的注释
        MIN_P = MIN_T = 1e-6
        # 混合批里的 greedy 请求：给个极小温度近似 argmax（softmax(logits / 1e-6)
        # 尖到几乎只剩最大值那一项，效果等价于贪心），同时避免温度为 0 时除零
        ts = [max(0.0 if p.is_greedy else p.temperature, MIN_T) for p in params]
        # top_k 默认 -1 表示不截断；这里等价地写成 vocab_size，好让整批共用同一个
        # kernel，再由下面 any() 判断能不能干脆不传这个参数
        top_ks = [p.top_k if p.top_k >= 1 else self.vocab_size for p in params]
        top_ps = [min(max(p.top_p, MIN_P), 1.0) for p in params]  # 夹到 (0, 1]
        temperatures = make_device_tensor(ts, torch.float32, self.device)
        top_k, top_p = None, None
        # 只要有一个请求要截断，整批就得走带截断的 kernel（参数按行生效，互不影响）
        if any(k != self.vocab_size for k in top_ks):
            top_k = make_device_tensor(top_ks, torch.int32, self.device)
        if any(p < 1.0 for p in top_ps):
            top_p = make_device_tensor(top_ps, torch.float32, self.device)
        return BatchSamplingArgs(temperatures, top_k=top_k, top_p=top_p)

    @nvtx_annotate("Sampler")
    def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        # 注意标签写了两处（装饰器一层 + 下面这行）→ profile 里是两层同名的嵌套
        # range；不影响功能，看 profile 时别当成采了两次
        with torch.cuda.nvtx.range("Sampler"):
            if args.temperatures is None:  # greedy sampling
                # 比大小不需要精度：直接用原始 logits，不必先转 float32
                return torch.argmax(logits, dim=-1)
            # 随机采样统一转 float32：decode 的 logits 常是 bf16/fp16，而上面那个
            # 极小温度会放大数值，低精度下误差会直接改变采样结果
            return sample_impl(logits.float(), args.temperatures, args.top_k, args.top_p)
