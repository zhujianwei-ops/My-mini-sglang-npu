"""Gated MLP 用的激活函数：silu_and_mul / gelu_and_mul。

这两个都是同一种融合：输入最后一维是 2×intermediate（gate、up 两路拼在一起，
由 LinearColParallelMerged 一次矩阵乘算出来），输出是
    out = act(gate) * up            （前一半做激活，后一半乘上去）
flashinfer 的 kernel 一次做完"切两半 + 激活 + 相乘"，省掉中间的临时张量。

`out=` 参数传已有缓冲时结果原地写入，decode 这种小张量、高频调用的场景
可以省掉一次分配（对 CUDA graph 也更友好：地址固定）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def silu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None):
    """SiLU 门控：silu(gate) * up —— Llama / Qwen 系的 MLP 都用这个。"""
    from flashinfer import silu_and_mul

    return silu_and_mul(x, out=out)


def gelu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None):
    """GELU 门控：gelu(gate) * up —— 给用 GELU 的模型（如 Gemma）留的。"""
    from flashinfer import gelu_and_mul

    return gelu_and_mul(x, out=out)


__all__ = ["silu_and_mul", "gelu_and_mul"]
