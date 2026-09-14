"""RoPE（旋转位置编码）。

分两块：

    查表        每个位置、每个频率的 cos/sin 预先算好，存成 _cos_sin_cache（一张
                [max_position, head_dim] 的表），forward 时按 positions 取行即可；
    频率修正    长上下文外推（llama3 / yarn）不动 cos/sin 的算法，只改 inv_freq
                （也就是"每个维度转多快"），所以可以在建表前用一个 post_process 钩子改掉。

RoPE 是**原地**做的（flashinfer 的 inplace kernel），直接改 q/k 张量，不产生新张量。
"""

from __future__ import annotations

import functools
import math
from typing import Any, Callable, Dict, Tuple

import torch

from .base import StateLessOP


class RotaryEmbedding(StateLessOP):
    """位置无关的表 + 一个按 positions 取行的 kernel。所以它没有可学习的参数
    （StateLessOP），但确实有一份张量缓存。"""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        post_process: None | Callable[[torch.Tensor], torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        # 只支持"整个 head 都转"；部分旋转（如 GPT-NeoX 的 partial rotary）没实现
        assert rotary_dim == head_size
        # inv_freq[j] = base^(-2j/d)，j = 0..d/2-1。arange(0, d, 2)/d 正好给出 2j/d
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        # 长上下文外推就在这里改频率（见 _get_rope 里的 llama3 / yarn 两个 post_process）
        if post_process is not None:
            inv_freq = post_process(inv_freq)
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        # 外积：freqs[i, j] = i * inv_freq[j] —— 位置 i 在频率 j 上的角度
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        # buffer, so don't load/save
        # 下划线开头 → BaseOP.state_dict/load_state_dict 会跳过它：这是运行时缓存，
        # 不是权重（权重文件里当然也没有它）
        self._cos_sin_cache = torch.cat((cos, sin), dim=-1)
        # flashinfer 的 rope kernel 只支持这几种 head_size（向量化路数的限制）
        assert self.head_size in [64, 128, 256, 512]

        from flashinfer import apply_rope_with_cos_sin_cache_inplace

        # 函数对象存在实例属性里；state_dict 只收 Tensor/BaseOP，所以不会被误收成权重
        self.apply_rope_with_cos_sin_cache_inplace = apply_rope_with_cos_sin_cache_inplace

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """按 positions 逐行取 cos/sin 并原地旋转 q、k。

        positions 是一维的绝对位置（batch.positions，每个 token 一个）——
        prefill 命中前缀缓存时它从 cached_len 起算，这正是 batch 要带 positions 的原因。
        """
        self.apply_rope_with_cos_sin_cache_inplace(
            positions=positions,
            query=query,
            key=key,
            head_size=self.head_size,
            cos_sin_cache=self._cos_sin_cache,
        )
        # 原地改完了，把同一个张量传回去只是方便链式调用
        return query, key


def _get_rope(
    head_dim: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: Dict[str, Any] | None = None,
) -> RotaryEmbedding:
    """按 rope_scaling 配置分派：改动都体现在传给 RotaryEmbedding 的 post_process 上。"""
    if rope_scaling is None:
        return RotaryEmbedding(head_dim, rotary_dim, max_position, base)
    # need to test some cases:
    match rope_scaling["rope_type"]:
        case "default":
            return RotaryEmbedding(head_dim, rotary_dim, max_position, base)

        case "llama3":
            # Llama 3.x 的外推：按波长分三段处理高频/低频维度，中间线性过渡 ——
            # 高频维度（波长短）不动，低频维度整体除以 scaling_factor
            scaling_factor: float = rope_scaling["factor"]
            low_freq_factor: float = rope_scaling["low_freq_factor"]
            high_freq_factor: float = rope_scaling["high_freq_factor"]
            original_max_position: int = rope_scaling["original_max_position_embeddings"]

            def post_process(inv_freq: torch.Tensor) -> torch.Tensor:
                # no smooth if low_freq_factor == high_freq_factor
                wave_len = 2 * math.pi / inv_freq
                if low_freq_factor == high_freq_factor:
                    # 不设过渡带：硬切换（短波长的保持原频率，长波长的整体缩放）
                    return torch.where(
                        wave_len < original_max_position / high_freq_factor,
                        inv_freq,
                        inv_freq / scaling_factor,
                    )

                # 过渡带：[low_freq_factor, high_freq_factor] 之间线性混合两种频率
                delta = high_freq_factor - low_freq_factor
                smooth = (original_max_position / wave_len - low_freq_factor) / delta
                smooth = torch.clamp(smooth, 0, 1)
                factor = (1 - smooth) / scaling_factor + smooth
                return factor * inv_freq

            return RotaryEmbedding(head_dim, rotary_dim, max_position, base, post_process)

        case "yarn":
            # YaRN 的 "NTK-by-parts"：低频维度按 1/factor 压缩，高频维度不动，
            # 中间用 ramp 线性过渡。beta_fast/beta_slow 决定过渡带的两端落在哪个维度上
            factor: float = rope_scaling["factor"]
            beta_fast: float = rope_scaling.get("beta_fast", 32.0)
            beta_slow: float = rope_scaling.get("beta_slow", 1.0)
            orig_max_pos: int = rope_scaling["original_max_position_embeddings"]

            def _find_correction_dim(num_rotations: float) -> float:
                # 反解"在原始上下文长度内正好转过 num_rotations 圈"的维度下标
                return rotary_dim * math.log(orig_max_pos / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

            low = max(math.floor(_find_correction_dim(beta_fast)), 0)
            high = min(math.ceil(_find_correction_dim(beta_slow)), rotary_dim // 2 - 1)

            def post_process(inv_freq: torch.Tensor) -> torch.Tensor:
                # ramp=0（<=low，高频）保持原样；ramp=1（>=high，低频）除以 factor
                ramp = torch.clamp(
                    (torch.arange(rotary_dim // 2, dtype=torch.float32) - low) / max(high - low, 1),
                    0, 1,
                )
                return (inv_freq / factor) * ramp + inv_freq * (1 - ramp)

            return RotaryEmbedding(head_dim, rotary_dim, max_position, base, post_process)

    raise ValueError(f"Unsupported {rope_scaling = }")


_ROPE_DEVICE: torch.device | None = None


def set_rope_device(device: torch.device):
    """记下"真设备"，供 get_rope 在 meta 上下文里绕开 meta 设备（见下）。

    Engine 在建模型之前调用（Engine.__init__ 里那句 set_rope_device(self.device)）。
    """
    global _ROPE_DEVICE
    _ROPE_DEVICE = device


@functools.cache
def get_rope(
    head_dim: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: Tuple[Tuple[str, Any], ...] | None = None,
) -> RotaryEmbedding:
    """工厂 + 缓存：同一组参数只建一份（cos/sin 表不便宜，且各层可以共用）。

    rope_scaling 收成"元组的元组"只是为了可哈希（dict 不能当 cache 的键），
    进来马上转回 dict 给 _get_rope。
    """
    rope_map = dict(rope_scaling) if rope_scaling is not None else None
    # 探针：torch.tensor([]) 会落在"当前默认设备"上，用它的 device 判断我们是不是正处在
    # `with torch.device("meta")` 里（Engine 建模型时就是）
    t = torch.tensor([])
    if t.device == torch.device("meta"):
        # we cannot use meta device for rope
        # meta 上建出来的 cos/sin 表没有真实数据，模型一跑就会炸；而且这份对象会被
        # functools.cache 记住，所以必须当场就在真设备上建好
        if _ROPE_DEVICE is None:
            raise RuntimeError(
                "We cannot use meta device for rope. Please call set_rope_device() first."
            )
        with torch.device(_ROPE_DEVICE):
            return _get_rope(head_dim, rotary_dim, max_position, base, rope_map)
    return _get_rope(head_dim, rotary_dim, max_position, base, rope_map)


__all__ = ["get_rope", "RotaryEmbedding", "set_rope_device"]
