"""attention 层的"前半段"：拿到 qkv 拼接张量后，切分 → （可选）QK-Norm → RoPE → 交给后端。

真正的 attention 计算和 KV 写入都在 ctx.attn_backend 里（见 attention/ 目录），
这里只负责把张量整理成后端要的形状。

本类没有可学习权重（权重在模型定义那边的 qkv_proj / o_proj 上），所以是 StateLessOP：
    self_attn.qkv_proj : LinearQKVMerged         → 算出这个 qkv
    self_attn.attn     : AttentionLayer（本类）  → 切分 + 位置编码 + 调后端
    self_attn.o_proj   : LinearOProj             → 把本类输出投影回 hidden
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from minisgl.core import get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.utils import div_even

from .base import StateLessOP
from .rotary import get_rope

if TYPE_CHECKING:
    from minisgl.layers import RMSNorm
    from minisgl.models import RotaryConfig


class AttentionLayer(StateLessOP):
    def __init__(
        self,
        layer_id: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rotary_config: RotaryConfig,
        q_norm: RMSNorm | None = None,
        k_norm: RMSNorm | None = None,
    ):
        # GQA 的前提：q 头数是 kv 头数的整数倍（多个 q 头共用一个 kv 头）
        assert num_qo_heads % num_kv_heads == 0
        self.layer_id = layer_id
        self.head_dim = head_dim
        tp_size = get_tp_info().size
        # 只记本 rank 的头数；qkv_proj 也是按这个切法切的，两边必须一致
        self.num_qo_heads = div_even(num_qo_heads, tp_size)
        self.num_kv_heads = div_even(num_kv_heads, tp_size, allow_replicate=True)
        # split 用的宽度（也是 o_proj 的输入宽度）
        self.qo_attn_dim = self.num_qo_heads * head_dim
        self.kv_attn_dim = self.num_kv_heads * head_dim
        # 各层参数一样 → get_rope 的 cache 命中，所有层共用同一份 cos/sin 表；
        # 传进去的 scaling 转成元组只是为了可哈希（见 rotary.get_rope）
        self.rotary = get_rope(
            head_dim=head_dim,
            rotary_dim=rotary_config.rotary_dim,
            max_position=rotary_config.max_position,
            base=rotary_config.base,
            rope_scaling=tuple(rotary_config.scaling.items()) if rotary_config.scaling else None,
        )
        # QK-Norm（Qwen3 等）：None 表示这个模型不用
        self.q_norm = q_norm
        self.k_norm = k_norm

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        """qkv: [T, qo_dim + 2*kv_dim]，T = 本批所有待算 token 数（扁平，不分请求）。"""
        ctx = get_global_ctx()
        # 切分顺序 q / k / v 是**约定**：权重加载时也必须按这个顺序拼（见 LinearQKVMerged）
        q, k, v = qkv.split([self.qo_attn_dim, self.kv_attn_dim, self.kv_attn_dim], dim=-1)
        if self.q_norm is not None:
            # forward_inplace 原地改；view 出来的是视图，所以改的就是 q 本身。
            # 注意顺序：**先 norm 再 rope**（反过来位置编码的数值就不对了）
            self.q_norm.forward_inplace(q.view(-1, self.num_qo_heads, self.head_dim))
        if self.k_norm is not None:
            self.k_norm.forward_inplace(k.view(-1, self.num_kv_heads, self.head_dim))
        # RoPE 也是原地：按 batch.positions 取 cos/sin，直接旋转 q、k
        q, k = self.rotary.forward(ctx.batch.positions, q, k)
        # q 要交给 attention kernel，用 3D；k/v 保持 2D —— 它们接下来只被 store_kv 逐 token
        # 写进 KV cache（attention 读的是 cache 里的 K/V，不是这里的入参）
        q = q.view(-1, self.num_qo_heads, self.head_dim)
        o = ctx.attn_backend.forward(q, k, v, self.layer_id, ctx.batch)
        # 摊平回 2D，接给 o_proj（本 rank 的输出宽度就是 qo_attn_dim）
        return o.view(-1, self.qo_attn_dim)
