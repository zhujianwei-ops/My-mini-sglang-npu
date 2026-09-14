"""MoE 层的权重容器 + 后端调用。

分工：
    模型定义那边   router/gate（算出 router_logits，每个 token 对每个专家的分数）
    本类           **专家权重**：所有专家的 w1/w2 打包成一个三维张量
    moe_backend    真正的路由与计算（moe/fused.py，走 sgl_kernel 的 topk + 分组 GEMM）

TP 的做法：**每个 rank 都持有全部专家**，但每个专家的中间维只切一段
（expert 维不切，所以没有专家并行；等价于把每个专家都当成一个 dense MLP 来切）。
"""

import torch
from minisgl.core import get_global_ctx
from minisgl.distributed import DistributedCommunicator, get_tp_info
from minisgl.utils import div_even

from .base import BaseOP


class MoELayer(BaseOP):
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        renormalize: bool = True,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
    ):
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        # 下划线开头 → 不会被当成权重
        self._comm = DistributedCommunicator()

        tp_info = get_tp_info()
        self.tp_size = tp_size = tp_info.size
        self.renormalize = renormalize
        self.activation = activation
        self.apply_router_weight_on_input = apply_router_weight_on_input
        # MoE 的中间维一定够大，不用 allow_replicate；切的是每个专家的中间维
        intermediate_size_per_partition = div_even(intermediate_size, tp_size)
        # 三维打包：[专家][输出][输入]。gate 和 up 两路仍然拼在第一维里
        # （2 × intermediate），形状和 dense MLP 的 gate_up 一样，只是前面多了专家维
        self.gate_up_proj = torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_size,
        )
        # down_proj 是行并行的那一半：输入被切过，输出回到 hidden
        self.down_proj = torch.empty(
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
        )

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
        """hidden_states: [T, hidden]，router_logits: [T, num_experts]（由模型的 gate 算出）。"""
        ctx = get_global_ctx()
        # 路由（topk_softmax）+ 专家计算都在后端里，本类只把打包好的权重递过去
        final_hidden_states = ctx.moe_backend.forward(
            hidden_states=hidden_states,
            w1=self.gate_up_proj,
            w2=self.down_proj,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
            activation=self.activation,
            apply_router_weight_on_input=self.apply_router_weight_on_input,
        )
        # 每个 rank 只算了自己那段中间维的部分和（和 dense MLP 的行并行同理），要合起来
        if self.tp_size > 1:
            final_hidden_states = self._comm.all_reduce(final_hidden_states)
        return final_hidden_states
