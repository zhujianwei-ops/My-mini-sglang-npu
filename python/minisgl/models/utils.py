"""各模型共用的三块积木：GatedMLP / MoEMLP / RopeAttn。

各模型的 decoder layer 基本就是"norm + RopeAttn + norm + MLP"，差异只在参数
（有没有 bias、有没有 QK-Norm、稠密还是 MoE），所以把这些组合抽出来复用。
注意 forward 里那些 `del` 不是摆设：Python 的引用计数会立刻回收大张量，
decode 时张量小、调用频繁，早一步释放能实打实降低显存峰值。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from minisgl.layers import (
    AttentionLayer,
    BaseOP,
    LinearColParallelMerged,
    LinearOProj,
    LinearQKVMerged,
    LinearReplicated,
    LinearRowParallel,
    MoELayer,
    RMSNorm,
    gelu_and_mul,
    silu_and_mul,
)
from minisgl.models import ModelConfig
from minisgl.utils import nvtx_annotate

if TYPE_CHECKING:
    import torch


class GatedMLP(BaseOP):
    """稠密 MLP：gate/up 一次算完，激活后乘起来，再 down 投影。

        x → gate_up_proj（列并行，输出 2×inter）→ act_fn（切两半相乘）
          → down_proj（行并行，末尾 all_reduce）→ 出
    """

    def __init__(self, config: ModelConfig):
        # 两路合并成一次矩阵乘：[inter, inter] 表示 gate 和 up 各占一半输出宽度
        self.gate_up_proj = LinearColParallelMerged(
            config.hidden_size,
            [config.intermediate_size, config.intermediate_size],
            has_bias=False,
        )

        # 激活函数按配置选；这两个函数都是融合 kernel（切两半 + 激活 + 相乘）
        FN_MAP = {"silu": silu_and_mul, "gelu": gelu_and_mul}
        act_fn = FN_MAP.get(config.hidden_act, None)
        if act_fn is None:
            raise ValueError(f"Unsupported activation function: {config.hidden_act}")
        # 函数属性：state_dict 只收 Tensor/BaseOP，所以不会被当成权重
        self.act_fn = act_fn
        # 行并行：输入是上面那个被切过的中间维，所以末尾要 all_reduce
        self.down_proj = LinearRowParallel(
            config.intermediate_size,
            config.hidden_size,
            has_bias=False,
        )

    @nvtx_annotate("MLP")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj.forward(x)  # [T, 2*inter_local]
        del x  # 后面用不到了，立刻还显存
        y = self.act_fn(gate_up)  # [T, inter_local]
        del gate_up
        return self.down_proj.forward(y)


class MoEMLP(BaseOP):
    """MoE 版 MLP：路由用的 gate + 打包好的专家。

    路由**不能切**：每个 rank 都必须算出完整的路由（不然没法知道本地该处理哪些
    token），所以 gate 用 LinearReplicated —— 复制的那份权重，不通信。
    """

    def __init__(self, config: ModelConfig):
        self.experts = MoELayer(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
        )
        # hidden → num_experts 的打分矩阵；复制式，每 rank 都有全量
        self.gate = LinearReplicated(
            config.hidden_size,
            config.num_experts,
            has_bias=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 上游本来就是 2D，这里改成显式的 (-1, hidden) 只是把"后端要 2D"这件事写清楚
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate.forward(hidden_states)
        final_hidden_states = self.experts.forward(
            hidden_states=hidden_states, router_logits=router_logits
        )
        final_hidden_states = final_hidden_states.view(num_tokens, hidden_dim)
        return final_hidden_states


class RopeAttn(BaseOP):
    """attention 的三件套：qkv 投影 → AttentionLayer（切分 + 位置编码 + 后端）→ o_proj。

    cfg 差异靠两个关键字参数区分：
        has_attn_bias   QKV 有没有 bias（Qwen2 有，Llama 没有）
        has_qk_norm     Qwen3 的 QK-Norm（对每个 head 的 head_dim 做 RMSNorm）
    """

    def __init__(
        self,
        config: ModelConfig,
        layer_id: int,
        *,
        has_attn_bias: bool = False,
        has_qk_norm: bool = False,
    ):
        head_dim = config.head_dim
        self.qkv_proj = LinearQKVMerged(
            hidden_size=config.hidden_size,
            head_dim=config.head_dim,
            num_qo_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            has_bias=has_attn_bias,
        )
        self.has_qk_norm = has_qk_norm
        if has_qk_norm:
            # 注意这里用的是**非融合**的 RMSNorm：作用在 head_dim 这一维上，
            # 不是主干残差那种"加 residual + norm"的场景
            self.q_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
        else:
            # 不用也得把属性置成 None：state_dict 遍历 __dict__ 时会看到它们，
            # 值是 None 不是张量，会被跳过
            self.q_norm = None
            self.k_norm = None
        self.attn = AttentionLayer(
            layer_id=layer_id,
            head_dim=head_dim,
            num_qo_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            rotary_config=config.rotary_config,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
        )
        # 输入宽度按**全局** q 头数算：LinearOProj 内部会 div_even 切成本 rank 的
        self.o_proj = LinearOProj(
            head_dim * config.num_qo_heads,
            config.hidden_size,
            has_bias=False,
        )

    @nvtx_annotate("MHA")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv_proj.forward(x)
        del x
        o = self.attn.forward(qkv)  # 这一层里就写进 KV cache 了
        return self.o_proj.forward(o)


__all__ = ["GatedMLP", "RopeAttn", "MoEMLP"]
