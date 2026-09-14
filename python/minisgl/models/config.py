"""把 HuggingFace 的 PretrainedConfig 拍扁成本仓库真正要用的字段。

HF 的 config 有两个麻烦，from_hf 主要在对付它们：
  1. 同一件事在不同模型里字段名不同（num_local_experts vs num_experts、
     rope_theta 在顶层还是在 rope_scaling 里……），只能逐个 getattr 兜底；
  2. 多模态模型把文本配置藏在 text_config 里，要拆出来用。
拍扁之后，模型结构代码只读 ModelConfig，不必关心这些差异。

frozen=True：建完之后不可变（Engine 的 _adjust_config 要改配置时得用
object.__setattr__ 硬来，见那处注释）。
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict
from transformers import PretrainedConfig


@dataclass(frozen=True)
class RotaryConfig:
    """RoPE 的全部参数，直接喂给 layers/rotary.py 的 get_rope。"""

    head_dim: int
    rotary_dim: int
    max_position: int
    base: float  # 就是 rope_theta
    scaling: Dict[str, Any] | None  # 外推配置（llama3 / yarn），None 表示不做外推


@dataclass(frozen=True)
class ModelConfig:
    """模型结构的全部参数。各字段与 HF config 的对应关系见 from_hf。"""

    num_layers: int
    num_qo_heads: int
    num_kv_heads: int  # 比 num_qo_heads 小就是 GQA
    head_dim: int
    hidden_size: int
    vocab_size: int
    intermediate_size: int
    rms_norm_eps: float
    rotary_config: RotaryConfig
    hidden_act: str
    tie_word_embeddings: bool  # lm_head 与 embedding 共享权重
    num_experts: int  # 0 表示不是 MoE
    num_experts_per_tok: int  # top_k
    moe_intermediate_size: int
    norm_topk_prob: bool  # MoE 路由权重是否归一化
    model_type: str
    architectures: list[str]  # 取第 0 个用来查注册表

    @property
    def is_moe(self) -> bool:
        # 靠 model_type 里有没有 "moe" 判断（如 "qwen3_moe"、"mixtral"）
        return "moe" in self.model_type

    @classmethod
    def from_hf(cls, config: PretrainedConfig) -> ModelConfig:
        # 多模态模型（如 Mistral3）：真正的文本配置在 text_config 里，
        # 换过去之后再把顶层特有的几个字段补回来（这些恰恰常常只写在外层）
        if hasattr(config, "text_config") and config.text_config is not None:
            top = config
            config = config.text_config
            for attr in ("architectures", "rope_theta", "rope_scaling"):
                if not getattr(config, attr, None) and getattr(top, attr, None):
                    setattr(config, attr, getattr(top, attr))

        # ---- 以下全是"字段名/缺省值"的兜底 ----
        # 没写 num_key_value_heads 就是 MHA（kv 头数 = q 头数）
        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        # 没写 head_dim 就按 hidden / heads 推（有的模型 head_dim ≠ hidden/heads）
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        tie_word_embeddings = getattr(config, "tie_word_embeddings", False)
        model_type = getattr(config, "model_type", "llama")
        # Mixtral 叫 num_local_experts，Qwen3-MoE 叫 num_experts
        num_experts = getattr(config, "num_local_experts", getattr(config, "num_experts", 0))
        num_experts_per_tok = getattr(config, "num_experts_per_tok", 0)
        moe_intermediate_size = getattr(config, "moe_intermediate_size", 0)
        norm_topk_prob = getattr(config, "norm_topk_prob", False)
        architectures = getattr(config, "architectures", ["LlamaForCausalLM"])

        # Llama/Qwen: rope_theta is a direct attr; Mistral: it's inside rope_scaling dict
        # 注意后一种写法下两边都没有就会 KeyError —— 对 Mistral 系这是硬要求
        rope_scaling = getattr(config, "rope_scaling", None)
        rope_theta = getattr(config, "rope_theta", None) or rope_scaling["rope_theta"]

        return cls(
            num_layers=config.num_hidden_layers,
            num_qo_heads=config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            rms_norm_eps=config.rms_norm_eps,
            tie_word_embeddings=tie_word_embeddings,
            rotary_config=RotaryConfig(
                head_dim=head_dim,
                # 只支持全旋转，所以 rotary_dim 恒等于 head_dim（rotary.py 里有 assert）
                rotary_dim=head_dim,
                max_position=config.max_position_embeddings,
                base=rope_theta,
                scaling=rope_scaling,
            ),
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_intermediate_size=moe_intermediate_size,
            norm_topk_prob=norm_topk_prob,
            model_type=model_type,
            architectures=architectures,
        )
