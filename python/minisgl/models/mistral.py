"""Mistral 模型：结构与 llama.py 逐行相同，差异全在 config 那一层：

    rope_theta 在 Mistral 的 HF config 里是放在 rope_scaling 字典里的
    （config.py 里专门为此写了 `or rope_scaling["rope_theta"]` 的兜底）。

另外 ModelConfig 里没有 sliding_window 之类的字段，所以模型体这边拿到的
旋转/注意力参数与 Llama 完全一样。

结构约定（残差流怎么走）见 llama.py 的模块注释，这里不重复。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
from minisgl.core import get_global_ctx
from minisgl.layers import BaseOP, OPList, ParallelLMHead, RMSNormFused, VocabParallelEmbedding
from minisgl.utils import nvtx_annotate

from .base import BaseLLMModel
from .utils import GatedMLP as MistralMLP
from .utils import RopeAttn as MistralAttn

if TYPE_CHECKING:
    from .config import ModelConfig


class MistralDecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        # 用默认参数：无 bias、无 QK-Norm —— 与 Llama 一致
        self.self_attn = MistralAttn(config, layer_id)
        self.mlp = MistralMLP(config)
        self.input_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

        # 下划线开头：不算权重；只用来在 profiler 里区分第几层
        self._layer_id = layer_id

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 四步：norm（并把 x 并进残差流）→ attention → norm → MLP；返回"增量 + 残差流"
        x, residual = self.input_layernorm.forward(x, residual)
        x = self.self_attn.forward(x)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        x = self.mlp.forward(x)
        return x, residual


class MistralModel(BaseOP):
    """embedding + 全部层 + 末尾 norm。"""

    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [MistralDecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids 是扁平的一维 token 序列（本批所有待算 token 拼在一起）
        x = self.embed_tokens.forward(input_ids)
        residual: torch.Tensor | None = None  # None 表示还没有残差流，首个 norm 直接跳过加法
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        # 末尾这次 norm 顺带把最后一层的增量并进残差流，所以只取 [0]
        return self.norm.forward(x, residual)[0]


class MistralForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = MistralModel(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        super().__init__()

    def forward(self) -> torch.Tensor:
        # 无参 forward：输入从全局 ctx 取（BaseLLMModel 的约定，见 models/base.py）。
        # 这里多绕一个 ids 局部变量，只是写法差异，和 llama.py 里的等价。
        ids = get_global_ctx().batch.input_ids
        output = self.model.forward(ids)
        logits = self.lm_head.forward(output)
        return logits


__all__ = ["MistralForCausalLM"]
