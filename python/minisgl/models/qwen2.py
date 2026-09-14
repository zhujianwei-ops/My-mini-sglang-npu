"""Qwen2 模型：结构与 llama.py 完全相同，只有一处实质差别 ——

    QKV 投影**带 bias**（has_attn_bias=True），且没有 QK-Norm。

bias 走的还是同一套合并流程：权重文件的 q_proj.bias / k_proj.bias / v_proj.bias
会分别被切分、按 q/k/v 顺序拼成 qkv_proj.bias（合并缓冲是按**完整键名**区分的，
所以 weight 和 bias 各攒各的，互不干扰）。o_proj / down_proj 依旧没有 bias。

结构约定（残差流怎么走）见 llama.py 的模块注释，这里不重复。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
from minisgl.core import get_global_ctx
from minisgl.layers import BaseOP, OPList, ParallelLMHead, RMSNormFused, VocabParallelEmbedding
from minisgl.utils import nvtx_annotate

from .base import BaseLLMModel
from .utils import GatedMLP as Qwen2MLP
from .utils import RopeAttn as Qwen2Attn

if TYPE_CHECKING:
    from .config import ModelConfig


class Qwen2DecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        # 两个开关就是 Qwen2 与 Llama 的全部差别：带 bias、不做 QK-Norm
        self.self_attn = Qwen2Attn(config, layer_id, has_qk_norm=False, has_attn_bias=True)
        self.mlp = Qwen2MLP(config)
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
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 四步：norm（并把 x 并进残差流）→ attention → norm → MLP；返回的是"增量 + 残差流"
        x, residual = self.input_layernorm.forward(x, residual)
        x = self.self_attn.forward(x)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        x = self.mlp.forward(x)
        return x, residual


class Qwen2Model(BaseOP):
    """embedding + 全部层 + 末尾 norm。"""

    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [Qwen2DecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
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


class Qwen2ForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = Qwen2Model(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        super().__init__()

    def forward(self) -> torch.Tensor:
        # 无参 forward：输入从全局 ctx 取（BaseLLMModel 的约定，见 models/base.py）
        output = self.model.forward(get_global_ctx().batch.input_ids)
        logits = self.lm_head.forward(output)
        return logits


__all__ = ["Qwen2ForCausalLM"]
