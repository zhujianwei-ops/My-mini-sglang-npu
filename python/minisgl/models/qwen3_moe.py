"""Qwen3-MoE 模型：骨架与 qwen3.py 相同，只是把 MLP 换成 MoE 版 ——

    MoEMLP = MoELayer（打包好的专家）+ 复制式的路由 gate（见 models/utils.py）

顺带两处差异：
  - 这个文件里的 Qwen3Model 与 qwen3.py 里的**只是同名**，是两个模块里各写各的
    类，不由同一个类派生（各自的 Qwen3DecoderLayer 层类型不同）；
  - 导出的是 Qwen3MoeForCausalLM，注册表里 "Qwen3MoeForCausalLM" 就指向这里。

权重侧：专家的 gate_up_proj / down_proj 在文件里是逐个专家分开的，加载时会被
weight.py 攒成 (num_experts, out, in) 的三维张量。

结构约定（残差流怎么走）见 llama.py 的模块注释，这里不重复。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
from minisgl.core import get_global_ctx
from minisgl.layers import BaseOP, OPList, ParallelLMHead, RMSNormFused, VocabParallelEmbedding
from minisgl.utils import nvtx_annotate

from .base import BaseLLMModel
from .utils import MoEMLP as Qwen3MLP
from .utils import RopeAttn as Qwen3Attn

if TYPE_CHECKING:
    from .config import ModelConfig


class Qwen3DecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        # attention 与 qwen3.py 完全一致：QK-Norm 开、无 bias
        self.self_attn = Qwen3Attn(config, layer_id, has_qk_norm=True)
        # 唯一差别：这行拿到的是 MoEMLP（专家 + 路由），不是稠密的 GatedMLP
        self.mlp = Qwen3MLP(config)
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
        # 四步：norm（并把 x 并进残差流）→ attention → norm → MLP；返回"增量 + 残差流"。
        # 对 MoE 来说"路由 + 选专家"都藏在这句 self.mlp.forward 里面。
        x, residual = self.input_layernorm.forward(x, residual)
        x = self.self_attn.forward(x)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        x = self.mlp.forward(x)
        return x, residual


class Qwen3Model(BaseOP):
    """embedding + 全部层 + 末尾 norm。"""

    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [Qwen3DecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
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


class Qwen3MoeForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = Qwen3Model(config)
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


__all__ = ["Qwen3MoeForCausalLM"]
