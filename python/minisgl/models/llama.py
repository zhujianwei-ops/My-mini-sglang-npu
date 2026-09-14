"""Llama 模型 —— 本目录其余几个模型文件都是它的变体（差异见各自文件开头）。

结构：

    embed_tokens → 若干 DecoderLayer → norm → lm_head

每个 DecoderLayer 是"norm → attention → norm → MLP"，两个 norm 都是**融合残差**的
RMSNormFused。这里有个贯穿全模型的约定，看 forward 的返回值就明白了：

    x        每个子层的**输出增量**（还没并进残差流）
    residual 已经累加好的**残差流**
    norm(x, residual) 做的是：residual += x; x = rmsnorm(residual)

所以子层的输出先是"增量"，进了下一个 norm 才被并进残差流。整串跑到最后，
`self.norm.forward(x, residual)[0]` 把最后一层的增量也并进去再归一化。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
from minisgl.core import get_global_ctx
from minisgl.layers import BaseOP, OPList, ParallelLMHead, RMSNormFused, VocabParallelEmbedding
from minisgl.utils import nvtx_annotate

from .base import BaseLLMModel
from .utils import GatedMLP as LlamaMLP
from .utils import RopeAttn as LlamaAttn

if TYPE_CHECKING:
    from .config import ModelConfig


class LlamaDecoderLayer(BaseOP):
    """一层 Transformer。只是把四块积木拼起来，自己没有别的逻辑。"""

    def __init__(self, config: ModelConfig, layer_id: int):
        # 注意 as 出来的是别名：下面用的 LlamaAttn / LlamaMLP 其实就是 utils 里的
        # RopeAttn / GatedMLP，换名字只是为了让这一层的代码读起来像 Llama 自己的实现
        self.self_attn = LlamaAttn(config, layer_id)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

        # 下划线开头：不会被 state_dict 当成权重；profiler 用它给每层单独打标记
        self._layer_id = layer_id

    # 抓 profile 时每一层是一个 nvtx range，名字里的 {} 由 _layer_id 填
    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 进 norm：residual += x（首层 residual 为 None 时直接用 x），x = norm(residual)
        x, residual = self.input_layernorm.forward(x, residual)
        x = self.self_attn.forward(x)  # 出 attention 的"增量"
        x, residual = self.post_attention_layernorm.forward(x, residual)  # 并进残差流
        x = self.mlp.forward(x)  # 出 MLP 的"增量"，交给下一层的第一个 norm 去并
        return x, residual


class LlamaModel(BaseOP):
    """embedding + 全部层 + 最后的 norm（lm_head 在外面那层）。"""

    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        # OPList：权重键里带层下标（model.layers.17.…），与权重文件的编号对应
        self.layers = OPList(
            [LlamaDecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids 是一维扁平的（本批所有待算 token 拼在一起），不是 [bs, seqlen]
        x = self.embed_tokens.forward(input_ids)
        residual: torch.Tensor | None = None  # 第一层的 norm 会用它为空这件事来跳过加法
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        # [0]：只要归一化后的结果，最后的 residual 已经没人要了
        # （这一步内部做了 residual += x，所以最后一层的增量不会漏掉）
        return self.norm.forward(x, residual)[0]


class LlamaForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = LlamaModel(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            # 配置说共享权重时，把 embedding 对象传给 lm_head，让它直接借用那份权重
            # （注意此时不能是 None，ParallelLMHead 里有 assert 校验这两者一致）
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        super().__init__()

    def forward(self) -> torch.Tensor:
        # 注意签名里**没有参数**（BaseLLMModel 的约定）：输入从全局 ctx 里取，
        # 这样 CUDA graph 抓图时才能用同样的一句 model.forward() 把 dummy batch 跑起来
        output = self.model.forward(get_global_ctx().batch.input_ids)
        logits = self.lm_head.forward(output)
        return logits


__all__ = ["LlamaForCausalLM"]
