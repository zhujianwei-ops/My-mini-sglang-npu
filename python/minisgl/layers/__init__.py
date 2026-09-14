"""layers 包的门面：把各个算子模块统一导出，模型定义只需要一句
`from minisgl.layers import ...`。

包内分工（按数据流顺序）：

    embedding.py   VocabParallelEmbedding / ParallelLMHead   词表并行
    linear.py      Linear*（列并行 / 行并行 / 复制）           TP 切分与通信
    norm.py        RMSNorm / RMSNormFused                     归一化（融合残差）
    activation.py  silu_and_mul / gelu_and_mul                门控 MLP 的激活
    rotary.py      get_rope / RotaryEmbedding                 RoPE 表与原地旋转
    attention.py   AttentionLayer                             切 qkv + QK-Norm + RoPE
    moe.py         MoELayer                                   MoE 专家权重容器
    base.py        BaseOP / OPList / StateLessOP              自定义的算子树框架

这里只做转出，不写逻辑；`__all__` 是公开 API 清单。
"""

from .activation import gelu_and_mul, silu_and_mul
from .attention import AttentionLayer
from .base import BaseOP, OPList, StateLessOP
from .embedding import ParallelLMHead, VocabParallelEmbedding
from .linear import (
    LinearColParallelMerged,
    LinearOProj,
    LinearQKVMerged,
    LinearReplicated,
    LinearRowParallel,
)
from .moe import MoELayer
from .norm import RMSNorm, RMSNormFused
from .rotary import get_rope, set_rope_device

__all__ = [
    "silu_and_mul",
    "gelu_and_mul",
    "AttentionLayer",
    "BaseOP",
    "StateLessOP",
    "OPList",
    "VocabParallelEmbedding",
    "ParallelLMHead",
    "LinearColParallelMerged",
    "LinearRowParallel",
    "LinearOProj",
    "LinearQKVMerged",
    "RMSNorm",
    "RMSNormFused",
    "get_rope",
    "set_rope_device",
    "LinearReplicated",
    "MoELayer",
]
