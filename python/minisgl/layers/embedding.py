"""词表并行的 Embedding 与 LM Head。

两者的权重都是按词表维切成一堆"段"，每 rank 只存一段：
    rank r 负责词表区间 [start_idx, start_idx + num_embeddings_tp)
通信方式不同，这是关键区别：
    Embedding  查表 → 本 rank 命中的行自己算，其余写 0，最后 **all_reduce** 相加
    LM Head    投影 → 每个 rank 算出自己那段 logits，最后 **all_gather** 拼成整表
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from minisgl.core import get_global_ctx
from minisgl.distributed import DistributedCommunicator, get_tp_info
from minisgl.utils import div_ceil, nvtx_annotate

from .base import BaseOP


class VocabParallelEmbedding(BaseOP):
    """按词表维切分的 Embedding：每 rank 只持有自己那一段词表的行。"""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        tp_info = get_tp_info()
        tp_rank = tp_info.rank
        self.tp_size = tp_info.size
        self.num_embeddings = num_embeddings
        # 用 div_ceil：**允许词表数不被 tp 整除**，每段向上取整（最后一段会短一点，
        # 由下面的 vocab_range 兜住，不会越界）
        self.num_embeddings_tp = div_ceil(num_embeddings, self.tp_size)
        start_idx = self.num_embeddings_tp * tp_rank
        # finish_idx 用 min 夹住：最后那个 rank 的段可能超出真实词表
        finish_idx = min(start_idx + self.num_embeddings_tp, num_embeddings)
        # (起始 id, 本段长度) —— 要传给 kernel 做范围判断
        self.vocab_range = (start_idx, finish_idx - start_idx)
        self.weight = torch.empty(self.num_embeddings_tp, embedding_dim)
        self._comm = DistributedCommunicator()

    @nvtx_annotate("Embedding")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 延迟导入：kernel 模块会触发 JIT 编译，放在模块顶层会拖慢 import
        from minisgl.kernel import indexing

        # 查表 kernel（kernel/index.py → csrc/jit/index.cu 的 masked_index_kernel）：
        #   x 是**一维扁平**的 token id（就是 batch.input_ids 的形状），输出 (L, hidden)
        #   - id 落在本 rank 的段里 → 拷那一行；
        #   - 不在 → 写 0（所以 all_reduce 之后每行正好由"负责它的那个 rank"贡献）。
        # vocab_range=None 表示单卡，没有范围判断，直接查表
        y = indexing(
            weights=self.weight,
            indices=x,
            vocab_range=self.vocab_range if self.tp_size > 1 else None,
        )

        return self._comm.all_reduce(y) if self.tp_size > 1 else y


class ParallelLMHead(VocabParallelEmbedding):
    """词表并行的输出头：把 hidden 投影到（被切分的）词表上。

    跟 Embedding 共用"按词表切段"的布局，所以直接继承过来复用 self.weight 的切法；
    但 forward 完全不同（投影而非查表），且多了两个特殊处理：只算最后一个 token、
    以及可以跟 Embedding 共享权重。
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
        tie_word_embeddings: bool = False,
        tied_embedding: VocabParallelEmbedding | None = None,
    ):
        super().__init__(num_embeddings, embedding_dim)
        # bias 也跟着词表切：本 rank 只持有自己那段 logits 的偏置
        self.bias = torch.empty(self.num_embeddings_tp) if bias else None
        self.tied_embedding = tied_embedding
        # "给了共享权重" 和 "配置说共享" 必须同时成立，否则说明模型定义写错了
        assert (tied_embedding is not None) == tie_word_embeddings

    def load_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        """权重共享时，把 lm_head 自己的那份权重从字典里"取走"并丢弃。

        权重文件里可能同时存了 embedding 和 lm_head（哪怕配置是共享的），
        不 pop 掉的话最外层那句 "Unexpected keys" 检查会报错。
        """
        if not self.tied_embedding:
            return super().load_state_dict(state_dict, prefix=prefix, _internal=_internal)
        else:
            # pop the lm_head.weights and lm_head.bias if they exist
            possible_weight = f"{prefix}.weight"
            possible_bias = f"{prefix}.bias"
            if possible_weight in state_dict:
                state_dict.pop(possible_weight)
            if possible_bias in state_dict:
                state_dict.pop(possible_bias)

    def state_dict(
        self,
        *,
        prefix: str = "",
        result: Dict[str, torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor]:
        """共享权重时交出空字典：真正的权重已经在 embedding 那份里了，
        这里再报一次会让同一个张量出现在两个键上（也会让 tied_embedding 被递归两遍）。"""
        if not self.tied_embedding:
            return super().state_dict(prefix=prefix, result=result)
        return {} if result is None else result

    @nvtx_annotate("LMHead")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        batch = ctx.batch
        bs = batch.size
        if batch.is_prefill:
            # prefill 只需要每个请求**最后一个 token** 的 logits（要用它采样），
            # 中间那些位置的 logits 没人要。lm_head 是 (hidden × vocab) 的大矩阵乘，
            # 这一刀省掉的是 prefill 阶段最贵的一步之一
            indices = batch.attn_metadata.get_last_indices(bs)
            x = x[indices].contiguous()
            del indices

        # 权重共享时用 embedding 的权重（本 rank 的切片是一样的）
        module = self.tied_embedding or self
        logits = F.linear(x, module.weight, self.bias)
        if self.tp_size == 1:
            return logits
        # 每个 rank 只算了自己那段词表的 logits，拼起来才是完整的
        input_shape = logits.shape
        output_tensor = self._comm.all_gather(logits)

        if bs == 1:
            # 单请求特例：gather 出来是 (tp, V_local)，按行摊平正好就是词表顺序
            # （rank0 的整段 + rank1 的整段 + ...），省掉下面的 permute/contiguous
            return output_tensor.view(1, -1)[:, : self.num_embeddings]

        # all_gather 是沿第 0 维拼的，得到的是 (tp * bs, V_local)：前 bs 行都属于 rank0。
        # 先 view 成 (tp, bs, V_local) 再 permute 成 (bs, tp, V_local)，
        # 这样每个请求的 V = [rank0 段, rank1 段, ...] 才是正确的词表顺序
        output_tensor = output_tensor.view((self.tp_size,) + input_shape)
        output_tensor = output_tensor.permute(1, 0, 2).contiguous()
        output_tensor = output_tensor.reshape(input_shape[:1] + (self.tp_size * input_shape[1],))
        # 每段是向上取整的，拼起来会比真实词表长，切掉多余的（kernel 不读这些位置）
        return output_tensor[:, : self.num_embeddings]
