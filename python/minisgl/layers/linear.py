"""TP（张量并行）下的各种 Linear。

每个类只决定两件事：**权重按哪种方式切**、**要不要 all_reduce**。

    列并行（切输出）：输入完整，每 rank 算一段输出          → 不需要通信
    行并行（切输入）：输入分段，各算部分和，再 all_reduce 求和 → 需要通信
    复制：            每 rank 都有全量权重，各算各的完整结果   → 不需要通信

权重张量的形状一律是「本 rank 的」形状（local_osize × local_isize），
全局形状只记录在 full_* 字段里备用。加载权重时必须按同样的规则切
（见 models/weight.py），两边不一致会直接踩到形状断言。
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from minisgl.distributed import DistributedCommunicator, get_tp_info
from minisgl.utils import div_even

from .base import BaseOP


class _LinearTPImpl(BaseOP):
    """Real implementation of a linear layer with tensor parallelism."""

    def __init__(
        self,
        full_isize: int,
        full_osize: int,
        local_isize: int,
        local_osize: int,
        has_bias: bool,
    ):
        # full_* 只是记账（调试、形状推导时用），真正参与计算的是 local_*
        self.full_input_size = full_isize
        self.full_output_size = full_osize
        self.local_input_size = local_isize
        self.local_output_size = local_osize
        # torch.empty 在 meta 设备下 = 只占形状不占显存，等 load_state_dict 换真权重
        self.weight = torch.empty(local_osize, local_isize)
        # bias 为 None 时属性名照样存在（值是 None）——state_dict 只收张量，会自动跳过
        self.bias = torch.empty(local_osize) if has_bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class LinearReplicated(_LinearTPImpl):
    """
    Linear layer where weights are replicated (not sharded) across all TP ranks.
    Each GPU holds the full weight matrix.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_bias: bool,
    ):
        # 注意 local_* 直接等于 full_*：没有切分，forward 后也不需要任何通信
        super().__init__(
            full_isize=input_size,
            full_osize=output_size,
            local_isize=input_size,
            local_osize=output_size,
            has_bias=has_bias,
        )


class LinearColParallelMerged(_LinearTPImpl):
    """列并行 + 多路合并：一个权重矩阵同时算出好几路输出（如 gate/up，或 q/k/v）。

    切法是"**每一路各自切**"而不是"先拼好再整体切"：
        local_osize = Σ div_even(size_i, tp)
    这样每个 rank 拿到的每一路都是完整输出里的连续一段 —— 加载权重时必须按同样的
    顺序切，否则本 rank 的 q 会配到别的 rank 的 k。
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: List[int],
        has_bias: bool,
    ):
        # check that all output sizes are divisible by tp_size
        tp_info = get_tp_info()
        tp_output_sizes = [div_even(size, tp_info.size) for size in output_sizes]
        output_size = sum(output_sizes)
        tp_output_size = sum(tp_output_sizes)
        super().__init__(input_size, output_size, input_size, tp_output_size, has_bias)


class LinearQKVMerged(_LinearTPImpl):
    """q/k/v 三合一投影（MHA/GQA 的标准做法：一次矩阵乘出三份，省两次读权重）。

    全局输出宽度 = (num_qo + 2 × num_kv) × head_dim，本 rank 的按头数切：
      - q 头必须能被 tp 整除（每 rank 拿 local_num_qo 个头）；
      - kv 头允许"复制"：tp 比 kv 头数还多时 allow_replicate=True 会让结果至少为 1，
        以免出现某个 rank 分到 0 个 kv 头（GQA + 大 TP 的边角情况）。
    """

    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_qo_heads: int,
        num_kv_heads: int,
        has_bias: bool,
    ):
        tp_info = get_tp_info()

        local_num_qo = div_even(num_qo_heads, tp_info.size)
        local_num_kv = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        full_isize = hidden_size
        full_osize = (num_qo_heads + 2 * num_kv_heads) * head_dim
        local_isize = hidden_size
        local_osize = (local_num_qo + 2 * local_num_kv) * head_dim
        super().__init__(full_isize, full_osize, local_isize, local_osize, has_bias)


class LinearOProj(_LinearTPImpl):
    """attention 的输出投影：行并行（输入被上一层的头切分切过，输出要合回完整宽度）。

    与下面的 LinearRowParallel 实现完全相同，分开命名只是为了表明用途。
    通信对象现取现用：DistributedCommunicator 是插件式的（默认 torch.distributed，
    启用 pynccl 后换成 pynccl 实现），所以不能提前把实现缓存下来。
    """

    def __init__(self, input_size: int, output_size: int, has_bias: bool):
        tp_info = get_tp_info()
        full_isize = input_size
        full_osize = output_size
        local_isize = div_even(input_size, tp_info.size)
        local_osize = output_size
        # 下划线开头 → 不会被 state_dict 当成权重收进去
        self._comm = DistributedCommunicator()
        self._tp_size = tp_info.size
        super().__init__(full_isize, full_osize, local_isize, local_osize, has_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)
        # 各 rank 只算了自己那段输入对应的"部分和"，必须加起来才是完整结果
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)
        return y


class LinearRowParallel(_LinearTPImpl):
    """通用行并行 Linear（MLP 的 down_proj 用的就是它）。

    输入按 tp 切分（上游的 gate/up 是列并行，输出本来就是分段的），
    输出保持完整宽度，所以末尾要 all_reduce。
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_bias: bool,
    ):
        tp_info = get_tp_info()
        local_input_size = div_even(input_size, tp_info.size)
        local_output_size = output_size
        self._comm = DistributedCommunicator()
        self._tp_size = tp_info.size
        super().__init__(input_size, output_size, local_input_size, local_output_size, has_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)
        return y
