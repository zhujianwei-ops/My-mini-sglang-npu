"""迷你模型框架：整个仓库不用 torch.nn，而是这套 BaseOP 组合成的对象树。

三个设计取舍：

  1. 参数直接以 torch.Tensor 存在 `__dict__` 里（没有 nn.Parameter / _parameters 簿记）。
     于是 Engine 里那句 load_state_dict 可以**直接 setattr 换张量**，把 meta 设备上
     建的骨架整块替换成真权重 —— nn.Module 的 copy_ 路线在 meta 张量上会报错。
  2. 前缀由"父对象的属性名"拼出来，得到 `model.layers.3.self_attn.qkv_proj.weight`
     这样的点分键，正好和权重文件里的名字对上（见 models/weight.py）。
  3. 没有 hooks / autograd 相关的开销，forward 路径极短 —— CUDA graph 抓取和每步
     replay 都在这条路径上，Python 侧每省一点都值钱。

state_dict / load_state_dict 是**成对实现**的：怎么拼前缀、怎么递归，两边必须一致。
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, Generic, List, TypeAlias, TypeVar

import torch

# 权重字典：键 = 点分路径（与权重文件里的名字一致），值 = 张量
_STATE_DICT: TypeAlias = Dict[str, torch.Tensor]


def _concat_prefix(prefix: str, name: str) -> str:
    """拼前缀：顶层对象没有前缀，此时不加那个「.」（否则键会以点开头）。"""
    return f"{prefix}.{name}" if prefix else name


class BaseOP:
    """所有算子/层的基类：一个 forward + 一对 state_dict/load_state_dict。"""

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any: ...

    def state_dict(self, *, prefix: str = "", result: _STATE_DICT | None = None) -> _STATE_DICT:
        """把自己（及子树）里的张量收进一个扁平的 {点分路径: 张量} 字典。

        result 是"就地累积"用的：递归时传同一个 dict 下去，避免每一层都新建再合并。
        """
        result = result if result is not None else {}

        for name, param in self.__dict__.items():
            # 下划线开头的是私有属性（如 _comm / _tp_size / _slice），不是权重，跳过
            if name.startswith("_"):
                continue
            if isinstance(param, torch.Tensor):
                result[_concat_prefix(prefix, name)] = param
            elif isinstance(param, BaseOP):
                # 子算子递归，前缀加上自己的属性名
                param.state_dict(prefix=_concat_prefix(prefix, name), result=result)

        return result

    def load_state_dict(
        self,
        state_dict: _STATE_DICT,
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        """按同样的遍历顺序把权重灌进来。

        两处刻意的严格：
          - `state_dict.pop(...)` 是**取走**，不是读取 —— 全部灌完后若还剩键，说明
            权重文件里有这个模型不认识的名字（拼错/版本不对），直接报错；
          - assert 形状和 dtype 一致，防止"能跑但算错"。
        `setattr` 直接把 meta 占位张量换成真张量（见模块注释第 1 条）。
        """
        for name, param in self.__dict__.items():
            if name.startswith("_"):
                continue
            if isinstance(param, torch.Tensor):
                item = state_dict.pop(_concat_prefix(prefix, name))
                assert isinstance(item, torch.Tensor)
                assert param.shape == item.shape and param.dtype == item.dtype
                setattr(self, name, item)
            elif isinstance(param, BaseOP):
                # 递归进去；_internal=True 表示"下面还有别的地方在管键"，先别急着报错
                param.load_state_dict(
                    state_dict, prefix=_concat_prefix(prefix, name), _internal=True
                )

        # 只有最外层这次检查才有意义：_internal 时键还没被兄弟模块消费完
        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")


class StateLessOP(BaseOP):
    """没有参数的算子（如激活函数）：两个字典方法都是空操作。

    但键检查仍然要做 —— 它的存在是为了让"权重文件里有它不认识的键"这种情况
    在任何一层都能被发现。
    """

    def __init__(self):
        super().__init__()

    def load_state_dict(
        self,
        state_dict: _STATE_DICT,
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")

    def state_dict(self, *, prefix: str = "", result: _STATE_DICT | None = None) -> _STATE_DICT:
        return result if result is not None else {}


T = TypeVar("T", bound=BaseOP)


class OPList(BaseOP, Generic[T]):
    """一串同类算子（如 Transformer 的几十层），成员靠**下标**区分。

    键里出现下标：`...layers.17.self_attn...`，与权重文件里的编号一一对应。
    列表自身不是 BaseOP，所以要专门实现一遍遍历 —— 注意它自己不持有张量，
    前缀拼接用的是 str(i) 而不是属性名。
    """

    def __init__(self, ops: List[T]):
        super().__init__()
        self.op_list = ops

    def state_dict(self, *, prefix: str = "", result: _STATE_DICT | None = None) -> _STATE_DICT:
        result = result if result is not None else {}
        for i, op in enumerate(self.op_list):
            op.state_dict(prefix=_concat_prefix(prefix, str(i)), result=result)
        return result

    def load_state_dict(
        self,
        state_dict: _STATE_DICT,
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        for i, op in enumerate(self.op_list):
            op.load_state_dict(state_dict, prefix=_concat_prefix(prefix, str(i)), _internal=True)

        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")
