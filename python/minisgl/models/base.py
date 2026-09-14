"""所有 LLM 模型的基类：只规定一件事 —— forward 不接受参数。

因为输入（input_ids / positions / out_loc）都挂在全局上下文里（见 core.Context），
模型这一层不用透传 batch。这个约定让 CUDA graph 抓取变得很简单：
GraphRunner 只管把 dummy batch 挂上去，然后调用同样签名的 `model.forward()`。

Engine 里就是这么用的：
    with get_global_ctx().forward_batch(batch):
        logits = model.forward()     # 模型自己去 ctx 里取输入
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from minisgl.layers import BaseOP

if TYPE_CHECKING:
    import torch


class BaseLLMModel(ABC, BaseOP):
    """同时是 BaseOP（能 state_dict/load_state_dict）和 ABC（必须实现 forward）。"""

    @abstractmethod
    def forward(self) -> torch.Tensor: ...
