from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from minisgl.core import SamplingParams

from .utils import deserialize_type, serialize_type


@dataclass
class BaseBackendMsg:
    """发给 backend（也就是 scheduler 进程）的消息族，按**收方**命名。

    这族的 encoder 挂在 tokenizer → scheduler 的 PUSH 队列上，decoder 挂在 scheduler
    的 PULL 队列上（见 scheduler/io.py）。encoder / decoder 只是给 zmq 队列当编解码
    钩子的静态方法，本身不含通信逻辑（实现在 utils.py）。
    """

    def encoder(self) -> Dict:
        return serialize_type(self)

    @staticmethod
    def decoder(json: Dict) -> BaseBackendMsg:
        # globals() 是本模块的名字空间：解码按 __type__ 里的类名在这里查类，
        # 所以消息（含嵌套类型）用到的类必须在本模块 import 过
        return deserialize_type(globals(), json)


@dataclass
class BatchBackendMsg(BaseBackendMsg):
    """把多条消息打成一批发送，省掉几次 zmq 往返。

    scheduler 侧收到后会拆开逐条处理（`_process_one_msg` 递归调用自己）；
    tokenizer 进程就是把一批 UserMsg / AbortBackendMsg 这样发过来的。
    """

    data: List[BaseBackendMsg]


@dataclass
class ExitMsg(BaseBackendMsg):
    """让 scheduler 退出：收到后 `_process_one_msg` 直接抛 KeyboardInterrupt。

    NOTE: 本仓库里搜不到发送方——在线模式的退出实际是 Ctrl-C 之后 kill 整棵进程树
    （见 api_server.py 的 shell 与 launch.py），所以这个分支是留给外部做"消息式退出"的。
    """

    pass


@dataclass
class UserMsg(BaseBackendMsg):
    """新请求：tokenizer 进程分词完成后构造（见 tokenizer/server.py）。

    NOTE: 别用 `==` 比较两条 UserMsg —— dataclass 生成的 __eq__ 会拿 input_ids 这个
    张量去做布尔判断，多元素张量直接抛 RuntimeError。代码里只用 isinstance 判断类型。
    """

    uid: int
    input_ids: torch.Tensor  # CPU 1D int32 tensor
    sampling_params: SamplingParams


@dataclass
class AbortBackendMsg(BaseBackendMsg):
    """取消一个请求；scheduler 按 uid 先去待 prefill 队列找，再去 decode 集合找。"""

    uid: int
