from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from minisgl.core import SamplingParams

from .utils import deserialize_type, serialize_type


@dataclass
class BaseTokenizerMsg:
    """发给 tokenizer 进程族的消息，按**收方**命名。

    "tokenizer 进程族"包括专门做 detokenize 的那一个（launch.py 里的
    minisgl-detokenizer-0），所以 scheduler 回采样结果用的 DetokenizeMsg 也属于这一族：
    它的 encoder 挂在 scheduler 的 PUSH 队列上（见 scheduler/io.py）。
    """

    @staticmethod
    def encoder(msg: BaseTokenizerMsg) -> Dict:
        # 写法与 BaseBackendMsg.encoder 略有不同（那个是实例方法），但两者都是
        # 以 `XxxMsg.encoder(obj)` 的形式被 zmq 队列调用的，效果一样
        return serialize_type(msg)

    @staticmethod
    def decoder(json: Dict) -> BaseTokenizerMsg:
        return deserialize_type(globals(), json)


@dataclass
class BatchTokenizerMsg(BaseTokenizerMsg):
    """多条打一批；对面收到后用 `_unwrap_msg` 拆回列表（见 tokenizer/server.py）。"""

    data: List[BaseTokenizerMsg]


@dataclass
class DetokenizeMsg(BaseTokenizerMsg):
    """一个请求刚采样出的 token，由 scheduler 发出（`Scheduler.send_result`）。

    finished 为 True 表示这条序列到此为止，tokenizer 侧解码完就该收尾了。
    """

    uid: int
    next_token: int
    finished: bool


@dataclass
class TokenizeMsg(BaseTokenizerMsg):
    """待分词的输入：纯文本，或 chat 格式的消息列表（[{role, content}, ...]）。"""

    uid: int
    text: str | List[Dict[str, str]]
    sampling_params: SamplingParams


@dataclass
class AbortMsg(BaseTokenizerMsg):
    """前端发来的取消请求；tokenizer 侧会转成 AbortBackendMsg 再转发给 scheduler。"""

    uid: int
