from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .utils import deserialize_type, serialize_type


@dataclass
class BaseFrontendMsg:
    """发给前端（api_server）的消息族，按**收方**命名。

    tokenizer 进程处理完分词 / 反分词后用这一族把结果送回去；解码器挂在前端的
    PULL 队列上，前端拿到 Batch 后用 `_unwrap_msg` 拆开（见 server/api_server.py）。
    """

    @staticmethod
    def encoder(msg: BaseFrontendMsg) -> Dict:
        return serialize_type(msg)

    @staticmethod
    def decoder(json: Dict) -> BaseFrontendMsg:
        return deserialize_type(globals(), json)


@dataclass
class BatchFrontendMsg(BaseFrontendMsg):
    """多条打一批：一轮消息里可能有多个请求的增量输出，打包发更省事。"""

    data: List[BaseFrontendMsg]


@dataclass
class UserReply(BaseFrontendMsg):
    """某个请求的增量输出，前端按 uid 找到对应的流推给客户端。"""

    uid: int
    incremental_output: str  # 增量文本片段，不是累积全量
    finished: bool  # 该请求已结束，前端可以收尾这条流
