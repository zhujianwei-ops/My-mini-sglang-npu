"""进程间消息的统一定义。三个"消息族"按**收方**划分，而不是按发方：

    BaseBackendMsg    发给 backend（scheduler）：UserMsg / AbortBackendMsg / ExitMsg
    BaseTokenizerMsg  发给 tokenizer 进程族：TokenizeMsg / DetokenizeMsg / AbortMsg
    BaseFrontendMsg   发给前端（api_server）：UserReply

    注意 DetokenizeMsg 属于 tokenizer 那一族：专门做 detokenize 的进程也在这族里。

一次请求的完整消息旅程：

    api_server ──TokenizeMsg──► tokenizer 进程（分词）
                                   │ UserMsg
                                   ▼
                              scheduler（调度 + 采样）
                                   │ DetokenizeMsg（每个新 token 一条）
                                   ▼
                            detokenizer 进程（增量解码）
                                   │ UserReply
                                   ▼
                              api_server（推给 HTTP 流）

每一族都自带 encoder / decoder（静态方法），供 zmq 队列在收发时做编解码钩子；
具体实现在 utils.py 里。这里只是把三族集中导出，方便 `from minisgl.message import X`。
"""

from .backend import AbortBackendMsg, BaseBackendMsg, BatchBackendMsg, ExitMsg, UserMsg
from .frontend import BaseFrontendMsg, BatchFrontendMsg, UserReply
from .tokenizer import AbortMsg, BaseTokenizerMsg, BatchTokenizerMsg, DetokenizeMsg, TokenizeMsg

__all__ = [
    "AbortMsg",
    "AbortBackendMsg",
    "BaseBackendMsg",
    "BatchBackendMsg",
    "ExitMsg",
    "UserMsg",
    "BaseTokenizerMsg",
    "BatchTokenizerMsg",
    "DetokenizeMsg",
    "TokenizeMsg",
    "BaseFrontendMsg",
    "BatchFrontendMsg",
    "UserReply",
]
