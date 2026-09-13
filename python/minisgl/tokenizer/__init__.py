"""tokenizer 进程族的实现：分词、反分词，以及把它们接到 zmq 上的 worker。

    tokenize.py    文本 → token id（输入侧，prompt 分词）
    detokenize.py  token id → 文本（输出侧，核心是滑窗增量解码）
    server.py      进程主体：收发消息，按类型分派给上面两个 manager

调用方只从这里取 tokenize_worker（见 server/launch.py）。
"""

from .server import tokenize_worker

__all__ = ["tokenize_worker"]
