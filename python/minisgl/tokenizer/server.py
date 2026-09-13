from __future__ import annotations

import multiprocessing as mp
from typing import List

import torch
from minisgl.message import (
    AbortBackendMsg,
    AbortMsg,
    BaseBackendMsg,
    BaseFrontendMsg,
    BaseTokenizerMsg,
    BatchBackendMsg,
    BatchFrontendMsg,
    BatchTokenizerMsg,
    DetokenizeMsg,
    TokenizeMsg,
    UserMsg,
    UserReply,
)
from minisgl.utils import ZmqPullQueue, ZmqPushQueue, init_logger, load_tokenizer


def _unwrap_msg(msg: BaseTokenizerMsg) -> List[BaseTokenizerMsg]:
    """拆批：BatchTokenizerMsg 拆成里面的列表，单条消息就是单元素列表。"""
    if isinstance(msg, BatchTokenizerMsg):
        return msg.data
    return [msg]


@torch.inference_mode()
def tokenize_worker(
    *,
    tokenizer_path: str,
    addr: str,
    create: bool,
    backend_addr: str,
    frontend_addr: str,
    local_bs: int,
    tokenizer_id: int = -1,
    model_source: str = "huggingface",
    ack_queue: mp.Queue[str] | None = None,
) -> None:
    """tokenizer 进程的主体：收消息 → 分词/反分词 → 回结果，一直循环到进程被杀。

    同一个函数干两件事，靠收到的消息类型区分（launch.py 会起多个进程分担）：

      - 收 TokenizeMsg / AbortMsg（来自前端）：分词后把 UserMsg / AbortBackendMsg
        打包发给 scheduler；
      - 收 DetokenizeMsg（来自 scheduler）：增量解码后把 UserReply 发给前端。

    专门跑第二种的那个进程，它的 addr 就是 scheduler 推送结果用的 zmq_detokenizer_addr，
    所以长文本生成的逐 token 解码不会拖慢新 prompt 的分词。

    inference_mode 覆盖整个函数：这里产出的 input_ids 张量不必带 autograd 记账。

    NOTE: `model_source` 参数没人传也没被用到——模型来源是在 server/args.py 解析参数时
    就处理掉的（那时模型已经下载完了），这里留着大概是历史遗留。
    """
    # 两个 PUSH 出口：一个回 scheduler，一个回前端；create=False 表示都由对方 bind
    send_backend = ZmqPushQueue(backend_addr, create=False, encoder=BaseBackendMsg.encoder)
    send_frontend = ZmqPushQueue(frontend_addr, create=False, encoder=BaseFrontendMsg.encoder)
    # 唯一的收信口：create 由调用方给，保证与 backend / 前端之间只有一方 bind（见 args.py）
    recv_listener = ZmqPullQueue(addr, create=create, decoder=BatchTokenizerMsg.decoder)
    assert local_bs > 0
    tokenizer = load_tokenizer(tokenizer_path)
    logger = init_logger(__name__, f"tokenizer_{tokenizer_id}")

    from .detokenize import DetokenizeManager
    from .tokenize import TokenizeManager

    tokenize_manager = TokenizeManager(tokenizer)
    detokenize_manager = DetokenizeManager(tokenizer)

    if ack_queue is not None:
        # 告诉启动方"我准备好了"，launch.py 会等齐所有子进程再放开服务
        ack_queue.put(f"Tokenize server {tokenizer_id} is ready")

    try:
        while True:
            # 先阻塞收一条，然后再把已经到达的尽量多收几条，攒成一批处理
            pending_msg = _unwrap_msg(recv_listener.get())
            while len(pending_msg) < local_bs and not recv_listener.empty():
                pending_msg.extend(_unwrap_msg(recv_listener.get()))
            # local_bs 是"这个进程一次至少处理几条"；launch.py 传的是 1，所以在线模式下
            # 实际不会攒批。empty() 是 poll(timeout=0)，不会在这里卡住

            logger.debug(f"Received {len(pending_msg)} messages")

            # 一批里三种消息可能混在一起，先分桶，再各处理各的
            detokenize_msg = [m for m in pending_msg if isinstance(m, DetokenizeMsg)]
            tokenize_msg = [m for m in pending_msg if isinstance(m, TokenizeMsg)]
            abort_msg = [m for m in pending_msg if isinstance(m, AbortMsg)]
            assert len(detokenize_msg) + len(tokenize_msg) + len(abort_msg) == len(pending_msg)
            if len(detokenize_msg) > 0:
                # scheduler 送来的 token → 增量文本 → 回前端（见 detokenize.py）
                replies = detokenize_manager.detokenize(detokenize_msg)
                batch_output = BatchFrontendMsg(
                    data=[
                        UserReply(
                            uid=msg.uid,
                            incremental_output=reply,
                            finished=msg.finished,
                        )
                        for msg, reply in zip(detokenize_msg, replies, strict=True)
                    ]
                )
                if len(batch_output.data) == 1:
                    batch_output = batch_output.data[0]  # 只有一条就不用包 Batch 了
                send_frontend.put(batch_output)

            if len(tokenize_msg) > 0:
                # 前端送来的文本 → token id → 回 scheduler。注意协议族换了：
                # 收的是 TokenizeMsg，发出去的是 UserMsg（text 已经变成 input_ids）
                tensors = tokenize_manager.tokenize(tokenize_msg)
                batch_output = BatchBackendMsg(
                    data=[
                        UserMsg(
                            uid=msg.uid,
                            input_ids=t,
                            sampling_params=msg.sampling_params,
                        )
                        for msg, t in zip(tokenize_msg, tensors, strict=True)
                    ]
                )
                if len(batch_output.data) == 1:
                    batch_output = batch_output.data[0]
                send_backend.put(batch_output)
            if len(abort_msg) > 0:
                # 取消消息本进程不处理，只是换个协议族转发给 scheduler
                # （uid 原样带上，scheduler 那边按 uid 找请求）
                batch_output = BatchBackendMsg(
                    data=[AbortBackendMsg(uid=msg.uid) for msg in abort_msg]
                )
                if len(batch_output.data) == 1:
                    batch_output = batch_output.data[0]
                send_backend.put(batch_output)
    except KeyboardInterrupt:
        pass  # Ctrl-C 时静静退出：进程一结束 zmq socket 就跟着关掉
