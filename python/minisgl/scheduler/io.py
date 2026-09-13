from __future__ import annotations

from typing import TYPE_CHECKING, Final, List

import torch
from minisgl.message import BaseBackendMsg, BaseTokenizerMsg, BatchTokenizerMsg, DetokenizeMsg
from minisgl.utils import ZmqPubQueue, ZmqPullQueue, ZmqPushQueue, ZmqSubQueue, init_logger

if TYPE_CHECKING:
    from .config import SchedulerConfig

logger = init_logger(__name__)


class SchedulerIOMixin:
    """
    Mixin class for Scheduler I/O operations.

    This class handles the communication between the scheduler and the tokenizer.

    Public Utilities:
        receive_msg: Function to receive messages from the tokenizer.
        send_result: Function to send results back to the tokenizer.
        sync_all_ranks: Function to synchronize all ranks on CPU side.

    在线模式下每个 scheduler 进程和 tokenizer 进程之间的两条 zmq 链路（地址都带 pid 后缀，
    见 SchedulerConfig，避免多个实例互相串线）：

        tokenizer ──PUSH──► zmq_backend_addr      ◄──PULL── scheduler rank0   （收请求）
        tokenizer ◄──PULL── zmq_detokenizer_addr  ◄──PUSH── scheduler rank0   （回结果）

    只有 rank0 和 tokenizer 直接对话；其他 rank 的消息由 rank0 广播补齐，它们自己不碰 zmq：

        scheduler rank0 ──PUB(zmq_scheduler_broadcast_addr)──► scheduler rank1..N

    所以 receive_msg / send_result 这两个"方法"在 __init__ 里按模式（离线 / 单 rank /
    多 rank / 是否 primary）被绑定成不同实现——调度主循环只管调 self.receive_msg /
    self.send_result，不需要知道自己跑在什么拓扑上。子类必须实现 run_when_idle，离线模式
    还要实现 offline_receive_msg / offline_send_result（见 Scheduler、llm.LLM）。
    """

    def __init__(self, config: SchedulerConfig, tp_cpu_group: torch.distributed.ProcessGroup):
        tp_info = config.tp_info
        # gloo 的 CPU 通信组：只用来广播"待收几条消息"和 barrier，不搬运张量
        # （Final 只是给类型检查看的，运行时无效果）
        self.tp_cpu_group: Final = tp_cpu_group

        # 离线模式（LLM 类）根本没有 tokenizer 进程：消息由本进程自己造，结果也写回本进程。
        # 把两个"方法槽"换成离线实现后直接返回，全程不碰 zmq
        if config.offline_mode:
            self.receive_msg = self.offline_receive_msg
            self.send_result = self.offline_send_result
            return  # early exit

        # ---- 以下是在线模式。第一步：建好本 rank 与 tokenizer 之间的链路（仅 rank0 有）----
        if tp_info.is_primary():
            # 收：UserMsg / AbortBackendMsg 都从这里进来。create=True 表示由 scheduler 负责
            # bind，地址是 zmq_backend_addr；tokenizer 那边只是 connect（见 tokenizer/server.py）
            self._recv_from_tokenizer: Final = ZmqPullQueue(
                config.zmq_backend_addr,
                create=True,
                decoder=BaseBackendMsg.decoder,
            )
            # 发：DetokenizeMsg 从这里出去。谁来 bind 由 backend_create_detokenizer_link 决定：
            # 独占一个 detokenizer 进程时由 backend 建（True），与 tokenizer 共用地址时让对方建
            self._send_into_tokenizer: Final = ZmqPushQueue(
                config.zmq_detokenizer_addr,
                create=config.backend_create_detokenizer_link,
                encoder=BaseTokenizerMsg.encoder,
            )

        # 第二步：选出本 rank 该用哪套收发实现。先按单 rank 填默认值，多 rank 时再按身份覆盖，
        # 这样判断只写一处，也不会漏掉某个分支
        recv = self._recv_msg_single_rank
        send = self._reply_tokenizer_rank0
        if tp_info.size > 1:
            if tp_info.is_primary():
                # rank0 兼做广播源：把收到的原始字节原样 PUB 给其他 rank，省掉重复编码
                recv = self._recv_msg_multi_rank0
                self._send_into_ranks: Final = ZmqPubQueue(
                    config.zmq_scheduler_broadcast_addr, create=True, encoder=BaseBackendMsg.encoder
                )
            else:
                # 其他 rank：消息全部来自 rank0；结果也只由 rank0 回给 tokenizer，自己什么都不发
                recv = self._recv_msg_multi_rank1
                send = self._reply_tokenizer_rank1
                self._recv_from_rank0: Final = ZmqSubQueue(
                    config.zmq_scheduler_broadcast_addr,
                    create=False,
                    decoder=BaseBackendMsg.decoder,
                )

        # 绑定本次扮演的角色：之后 receive_msg / send_result 就是上面选中的实现
        self.receive_msg = recv
        self.send_result = send

    def run_when_idle(self):
        """阻塞等消息之前调一次，让子类做点后台活（Scheduler 用它打日志并校验缓存）。"""
        raise NotImplementedError("should be implemented")

    def offline_receive_msg(self, blocking: bool = False) -> List[BaseBackendMsg]:
        """离线模式的收消息：由 LLM 实现，从自己攒的待办列表里按预算取请求。"""
        raise NotImplementedError("should be implemented")

    def offline_send_result(self, reply: List[DetokenizeMsg]) -> None:
        """离线模式的回结果：由 LLM 实现，不再走网络，直接把 token 记进 status_map。"""
        raise NotImplementedError("should be implemented")

    def sync_all_ranks(self) -> None:
        """所有 rank 在 CPU 侧对上进度；Scheduler.shutdown 里调它保证一起退出。"""
        # barrier() 返回 Work 对象，.wait() 会一直阻塞到所有 rank 都到达
        self.tp_cpu_group.barrier().wait()

    def _recv_msg_single_rank(self, blocking: bool = False) -> List[BaseBackendMsg]:
        """单 rank 在线模式的收消息：顺手把 socket 里已经到齐的消息一并取走。

        blocking 由调用方（overlap_loop）按"确实没活干"算出来；为 True 时保证至少返回
        一条消息，主循环不会空转着回到调度逻辑。
        """
        pending_msgs: List[BaseBackendMsg] = []
        if blocking:
            # 阻塞路径：先让子类做后台活，再睡等第一条消息进来
            self.run_when_idle()
            pending_msgs.append(self._recv_from_tokenizer.get())
        # 再收已经到达的：empty() 内部是 poll(timeout=0)，只看一眼，不会阻塞
        while not self._recv_from_tokenizer.empty():
            pending_msgs.append(self._recv_from_tokenizer.get())
        return pending_msgs

    def _recv_msg_multi_rank0(self, blocking: bool = False) -> List[BaseBackendMsg]:
        """多 rank 时 rank0 的收消息：自己从 tokenizer 收，再原样转发给其他 rank。

        转发的是原始字节（get_raw / put_raw），不是重新编码的对象：其他 rank 拿到的
        就是 tokenizer 发来的 byte 流，各自本地解码即可，省一次 msgpack 编码。
        """
        pending_msgs: List[BaseBackendMsg] = []
        if blocking:
            # rank0 阻塞在 tokenizer 上。这条消息也要广播出去，但不走下面的计数逻辑
            self.run_when_idle()
            raw = self._recv_from_tokenizer.get_raw()
            self._send_into_ranks.put_raw(raw)
            pending_msgs.append(self._recv_from_tokenizer.decode(raw))

        pending_raw_msgs: List[bytes] = []
        while not self._recv_from_tokenizer.empty():
            pending_raw_msgs.append(self._recv_from_tokenizer.get_raw())

        # broadcast the number of raw messages to all ranks
        # 必须广播条数：SUB 端没有 empty() 之类的办法判断"这一轮收完了"，只能靠这个数字。
        # 每轮都会广播一次（哪怕条数是 0），它是各 rank 之间的同步点
        # （上面 blocking 那条两边都各自记了一笔，所以不参与计数）
        src_tensor = torch.tensor(len(pending_raw_msgs))
        self.tp_cpu_group.broadcast(src_tensor, root=0).wait()

        for raw in pending_raw_msgs:
            self._send_into_ranks.put_raw(raw)
            pending_msgs.append(self._recv_from_tokenizer.decode(raw))
        return pending_msgs

    def _recv_msg_multi_rank1(self, blocking: bool = False) -> List[BaseBackendMsg]:
        """非 primary rank 的收消息：消息全部来自 rank0 的 PUB，本 rank 不碰 tokenizer。"""
        pending_msgs: List[BaseBackendMsg] = []
        if blocking:
            # 与 rank0 的阻塞接收一一对应：两个分支各 rank 必须一致地走，
            # 否则一边卡在 get、一边卡在 broadcast，或者两边记的条数对不上
            self.run_when_idle()
            pending_msgs.append(self._recv_from_rank0.get())

        # ensure all ranks have the same number of raw messages
        # 与 rank0 的计数广播配对。初值 -1 只是占位，正常一定会被广播覆盖成真实条数
        dst_tensor = torch.tensor(-1)
        self.tp_cpu_group.broadcast(dst_tensor, root=0).wait()
        dst_length = int(dst_tensor.item())

        for _ in range(dst_length):
            pending_msgs.append(self._recv_from_rank0.get())
        return pending_msgs

    def _reply_tokenizer_rank0(self, reply: List[DetokenizeMsg]) -> None:
        """把采样结果回给 tokenizer；只有 primary rank 手里有这条链路。"""
        num_reply = len(reply)
        logger.debug_rank0(f"Replying to tokenizer: {num_reply} messages")
        if num_reply == 1:
            # 只有一条就直接发，省掉一层 Batch 包装
            self._send_into_tokenizer.put(reply[0])
        elif num_reply > 1:
            # 多条打包成一条消息，少几次 zmq 往返（对面的 _unwrap_msg 会拆回成列表）
            self._send_into_tokenizer.put(BatchTokenizerMsg(data=reply))  # type: ignore
        # num_reply == 0 时什么都不发：空批次没有内容，也不该惊动对面

    def _reply_tokenizer_rank1(self, reply: List[DetokenizeMsg]) -> None:
        """非 primary rank 不回消息：结果统一由 rank0 回给 tokenizer（见上一个函数）。"""
        _ = reply  # do nothing for non-primary ranks
