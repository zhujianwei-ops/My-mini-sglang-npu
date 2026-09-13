"""decode 阶段：管住"已经 prefill 完、正在一个 token 一个 token 往外吐"的那些请求。

    DecodeManager   持有它们在跑的集合（running_reqs），每轮把它们打成一个 decode 批，
                    并在 forward 之后更新集合成员（新 prefill 完的进来、到长度上限的出去）。

和 prefill 那边（见 prefill.py）的分工：prefill 决定"谁能进来"，decode 只管"已经进来的怎么排批"。
decode 批不需要切块，也不需要 token 预算——complete_one() 每轮只把 device_len 推进 1 个
token，所以每个请求的 extend_len 恒为 1，一批的规模就是"在飞请求数"。

它唯一要替别人操心的地方是空间：inflight_tokens 把"这些请求将来还会占掉的 KV"算出来，
交给 PrefillAdder 当预留量，免得 prefill 把空间吃光、让在跑的 decode 无页可用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Set

from minisgl.core import Batch, Req


@dataclass
class DecodeManager:
    page_size: int  # KV 页的 token 数（engine/config.py，默认 1），只用来估预留空间
    # 在跑的 decode 请求。Req 是 @dataclass(eq=False)，按对象身份哈希，所以能直接放进 set；
    # 也正因为存的就是调度器手里那些 Req 对象本身，complete_one() 推进的长度这里直接可见。
    running_reqs: Set[Req] = field(default_factory=set)

    def filter_reqs(self, reqs: Iterable[Req]) -> None:
        """每轮 forward 之后调用（scheduler._forward 的最后一步），更新 decode 集合。

        传进来的是刚提交的那批 forward 的 reqs，于是这里同时干了两件事：

          - 收：本轮刚 prefill 完的请求从此进入 decode（这是请求进入 running_reqs 的
            唯一入口）；
          - 剔：用 can_decode 过滤掉已经到长度上限的（remain_len == 0），它们不再参与
            decode。ChunkedReq 的 can_decode 恒为 False（见 prefill.py），所以分块中的
            请求永远进不来，会一直留在 prefill 队列里。

        写成"用新集合替换旧集合"而不是原地增删，是因为旧成员要一个不少地保留（union
        里那些对象不会被复制，身份不变）。位置也有讲究：必须放在 forward 之后，因为
        can_decode 读的是 complete_one() 刚推进过的 remain_len。
        """
        self.running_reqs = {req for req in self.running_reqs.union(reqs) if req.can_decode}

    def remove_req(self, req: Req) -> None:
        """请求结束时把它从 decode 集合里摘掉（资源由调用方的 _free_req_resources 释放）。

        用 discard 而不是 remove：元素不在集合里也不报错。overlap 调度下同一个请求可能被
        释放两次（上一轮已释放、这批还在飞，见 scheduler.py 里 finished_reqs 的判重 NOTE），
        discard 的幂等性正好合适。
        """
        self.running_reqs.discard(req)

    def abort_req(self, uid: int) -> Req | None:
        """按 uid 找到被取消的请求、从集合里摘掉，并返回它让调用方释放资源。

        不在集合里就返回 None —— 那种情况说明请求还堵在 prefill 队列里排队，交给
        PrefillManager.abort_req 处理。
        """
        for req in self.running_reqs:
            if req.uid == uid:
                # 边迭代边改集合一般会炸，这里是安全的：remove 之后立刻 return，
                # 迭代不会再有下一次
                self.running_reqs.remove(req)
                return req
        return None

    @property
    def inflight_tokens(self) -> int:
        """在飞 decode 请求"将来还会占掉"的 KV 空间（按 token 数计），作为 PrefillAdder
        的 reserved_size 初值（见 prefill.py）：新请求得先给它们把地方留出来。

        除 remain_len 外，每个请求还要多算 (page_size - 1) 个 token（上游注释写的
        "1 page reserved"）：KV 是按整页分配的，请求跨页时整页拿走，最后一页未必能用满，
        于是它新占的空间最多比 remain_len 多出 (page_size - 1) 个 token —— 正好是一个
        没填满的页。page_size == 1 时这一项为 0；用 trtllm 后端时 page_size 是 16/32/64，
        这一项才真正起作用。

        这是按"每个请求都走到最坏情况"估出来的上界，偏保守：宁可让 prefill 少放几个请求，
        也不要让在跑的 decode 中途没页可用。
        """
        tokens_reserved = (self.page_size - 1) * len(self.running_reqs)  # 1 page reserved
        return sum(req.remain_len for req in self.running_reqs) + tokens_reserved

    def schedule_next_batch(self) -> Batch | None:
        """把所有在跑的 decode 请求打成一个批；没有请求就返回 None。

        按 uid 排序不是为了好看：running_reqs 是 set，迭代顺序取决于对象地址（Req 按身份
        哈希），而 TP 各 rank 是不同进程、地址不同，迭代顺序也就不一样。批里的行顺序决定了
        logits / 采样结果跟请求的对应关系（TP 下还要跨 rank 通信），顺序不一致会直接串位，
        所以必须排序把它固定下来（另见 core.py 中 Batch.reqs 的说明）。
        """
        if not self.runnable:
            return None
        return Batch(reqs=sorted(self.running_reqs, key=lambda req: req.uid), phase="decode")

    @property
    def runnable(self) -> bool:
        """集合非空 = 有活可干；overlap_loop 用它决定这一轮要不要阻塞等消息。"""
        return len(self.running_reqs) > 0
