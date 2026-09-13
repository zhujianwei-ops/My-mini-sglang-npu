"""prefill 阶段的两件事：排队，以及把请求切成能吃下的块。

     PrefillManager   待 prefill 的 FIFO 队列：新请求进来先排队（不占任何资源），
                      每轮调度时按顺序往后取，能塞几个塞几个。
     PrefillAdder     一次调度用的临时记账本：token 预算和 KV 空间都在它手里消耗。

两个约束决定了这里的复杂度：

  1. 单次 forward 的 token 数有预算（`max_extend_tokens`），长 prompt 装不下就得切块，
     用 ChunkedReq 表示"这一块不算完，下一轮接着来"；
  2. KV 空间要预留：既要给正在 decode 的请求留出它们未来要长的部分，也要给新请求
     预留它"最终会长到多大"，否则请求跑到一半才发现放不下就没法收拾了。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Tuple

import torch
from minisgl.core import Batch, Req
from minisgl.utils import init_logger

from .utils import PendingReq

if TYPE_CHECKING:
    from minisgl.kvcache import BaseCacheHandle
    from minisgl.message import UserMsg

    from .cache import CacheManager
    from .decode import DecodeManager
    from .table import TableManager

logger = init_logger(__name__)


class ChunkedReq(Req):
    """被切块的 prefill 请求（一块 = 一次 forward 能吃下的那部分 prompt）。

    和普通 Req 只差两处，都是为了表示"这次 forward 不产出 token"：

      - `append_host` 直接报错：没有采样结果，谁都不该往 host 序列里追加；
      - `can_decode` 恒为 False：于是它进不了 decode 集合（DecodeManager.filter_reqs 会
        过滤掉），`_make_write_tuple` 也会给它写 -1 哨兵，采样结果被丢弃。

    分块进度靠 `cached_len` 传递：每次 forward 后 Engine 会调 `complete_one()`，把
    cached_len 追平到 device_len，所以下一块可以直接拿这个 cached_len 当起点
    （见 PrefillAdder._add_one_req）。

    顺带一提：切块时 max_device_len 只按这一块算，所以对 ChunkedReq 来说 remain_len
    之类的数字没有意义——但 can_decode 恒为 False，调度器不会去用它。
    """

    def append_host(self, next_token: torch.Tensor) -> None:
        raise NotImplementedError("ChunkedReq should not be sampled")

    @property
    def can_decode(self) -> bool:
        return False  # avoid being added to decode manager


@dataclass
class PrefillAdder:
    """一次调度过程里的记账本：token_budget 是还能放多少 token 进这批，
    reserved_size 是"已经答应出去、将来要被占用的 KV 空间"。

    它是每轮调度临时建的（见 PrefillManager.schedule_next_batch），所以这里消耗掉的
    预算不需要还原。
    """

    token_budget: int  # 本批还能塞多少 token，每放一个请求就减掉它的块大小
    reserved_size: int  # 已承诺的 KV 空间：在飞 decode 的 + 本批已放入请求的
    cache_manager: CacheManager
    table_manager: TableManager

    def _try_allocate_one(self, req: PendingReq) -> Tuple[BaseCacheHandle, int] | None:
        """给一个新请求占位：查前缀缓存 → 锁定命中前缀 → 分配 table slot。

        成功返回 (缓存句柄, table_idx)，空间不够或没 slot 了返回 None。
        """
        # 每个在跑的请求占 page_table / token_pool 的一行，行数有上限
        if self.table_manager.available_size == 0:
            return None

        # TODO: consider host cache match case
        # 查前缀缓存看能命中多少 token（match_req 只匹配到 input_len-1，见 cache.py）
        handle = self.cache_manager.match_req(req).cuda_handle
        cached_len = handle.cached_len
        # TODO: better estimate policy
        # 预估这个请求最终要占多少 KV：剩下来要算的 prompt + 全部输出长度
        extend_len = req.input_len - cached_len
        estimated_len = extend_len + req.output_len

        # 第一道检查：放不下就先别进来，免得跑到一半空间不够
        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            return None
        # 注意 lock 本身就会吃掉 available_size：锁住前缀 = 把这部分页从"可驱逐"
        # 变成"受保护"，于是 available_size 变小，所以锁完必须再查一次。
        # 超了就解锁回滚——unlock 返回 None，这一行等于"回滚并返回失败"
        self.cache_manager.lock(handle)
        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            return self.cache_manager.unlock(handle)

        table_idx = self.table_manager.allocate()
        if cached_len > 0:  # NOTE: set the cached part
            # 把命中前缀的 token id 和物理页一起搬进本请求自己的行。
            # 页表这一行是必须的：attention 后端就是按 page_table[table_idx] 整行去读
            # 前缀 KV 的（见 attention/fa.py、attention/fi.py）；token id 一起搬是为了
            # 让这一行始终是完整的序列（分块 prefill 续块时只补后面的部分）
            device_ids = self.table_manager.token_pool[table_idx][:cached_len]
            page_entry = self.table_manager.page_table[table_idx][:cached_len]
            # 先 pin 一份再拷，异步 H2D 才真的异步（见 _add_one_req 里同样的写法）
            device_ids.copy_(req.input_ids[:cached_len].pin_memory(), non_blocking=True)
            page_entry.copy_(handle.get_matched_indices())

        return handle, table_idx

    def _add_one_req(
        self,
        pending_req: PendingReq,
        cache_handle: BaseCacheHandle,
        table_idx: int,
        cached_len: int,
    ) -> Req:
        """把请求（可能只是它的一块）变成真正的 Req 加进本批。

        cached_len 是"这个请求已经算好的前缀长度"：命中缓存的部分，或前几块 prefill
        算完的部分。它之后的部分才是本次要算的。
        """
        remain_len = pending_req.input_len - cached_len  # prompt 里还没算的
        chunk_size = min(self.token_budget, remain_len)  # 这块最多能算多少
        is_chunked = chunk_size < remain_len  # 还有剩 → 只能算一块，用 ChunkedReq 标记
        CLS = ChunkedReq if is_chunked else Req
        self.token_budget -= chunk_size
        # 预留量按"整个请求最终要占的 KV"算（剩余 prompt + 全部输出），而不是只看这一块：
        # 否则同一轮后面的请求会把这部分空间重复算进 available_size，等于超卖
        self.reserved_size += remain_len + pending_req.output_len
        # NOTE: update the tokens ids only; new pages will be allocated in the scheduler
        # 这里只把这一个块的 token id 拷进 token_pool；页表留到 scheduler 的
        # CacheManager.allocate_paged 去填（那时才由 _prepare_batch 定下 device_len）
        _slice = slice(cached_len, cached_len + chunk_size)
        device_ids = self.table_manager.token_pool[table_idx, _slice]
        device_ids.copy_(pending_req.input_ids[_slice].pin_memory(), non_blocking=True)
        return CLS(
            # input_ids 截到这一块为止：Req.device_len 就是这么算出来的，
            # 因此 __post_init__ 里 cached_len < device_len 的断言成立
            input_ids=pending_req.input_ids[: cached_len + chunk_size],
            table_idx=table_idx,
            cached_len=cached_len,
            output_len=pending_req.output_len,
            uid=pending_req.uid,
            cache_handle=cache_handle,
            sampling_params=pending_req.sampling_params,
        )

    def try_add_one(self, pending_req: PendingReq) -> Req | None:
        """尝试把一个待处理请求放进本批；返回 None 表示放不下（预算或空间不够）。"""
        # 1. 预算用光就不必再试了（调用方收到 None 会停止后续尝试）
        if self.token_budget <= 0:
            return None

        # 2. 上一轮把它切块了：这一块接着用原来的 slot 和缓存句柄，起点是上一块的进度
        if chunked_req := pending_req.chunked_req:
            return self._add_one_req(
                pending_req=pending_req,
                cache_handle=chunked_req.cache_handle,
                table_idx=chunked_req.table_idx,
                # 这个 cached_len 已被 complete_one 追平成"上一块算到哪了"
                cached_len=chunked_req.cached_len,
            )

        # 3. 新请求：先占位（查缓存 / 锁前缀 / 分 slot），再生成 Req
        if resource := self._try_allocate_one(pending_req):
            cache_handle, table_idx = resource
            return self._add_one_req(
                pending_req=pending_req,
                cache_handle=cache_handle,
                table_idx=table_idx,
                cached_len=cache_handle.cached_len,  # 命中前缀的长度
            )

        return None


@dataclass
class PrefillManager:
    """待 prefill 请求的队列（FIFO）。

    `pending_list` 里只有"还没 prefill 完"的请求；已经进 decode 的由 DecodeManager
    管，被取消的在 abort_req 里摘掉。每个 PendingReq 上的 chunked_req 字段用来跨轮
    传递分块状态（None 表示这是个还没开始的新请求）。
    """

    cache_manager: CacheManager
    table_manager: TableManager
    decode_manager: DecodeManager
    pending_list: List[PendingReq] = field(default_factory=list)

    def add_one_req(self, req: UserMsg) -> None:
        """收到新请求：只是入队，此时还不占 table slot、也不占 KV 空间。"""
        self.pending_list.append(PendingReq(req.uid, req.input_ids, req.sampling_params))

    def schedule_next_batch(self, prefill_budget: int) -> Batch | None:
        """按 FIFO 顺序尽量多塞几个请求进这一批 prefill；没得跑返回 None。"""
        if len(self.pending_list) == 0:
            return None

        # estimated offset due to in-flight decode
        # 给正在 decode 的请求留出空间：它们每个还会继续长 remain_len 个 token
        # （另外每个请求多留一页，见 DecodeManager.inflight_tokens）
        adder = PrefillAdder(
            token_budget=prefill_budget,
            reserved_size=self.decode_manager.inflight_tokens,
            cache_manager=self.cache_manager,
            table_manager=self.table_manager,
        )
        reqs: List[Req] = []
        chunked_list: List[PendingReq] = []
        for pending_req in self.pending_list:
            if req := adder.try_add_one(pending_req):
                # 放进去了：先清掉分块标记，只有"这次仍然没算完"才重新记上
                pending_req.chunked_req = None
                if isinstance(req, ChunkedReq):
                    # 记住这块的状态，下一轮从它接着算
                    pending_req.chunked_req = req
                    chunked_list.append(pending_req)
                reqs.append(req)
            else:
                break  # We cannot add more requests
                # 严格 FIFO：放不下就整批停住。后面的请求只会排在更后面，跳过它去装
                # 别人会破坏公平性，让先到的请求一直排不上
        if len(reqs) == 0:
            return None
        # 重排队列：[还要接着分块的] + [没轮到的]。
        # 切块的排最前面——它们已经占着 slot 和缓存句柄，下一轮优先续上；
        # 两段内部都保持原有顺序
        self.pending_list = chunked_list + self.pending_list[len(reqs) :]
        return Batch(reqs=reqs, phase="prefill")

    def abort_req(self, uid: int) -> Req | None:
        """把被取消的请求从等待队列里摘掉，并返回需要释放资源的那个 Req。

        已经分过块的请求占着 table slot 和缓存句柄，调用方要拿返回的 ChunkedReq 去
        `_free_req_resources`；还没开始的请求返回 None（没有资源要收）。
        """
        for i, req in enumerate(self.pending_list):
            if req.uid == uid:
                self.pending_list.pop(i)
                return req.chunked_req
        return None

    @property
    def runnable(self) -> bool:
        """队列非空 = 有活可干；overlap_loop 用它决定这一轮要不要阻塞等消息。"""
        return len(self.pending_list) > 0
