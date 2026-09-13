"""CacheManager：把"前缀缓存"（kvcache/，只认 token 序列）和"物理页分配"接到一起的中间层。

底层的前缀缓存只知道"哪些 token 算过 KV、KV 落在哪些槽位"；上层调度器要的是能直接读写
的页表。这一层负责三件事：

    free_slots / page_table   物理页的分配回收、以及往页表里填槽位
    available_size            还有多少空间可用（可淘汰的 + 完全空闲的）
    lock / unlock             保护某个请求正在读写的前缀，别被别的请求淘汰掉

页（page）是空间管理的最小单位，一页 = 连续 page_size 个 token 的 KV：

  - 分配按整页做（allocate_paged）：请求新覆盖的 token 若跨页，就整页给它；
  - free_slots 里存的是每页的起始 token 槽位，永远是 page_size 的整数倍；
  - 前缀缓存插入/淘汰也只按整页算，不足一页的尾巴不缓存（见 cache_req）。

这里有个贯穿全文的布局约束：**同一页内的槽位必须在物理上连续**。因为 attention 后端
只从页表里取每页的起点（例如 fa.py 里 `page_table[row, :seqlen:page_size]`），然后按
连续内存读整页 —— 页内不连续就会读到别人的 KV。_page_to_token 就是为它服务的。
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, List, Tuple

import torch
from minisgl.core import Req
from minisgl.kvcache import BaseCacheHandle, MatchResult, create_prefix_cache
from minisgl.utils import div_ceil

if TYPE_CHECKING:
    from .utils import PendingReq


class CacheManager:
    def __init__(self, num_pages: int, page_size: int, page_table: torch.Tensor, type: str):
        # The `_free_slots` follows a page-aligned manner. For example, if page_size = 2,
        # the `_free_slots` may look like [0, 2, 4, 6, ...], and each slot represents a page.
        # 空闲页用"页起始槽位"表示：初始化就是所有页的起点 [0, p, 2p, ...]
        device = page_table.device
        self.free_slots = torch.arange(num_pages, dtype=torch.int32, device=device) * page_size
        # 前缀缓存实现由 type 选（registry 见 kvcache/__init__.py：radix / naive）
        self.prefix_cache = create_prefix_cache(device=device, type=type)
        self.device = device
        self.num_pages = num_pages
        # [max_running_reqs, max_seq_len]：行 = 请求的 table_idx，列 = token 位置，
        # 值 = 该 token 的 KV 落在哪个物理槽位
        self.page_table = page_table
        self.page_size = page_size

    def match_req(self, req: PendingReq) -> MatchResult:
        """查这个请求的 prompt 能命中多长的前缀缓存。

        只拿 input_ids[:input_len-1] 去匹配：最后一个 token 故意不匹配 —— 命中了就等于
        它的 KV 已经算好，本次 forward 无事可做（Req 也要求 cached_len < device_len，
        至少要留 1 个 token 给它）。
        """
        input_len = req.input_len
        assert input_len > 0, "Input length must be greater than 0."
        return self.prefix_cache.match_prefix(req.input_ids[: input_len - 1])

    @property
    def available_size(self) -> int:
        """还能放多少 token 的 KV：可淘汰的 + 完全空闲的。

        受保护的部分（有请求锁着、正在读写的 KV）不算进来 —— 那些动不得。
        """
        return self.prefix_cache.size_info.evictable_size + len(self.free_slots) * self.page_size

    def lock(self, handle: BaseCacheHandle) -> None:
        """锁住命中的前缀：这部分 KV 马上要被请求读写，不能被别人淘汰掉。

        注意锁本身就会吃掉 available_size（页从可淘汰变成受保护），所以调用方
        锁完得再查一次空间（见 PrefillAdder._try_allocate_one 里的双重检查）。
        """
        self.prefix_cache.lock_handle(handle, unlock=False)

    def unlock(self, handle: BaseCacheHandle) -> None:
        """解锁。返回 None，所以 `return self.cache_manager.unlock(handle)` 就是
        "回滚并返回失败"的惯用写法（见 PrefillAdder._try_allocate_one）。"""
        self.prefix_cache.lock_handle(handle, unlock=True)

    def allocate_paged(self, reqs: List[Req]) -> None:
        """给每个请求本次新覆盖的 [cached_len, device_len) 分配物理页，写进它的页表行。

        按页换算（div_ceil）：
          - first_page 从 div_ceil(cached_len) 起：cached_len 所在的那一页早就分配好了
            （prefill 第一块或上一轮 decode 分配时，整页的位置是一起写好的），
            所以从它后面那一页开始；
          - last_page 用 div_ceil(device_len)：把最后那个不满一页的部分也包进来。
        先统计总共要几页，一次性分配（_allocate 里可能触发淘汰），再统一填页表。
        """
        needed_pages = 0
        allocation_info: List[Tuple[int, int, int]] = []
        for req in reqs:
            first_page = div_ceil(req.cached_len, self.page_size)
            last_page = div_ceil(req.device_len, self.page_size)
            if last_page > first_page:  # 没跨页就不用分配（decode 大多数轮次走这里）
                needed_pages += last_page - first_page
                allocation_info.append((req.table_idx, first_page, last_page))
        if needed_pages > 0:
            allocated = self._page_to_token(self._allocate(needed_pages))
            _write_page_table(self.page_table, allocated, allocation_info, self.page_size)

    def cache_req(self, req: Req, *, finished: bool) -> None:
        """收尾：把请求算好的 KV 前缀插进前缀缓存，并处理随之而来的空间回收。

        finished=False 用于请求刚 prefill 完（后面还要继续 decode）；
        finished=True 用于请求彻底结束（见 Scheduler._free_req_resources）。

        上面那段英文注释画的是几个 token 区间的关系：这次要插的是 [0, cached_len)，
        其中 [0, old_handle.cached_len) 在 prefill 之前就已经在缓存里了。插入时才知道
        "别人抢先插过"的那一段，以及插不进缓存的尾巴，都是本请求多占的，要还给 free_slots。
        """
        insert_ids = req.input_ids[: req.cached_len]
        # page_table 这一行是视图，insert_prefix 里会 clone 需要长期保留的部分
        page_indices = self.page_table[req.table_idx, : req.cached_len]
        old_handle = req.cache_handle
        # 塞进前缀缓存，返回 (插入前就已缓存多长, 新句柄)。
        # 注意缓存里存的是"token → 槽位"的映射，槽位本身是和本请求共用的
        cached_len, new_handle = self.prefix_cache.insert_prefix(insert_ids, page_indices)
        # unlock until all operations on handle is done
        # 先解锁旧句柄：它保护的范围已完全被新句柄覆盖，留着会重复计数、永远淘汰不掉。
        # 这中间不会有淘汰发生（调度器单线程，且这几步都不分配空间）
        self.unlock(old_handle)
        # this part is already in the prefix cache, free it
        # [old, cached_len) 现在归缓存所有（插入时发现别人先插了），自己那份要还掉；
        # 不还的话同一批页被两边各算一次，页数就对不上（check_integrity 能查出来）
        self._free(page_indices[old_handle.cached_len : cached_len])
        if finished:  # this tail part should be freed
            # 请求结束了：插不进缓存的那截尾巴（不足一页）以后没人再用，还掉
            self._free(page_indices[new_handle.cached_len :])
        else:  # keep the tail part, update the handle
            # 还要继续 decode：尾巴留着（马上要在上面写新算的 KV），换上新句柄并锁住 ——
            # 锁的是 root 到新节点的整条路径，于是请求正在读的 [0, cached_len) 也受保护
            req.cache_handle = new_handle
            self.lock(new_handle)

    def check_integrity(self) -> None:
        """自检：每个页要么在 free_slots 里，要么被前缀缓存持有，两边的页数必须配平。

        调度器空闲时调用（Scheduler.run_when_idle），用来抓"页漏了 / 记重了"这类问题：
        前缀缓存的容量记账和 free_slots 是两套独立维护的数字，靠这条恒等式互相验证。

        对不上的话基本都是 cache_req 里那几笔回收没做干净。
        """
        self.prefix_cache.check_integrity()
        cache_pages = self.prefix_cache.size_info.total_size // self.page_size
        if len(self.free_slots) + cache_pages != self.num_pages:
            raise RuntimeError(
                "CacheManager integrity check failed:"
                f" free_pages({len(self.free_slots)}) +"
                f" cache_pages({cache_pages}) != num_pages({self.num_pages})"
            )
        if self.page_size > 1:
            # 严格守住"free_slots 只存页起点"这一条，否则整页分配的前提就破了
            assert torch.all(self.free_slots % self.page_size == 0)

    @contextmanager
    def lazy_free_region(self):
        """区间内 `_free` 变成"只记账、不归还"，退出时一次性 cat 回 free_slots。

        用途见 Scheduler._process_last_data：一轮里可能有多个请求同时结束，每次 _free
        都是一次 torch.cat（重新分配 + 拷贝），攒起来一次做完能省掉这些零碎开销。

        实现是拿闭包把实例属性 self._free 盖住（覆盖掉类里的方法），退出时 del 恢复。
        """

        def lazy_free(indices: torch.Tensor) -> None:
            lazy_free_list.append(indices[:: self.page_size])  # 只留页起点

        lazy_free_list: List[torch.Tensor] = []
        try:
            self._free = lazy_free
            yield
        finally:
            # 出异常也要把攒下的页还回去，所以写 finally；del 之后 self._free 又变回类方法
            del self._free
            self.free_slots = torch.cat([self.free_slots] + lazy_free_list)

    def _allocate(self, needed_pages: int) -> torch.Tensor:
        """取 needed_pages 个空闲页，返回它们的页起始槽位；不够就先从前缀缓存淘汰。

        淘汰的粒度也是页：evict 按 token 数要空间，拿回来的 value 里每页的槽位是连续的，
        所以每隔 page_size 取一个正好是页起点。
        """
        if needed_pages > (free_pages := len(self.free_slots)):
            evicted = self.prefix_cache.evict((needed_pages - free_pages) * self.page_size)
            self.free_slots = torch.cat([self.free_slots, evicted[:: self.page_size]])
            assert len(self.free_slots) >= needed_pages, "Eviction did not free enough space."
        allocated = self.free_slots[:needed_pages]
        self.free_slots = self.free_slots[needed_pages:]  # 从头部取走 = 占用
        return allocated

    def _free(self, indices: torch.Tensor) -> None:
        """把 indices 覆盖的页还回 free_slots（只留每页起始槽位）。

        传进来的区间一定对齐到整页：它的两端都来自句柄的 cached_len，而句柄长度恒为
        page_size 的整数倍（见 radix_cache.insert_prefix 的 align_down）。所以每隔
        page_size 取一个就是页起点。
        """
        if len(indices) > 0:
            self.free_slots = torch.cat([self.free_slots, indices[:: self.page_size]])

    def _page_to_token(self, pages: torch.Tensor) -> torch.Tensor:
        """把页起始槽位展开成整页的槽位：[p*s, p*s+1, ..., p*s+p-1]。

        页内必须连续，这是 attention 后端的前提：它只取每页起点，然后按连续内存读整页
        （fa.py / trtllm.py 里的 `page_table[row, :seqlen:page_size]`）。
        """
        if self.page_size == 1:
            return pages
        # [X * page_size] -> [X * page_size, ..., X * page_size + page_size - 1]
        offsets = torch.arange(self.page_size, device=self.device, dtype=torch.int32)
        return (pages.unsqueeze(1) + offsets).flatten()


def _write_page_table(
    page_table: torch.Tensor,
    allocated: torch.Tensor,
    allocation_info: List[Tuple[int, int, int]],
    page_size: int,
) -> None:
    """把 allocated 里的槽位填进 page_table 对应的 (行, token 位置)。

    要写的位置先在 host 侧拼成两个索引数组（pinned 内存，异步拷到 device），再一次性
    scatter 过去 —— 比每个请求单独写一次 page_table 少很多 kernel 启动。
    """
    needed_tokens = len(allocated)
    table_idx_host = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=True)
    positions_host = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=True)
    offset = 0
    for table_idx, first_page, last_page in allocation_info:
        first_pos, last_pos = first_page * page_size, last_page * page_size
        length = last_pos - first_pos
        # 这一段 length 个位置都属于同一个请求行，且 [first_pos, last_pos) 是整页边界
        table_idx_host[offset : offset + length].fill_(table_idx)
        torch.arange(first_pos, last_pos, out=positions_host[offset : offset + length])
        offset += length
    assert offset == needed_tokens, "Mismatch in allocated tokens and filled tokens."
    table_idxs = table_idx_host.to(page_table.device, non_blocking=True)
    offsets = positions_host.to(page_table.device, non_blocking=True)
    page_table[table_idxs, offsets] = allocated
