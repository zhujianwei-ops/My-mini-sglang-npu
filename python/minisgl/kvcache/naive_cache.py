"""一个"什么都不缓存"的前缀缓存实现，由 `--cache-type naive` 选中。

它的所有方法都退化成空操作：匹配永远返回 0 长度、插入永远返回 0、锁是 no-op、
容量信息恒为 0。用途是**把前缀复用彻底关掉**（对比 radix_cache.py 那套前缀树），
常见于想做基准对比、或想排除"前缀命中"这个变量的时候。

关掉之后有两个连带效果：

  - CacheManager.available_size 变成"整个空闲池"，而且因为 evict 不可用、缓存也不会
    占用任何页，请求只要能进来就一定能跑完（不存在"跑到一半空间被别人抢走"的问题）；
  - 页不会漏：请求结束时 CacheManager.cache_req 仍会走一遍，把整段页还回 free_slots
    （见下面 insert_prefix 返回 0 长度后 cache_req 里的分支）。
"""

import torch

from .base import BaseCacheHandle, BasePrefixCache, InsertResult, MatchResult, SizeInfo


class NaiveCacheHandle(BaseCacheHandle):
    # 类属性：由 NaivePrefixCache 建好实例后写进来（见下面 __init__）。
    # 因为 handle 是 frozen dataclass、且 cached_len 恒为 0，这里只需要一个全局空张量
    empty_tensor: torch.Tensor  # should be set by NaivePrefixCache

    def __init__(self):
        super().__init__(cached_len=0)  # 永远"没命中"，所以长度恒为 0

    def get_matched_indices(self) -> torch.Tensor:
        return self.empty_tensor


class NaivePrefixCache(BasePrefixCache):
    def __init__(self, device: torch.device):
        self.device = device
        self.empty_tensor = torch.empty(0, dtype=torch.int32, device=device)
        # 写到类属性上给所有 handle 共用；同进程建第二个实例会覆盖它，但反正都指向
        # 一个长度为 0 的张量，没有实际影响（写法偏脏，注释在此免得看的人以为有深意）
        NaiveCacheHandle.empty_tensor = self.empty_tensor
        super().__init__()

    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        pass  # 没有东西需要保护：任何请求都拿不到缓存里的槽位

    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        # 永远不命中：返回一个 cached_len = 0 的空句柄
        return MatchResult(NaiveCacheHandle())

    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> InsertResult:
        # 什么都不记（cached_len = 0 → 调用方要释放的那段是空的），也没换句柄
        return InsertResult(0, NaiveCacheHandle())

    def evict(self, size: int) -> torch.Tensor:
        if size == 0:
            return self.empty_tensor  # 基类约定：evict(0) 永远安全
        # 要淘汰就说明池子不够大了 —— 但 naive 没有可淘汰的东西，只能直接报错
        raise NotImplementedError("NaiveCacheManager does not support eviction.")

    def reset(self) -> None:
        pass

    @property
    def size_info(self) -> SizeInfo:
        # 缓存不持有任何 token，于是 CacheManager.available_size = 整个空闲池
        return SizeInfo(evictable_size=0, protected_size=0)

    def check_integrity(self) -> None:
        pass
