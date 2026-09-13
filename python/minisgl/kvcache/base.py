"""KV 缓存的两个抽象层：存 K/V 的**池子**，和查前缀的**目录**。

这个包里有两个名字很像、但管的是完全不同事的东西：

    BaseKVCachePool     显存里那块真正的 K/V 张量，按"槽位"（slot）存一个 token 的 K/V。
                        实现只有 MHAKVCache（mha_pool.py）。
    BasePrefixCache     "哪些 token 算过 KV、KV 落在哪些槽位"的索引。实现有两个：
                        radix（压缩前缀树，radix_cache.py）和 naive（完全不缓存，
                        naive_cache.py），由 --cache-type 选。

一句话：池子是**存储**，前缀缓存是**目录**。目录里记的槽位最终都指向池子里的位置；
把两者接起来、并负责页分配回收的是 scheduler/cache.py 里的 CacheManager。

本文件只放接口和约定，实现见同目录其它文件。几条容易踩的约定：

  - 句柄（BaseCacheHandle）给出的槽位，必须先**锁住**它才能用，否则可能被 evict 掉；
  - match_prefix 不改缓存，insert_prefix / evict 会改；
  - evict(0) 永远安全；实际淘汰量可能大于请求量（淘汰以整段/整页为粒度）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import NamedTuple

import torch


class BaseKVCachePool(ABC):
    """
    Base class for key-value caches.
    This class defines the interface for key-value caches used.
    """

    # 取第 index 层的 K / V 张量（index 是层号，不是页号）。
    # 返回的张量把"页"和"页内偏移"拍平成了一维槽位：一页 = page_size 个连续槽位，
    # 所以一张页表里只存每页起始槽位，就能定位整页（见 mha_pool.py 的 _storage_shape）
    @abstractmethod
    def k_cache(self, index: int) -> torch.Tensor: ...

    @abstractmethod
    def v_cache(self, index: int) -> torch.Tensor: ...

    # 把这一层刚算出来的 k/v 写进 out_loc 指定的槽位（每个 token 一个槽位）
    @abstractmethod
    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None: ...

    @property
    @abstractmethod
    def device(self) -> torch.device: ...

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype: ...

    @property
    @abstractmethod
    def num_layers(self) -> int: ...


@dataclass(frozen=True)
class BaseCacheHandle(ABC):
    """前缀缓存的句柄：一次匹配（或一次插入）的结果，也是后续操作的凭据。

    frozen：句柄一旦发出去就代表"那一刻的状态"，不该被就地改（要换就整只换，
    例如 Req.cache_handle）。cached_len 是它能覆盖的 token 数：

        match_prefix 返回的句柄      = 匹配到的长度（可能是 0）
        insert_prefix 返回的句柄     = 插入之后缓存里的总长度

    拿到句柄后要做两件事：先 lock，再用 get_matched_indices() 拿槽位。
    """

    cached_len: int

    # 返回这段前缀每个 token 的物理槽位（长度 = cached_len，按 token 顺序）。
    # 调用方会把它整段拷进请求自己的页表行（见 PrefillAdder._try_allocate_one），
    # 所以顺序和长度必须严格对上
    @abstractmethod
    def get_matched_indices(self) -> torch.Tensor: ...


class SizeInfo(NamedTuple):
    """容量信息，单位是 token。两者的区别只在"能不能被 evict 动"：

    evictable   没人用的部分，随时可以淘汰掉腾出空间
    protected   有请求锁着的部分（正在读写它的 KV），动不得
    """

    evictable_size: int
    protected_size: int

    @property
    def total_size(self) -> int:
        """缓存持有的全部 token 数；CacheManager.check_integrity 用它和空闲页配平。"""
        return self.evictable_size + self.protected_size


class InsertResult(NamedTuple):
    cached_len: int  # length already in cache before insertion (should be freed)
    # ↑ 插入前就已在缓存里的长度。这段内容现在归缓存所有，调用方手里那份槽位要释放掉，
    #   否则同一批页被两边各记一次（见 CacheManager.cache_req）
    handle: BaseCacheHandle  # cache handle for the inserted prefix


class MatchResult(NamedTuple):
    cuda_handle: BaseCacheHandle
    # 名字里的 cuda 指"匹配到的槽位在 device 上"（真去读 KV 的是 GPU）。
    # TODO: support HiCache —— 以后如果加 host 侧缓存，这里会多一个 handle 字段


class BasePrefixCache(ABC):
    """前缀缓存（目录）的接口：查、插、锁、淘汰。

    它只维护"token 序列 → 物理槽位"的映射和容量记账，不碰 KV 数据本身 ——
    数据在别处（BaseKVCachePool）。
    """

    @abstractmethod
    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        """
        Lock or unlock a cache handle.
        This operation will not modify the cache, but change the size info only.
        When a handle is locked, it cannot be evicted.
        Handles must be locked before the previously-returned tensor of `match_prefix` is used.
        Otherwise it may be evicted by calling evict.

        Args:
            handle (BaseCacheHandle): The cache handle to lock or unlock.
            unlock (bool): Whether to unlock the handle. Defaults to False.
        """
        # 中文要点：锁/解锁不改缓存内容，只动容量记账（evictable ↔ protected）。
        # 锁住之后这段不会被 evict；用 match_prefix 给的槽位之前必须先锁。

    @abstractmethod
    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        """
        Match prefix and return the indices of the matched prefix in the cache.
        This operation will not modify the cache.
        The returned indices is only safe to use when the handle is locked.

        Args:
            input_ids (torch.Tensor): The input ids to match. Shape: (seq_len,)
        Returns:
            MatchResult: The match result containing the cache handles.
        """
        # 中文要点：只读操作。返回的 handle.cached_len 是匹配长度（可能是 0），
        # 拿它的槽位之前记得先 lock。

    @abstractmethod
    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> InsertResult:
        """
        Insert a new prefix into the cache.
        This operation will modify the cache.
        Args:
            input_ids (torch.Tensor): The input ids to insert. Shape: (seq_len,)
            indices (torch.Tensor): The indices to store the new prefix. Shape: (seq_len,)

        Returns:
            InsertResult: The result of the insertion.
        """
        # 中文要点：会改缓存。返回 (插入前已缓存的长度, 新句柄)：
        # 前半段现在归缓存所有，调用方要把自己那份槽位还回去（否则页数记账会重）。

    @abstractmethod
    def evict(self, size: int) -> torch.Tensor:
        """
        Evict some prefixes from the cache to free up space.
        This operation will modify the cache.
        Note that evict 0 is always safe and does nothing.
        Note that the actual evict size may be larger than the requested size.
        Args:
            size (int): The size to evict.

        Returns:
            torch.Tensor: The indices evicted. Shape: (evict_size,)
        Raises:
            RuntimeError: If the requested size is larger than the evictable size.
        """
        # 中文要点：按 token 数请求，但淘汰粒度是"一整段前缀"，所以实际可能多给；
        # size = 0 时什么都不做直接返回空张量（调用方不用特判）。

    @abstractmethod
    def reset(self) -> None:
        """Reset the cache manager and the underlying cache."""

    @property
    @abstractmethod
    def size_info(self) -> SizeInfo:
        """Get the size information of the cache."""
        # CacheManager.available_size 直接用的就是这里的 evictable_size

    @abstractmethod
    def check_integrity(self) -> None:
        """Check the integrity of the cache. Raise an error if the cache is corrupted."""
