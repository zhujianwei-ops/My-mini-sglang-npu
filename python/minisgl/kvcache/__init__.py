"""KV 缓存包的出口：两个工厂 + 两个抽象的转发。

这层负责"按名字造对象"，把实现细节挡在外面：

    create_kvcache_pool(...)   造 KV 池（存储），目前只有 MHA 一种
    create_prefix_cache(...)   造前缀缓存（目录），由 --cache-type 选 radix / naive
                               （默认 radix，见 scheduler/config.py:cache_type）

两个抽象对应两件不同的事，别混：

    BaseKVCachePool    显存里那块真正的 K/V 张量（存数据）
    BasePrefixCache    哪些 token 算过 KV、落在哪些槽位（做索引）

新增一种前缀缓存只要写个类 + 用 @SUPPORTED_CACHE_MANAGER.register("名字") 注册，
--cache-type 的可选值（choices）会自动跟着变，不用改别的地方。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from minisgl.utils import Registry

if TYPE_CHECKING:
    import torch
    from minisgl.models import ModelConfig

from .base import (
    BaseCacheHandle,
    BaseKVCachePool,
    BasePrefixCache,
    MatchResult,
    SizeInfo,
)


class CacheManagerCreator(Protocol):
    """工厂函数的形状：只收一个 device，返回 BasePrefixCache。

    正因如此，实现（如 RadixPrefixCache）拿不到 page_size，得从全局上下文
    get_global_ctx() 里取（见 radix_cache.py）。
    """

    def __call__(self, device: torch.device) -> BasePrefixCache: ...


# 名字 → 工厂函数 的注册表；--cache-type 的 choices 就是 supported_names()
SUPPORTED_CACHE_MANAGER = Registry[CacheManagerCreator]("Cache Manager")


def create_kvcache_pool(
    model_config: ModelConfig,
    num_pages: int,
    page_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> BaseKVCachePool:
    """造 KV 池。目前只有 MHA 版，所以这里没有 registry，直接 new。"""
    from .mha_pool import MHAKVCache  # TODO: support other variants (e.g. MLA)

    return MHAKVCache(
        num_kv_heads=model_config.num_kv_heads,
        num_pages=num_pages,
        page_size=page_size,
        num_layers=model_config.num_layers,
        head_dim=model_config.head_dim,
        device=device,
        dtype=dtype,
    )


@SUPPORTED_CACHE_MANAGER.register("naive")
def create_naive_cache(device: torch.device):
    """不缓存任何前缀（见 naive_cache.py）。"""
    from .naive_cache import NaivePrefixCache

    return NaivePrefixCache(device=device)


@SUPPORTED_CACHE_MANAGER.register("radix")
def create_radix_cache(device: torch.device):
    """前缀树实现（见 radix_cache.py），默认选它。"""
    from .radix_cache import RadixPrefixCache

    return RadixPrefixCache(device=device)


def create_prefix_cache(device: torch.device, type: str) -> BasePrefixCache:
    """按名字造前缀缓存。名字不在表里会 KeyError（CLI 那边已用 choices 限制过）。"""
    # 实现放在函数体里延迟导入：用到哪个才导入哪个
    return SUPPORTED_CACHE_MANAGER[type](device)


__all__ = [
    "create_kvcache_pool",
    "create_prefix_cache",
    "BaseKVCachePool",
    "BaseCacheHandle",
    "BasePrefixCache",
    "SizeInfo",
    "MatchResult",
    "SUPPORTED_CACHE_MANAGER",
]
