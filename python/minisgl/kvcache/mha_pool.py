"""KV 池：显存里那块真正存 K/V 的张量（MHA 版）。

这里的 "池" 就是一块大显存，按**槽位**（slot）划分：一个槽位存一个 token 在某一层的一个
KV 头组。槽位下标是扁平的 —— 一页占 page_size 个连续槽位：

    _kv_buffer        [2, num_layers, num_pages, page_size, local_kv_heads, head_dim]
                       ↑ K/V          ↑ 页号    ↑ 页内偏移   ↑ 本 rank 的 KV 头
    _storage_shape    [num_pages * page_size, local_kv_heads, head_dim]
                       └─ 把"页号 + 页内偏移"拍平成一维 → 这就是页表里那个槽位

关键点：**一页在物理上就是一块连续内存**（`(page_size, heads, dim)`）。所以页表里只存
每页的起始槽位，就能定位整页 —— 这正是 scheduler/cache.py 里那条"页内必须连续"约束的
由来，也是 attention 后端能只取 `page_table[row, :seqlen:page_size]` 就去读整页的原因。
"""

from __future__ import annotations

import torch
from minisgl.distributed import get_tp_info
from minisgl.utils import div_even

from .base import BaseKVCachePool


class MHAKVCache(BaseKVCachePool):
    """
    Base class for key-value caches.
    This class defines the interface for key-value caches used in LLMs.
    """
    # ↑ 这段 docstring 是从 BaseKVCachePool 抄过来的，其实这里是**具体实现**（不是基类）：
    #   目前代码里唯一一种 KV 池，MHA（含 GQA/MQA，因为头数可以是每 rank 分到的份数）

    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        tp_info = get_tp_info()
        # TP 切分：KV 头按 rank 均分（每 rank 只存自己那份，attention 各算各的头）。
        # allow_replicate=True 处理"KV 头比 rank 还少"的情况：能整除时本 rank 只留 1 个头
        # （大 TP + GQA 时会出现，头数被复制而不是切分）
        local_kv_heads = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        # 一块连续显存装下所有层、所有页的 K 和 V：第 0 维区分 K/V，第 5 维是 TP 后的头数
        self._kv_buffer = torch.empty(
            (2, num_layers, num_pages, page_size, local_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )
        self._num_layers = num_layers
        # K/V 各是一半张量（同一块显存的两个视图，不额外占空间）
        self._k_buffer = self._kv_buffer[0]
        self._v_buffer = self._kv_buffer[1]
        self._device = device
        # 把 (页号, 页内偏移) 拍平成"槽位"一维：槽位 s 属于第 s // page_size 页
        self._storage_shape = (num_pages * page_size, local_kv_heads, head_dim)

    def k_cache(self, index: int) -> torch.Tensor:
        """第 index 层的 K 张量（未拍平，形状 [num_pages, page_size, heads, dim]）。"""
        return self._k_buffer[index]

    def v_cache(self, index: int) -> torch.Tensor:
        """第 index 层的 V 张量。"""
        return self._v_buffer[index]

    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None:
        """把这一层算出的 k/v 写进 out_loc 指定的槽位（一个 token 一个槽位）。

        view 成 _storage_shape 之后，槽位就是一个扁平的 token 下标，kernel 拿去 scatter：
        k_cache[out_loc[i]] = k[i]。
        """
        from minisgl.kernel import store_cache

        store_cache(
            k_cache=self._k_buffer[layer_id].view(self._storage_shape),
            v_cache=self._v_buffer[layer_id].view(self._storage_shape),
            indices=out_loc,
            k=k,
            v=v,
        )

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._kv_buffer.dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers
