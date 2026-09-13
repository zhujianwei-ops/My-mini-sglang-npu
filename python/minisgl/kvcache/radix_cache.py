"""前缀缓存的前缀树实现：把"算过的 KV"按 token 序列索引起来，让相同前缀的请求直接复用。

结构是一棵**压缩**前缀树（radix tree）：节点里存的不是单个 token，而是一段连续 token
（segment），所以树高远小于序列长度。

    root ──[tok0..tok7]──► A ──[tok8..tok15]──► B
                            └──[tok20..]──► C

  - 查找（match_prefix）：从 root 往下走，能走多深走多深，返回匹配长度和末端节点（句柄）。
    若某段只匹配了一半，就把它**当场劈开**（split_at），保证返回的节点总是"正好结束在
    匹配长度处"，这样插入新段时能直接往上挂。
  - 插入（insert_prefix）：把新算完的 [0, cached_len) 挂进树，只挂对得上的那一截。
  - 淘汰（evict）：从叶子往上按 LRU 删；被请求锁住的（ref_count > 0）不能动。

两个计数决定一个节点能不能被淘汰：

    ref_count    有多少个请求正"锁着"这条路径（锁是按整条 root→节点 的路径算的）
    timestamp    最后一次被访问的时间，淘汰时先扔最老的（LRU）

缓存粒度是页（page）：每个节点长度恒为 page_size 的整数倍，匹配也向下对齐到整页 ——
半页的 KV 不稳定（那一页剩下的槽位之后可能被别的请求占去），所以不缓存。
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple, TypeAlias

import torch
from minisgl.core import get_global_ctx
from minisgl.utils import align_down

from .base import BaseCacheHandle, BasePrefixCache, InsertResult, MatchResult, SizeInfo

# key 函数：把"从某位置开始的剩余 token"映射成 children 字典的 key，只看开头一小段
KEY_FN: TypeAlias = Callable[[torch.Tensor], Any]


class RadixTreeNode:
    """前缀树的一个节点 = 一段连续 token。

    它代表"root 到这里的整条 token 序列"，但自己只存这一段（_key），完整序列靠 parent
    链回溯。_key / _value / _length 在 __init__ 里只声明类型、不赋值，真正填进去是在
    set_key_value（节点可能是 split_at 造出来的，也可能先建好再填）。
    """

    counter: int = 0  # 全局自增，用来发 uuid（纯调试用，不参与任何逻辑）

    def __init__(self, key_fn: KEY_FN, tic: int | None = None) -> None:
        self.key_fn = key_fn
        self.children: Dict[Any, RadixTreeNode] = {}  # key_fn(子节点._key) -> 子节点
        self._parent: RadixTreeNode | None = None
        self.ref_count: int = 0  # 被锁的次数；> 0 表示这条路径正在被用，不能淘汰
        self.uuid = RadixTreeNode.counter
        RadixTreeNode.counter += 1
        # tic 由 split_at 透传（新节点继承原节点的时间戳）；不传就取当前时间
        self.timestamp = tic or time.monotonic_ns()

        # these fields should be updated later
        self._key: torch.Tensor
        self._value: torch.Tensor
        self._length: int

    def set_key_value(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """填上这一段 token（key）和它对应的物理 KV 槽位（value）。"""
        assert len(key) == len(value)
        self._key = key
        self._value = value
        self._length = len(key)

    def set_parent(self, parent: RadixTreeNode) -> None:
        """认父，并把自己登记进父节点的 children 表。

        key 取的是自己这段的第一页 token —— 因为往下走的时候是用"剩余 token 的开头一页"
        去查 children 的，两边算法必须一致才对得上（见 _get_key_fn）。
        """
        self._parent = parent
        parent.children[self.key_fn(self._key)] = self

    @property
    def length(self) -> int:
        return self._length

    @property
    def parent(self) -> RadixTreeNode:
        assert self._parent is not None
        return self._parent

    @property
    def value(self) -> torch.Tensor:
        return self._value

    def is_root(self) -> bool:
        return self._parent is None

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def get_match_len(self, input_ids: torch.Tensor) -> int:
        """自己和 input_ids 从头开始能对上多少 token（公共前缀长度）。

        实现在 C++（见 kernel/radix.py 的 fast_compare_key）：Python 里逐 token 比太慢，
        而一段动辄上千 token。
        """
        from minisgl.kernel import fast_compare_key

        # compare key and input_ids, find the first diff
        return fast_compare_key(self._key, input_ids)

    def split_at(self, pos: int) -> RadixTreeNode:
        """在 pos 处把自己劈成两段，返回前半段新节点，自己留后半段：

            parent ─[自己: 0..len)─►        parent ─[new: 0..pos)─►
                                                     └─[自己: pos..len)─►

        调用方（_tree_walk）保证 pos >= page_size：能走进这个节点就说明开头整整一页都
        对上了（key 就是一页 token）；也正因为 pos >= 1，下面 0 < pos 的断言才站得住。

        new_node 继承自己的 ref_count：被锁的是"到这里的整条路径"，劈开后前半段同样在
        被使用；而且 unlock 是沿路径按各节点 length 逐段减回去的，两段之和必须等于原来
        那一段，锁的容量记账才配平。
        """
        assert 0 < pos < self.length
        parent = self.parent

        # 时间戳先透传（新节点"继承"原节点的最近使用时间），由调用方决定是否刷新成现在
        new_node = RadixTreeNode(self.key_fn, self.timestamp)
        new_node.set_key_value(self._key[:pos], self._value[:pos])
        new_node.set_parent(parent)  # key 不变，正好顶掉父节点里原来指向自己的那一项
        new_node.ref_count = self.ref_count

        self.set_key_value(self._key[pos:], self._value[pos:])
        self.set_parent(new_node)  # 后半段挂到前半段下面

        return new_node

    def __lt__(self, other: RadixTreeNode) -> bool:
        """给 heapq 用：按 timestamp 排，于是堆顶总是最久没被访问的叶子（LRU）。"""
        return self.timestamp < other.timestamp


@dataclass(frozen=True)
class RadixCacheHandle(BaseCacheHandle):
    """匹配结果句柄：基类的 cached_len（匹配长度）+ 匹配到的末端节点。

    冻结是有意的：句柄按值比较、可哈希，而且它一旦发出去就代表"那一刻的匹配结果"，
    不该被就地修改（req.cache_handle 要换就整只换）。
    """

    node: RadixTreeNode

    def get_matched_indices(self) -> torch.Tensor:
        """把这条路径上每一段的物理 KV 槽位按 root → node 的顺序拼起来。

        结果长度 = 匹配到的 token 数，正好可以整段拷进请求自己的 page_table 行
        （见 PrefillAdder._try_allocate_one）。沿途要走 parent 链，所以必须在句柄还被
        锁着的时候调用 —— 否则中间某段可能已经被淘汰掉了。
        """
        node = self.node
        value_list: List[torch.Tensor] = []
        while not node.is_root():  # root 是哨兵，没有 value
            value_list.append(node.value)
            node = node.parent
        value_list.reverse()  # 自底向上收集的，反过来才是 token 顺序
        return torch.cat(value_list)


class RadixPrefixCache(BasePrefixCache):
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        # 工厂只把 device 传进来（见 kvcache/__init__.py 的 registry），页大小得从
        # 全局上下文里拿
        self.page_size = get_global_ctx().page_size
        self.key_fn = _get_key_fn(self.page_size)
        self.empty_tensor = torch.empty(0, dtype=torch.int32, device=device)
        # 容量记账：evictable 是"随时能扔的"，protected 是"有请求锁着、动不得的"
        self.evictable_size = 0
        self.protected_size = 0
        self.root_node = RadixTreeNode(self.key_fn)
        self.root_node.ref_count = 1  # root is always protected
        # ↑ 给 1 而不是 0：root 永远不能被淘汰，evict 里 ref_count == 0 的断言也就自动
        #   把它排除在外

    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        """锁 / 解锁一条路径（从末端节点一路到 root），只动容量记账，不动树本身。

        加锁 = 沿路每个节点 ref_count += 1；节点从 0 变 1 时（第一次被锁）把它这一段的
        长度从 evictable 挪进 protected，再加锁就只加计数、不重复挪（所以解锁时也只有
        减到 0 才挪回来）。于是 evictable_size 始终等于"当下真的能扔掉的量"。

        为什么必须锁：句柄给出的槽位会被请求拿去读写 KV，中途被 evict 掉就会读到别人的
        数据。基类文档也写了这个要求：用 match_prefix 返回的槽位之前必须先锁。
        """
        assert isinstance(handle, RadixCacheHandle)
        node = handle.node
        if unlock:
            while not node.is_root():
                node.ref_count -= 1
                assert node.ref_count >= 0
                if node.ref_count == 0:  # 最后一个持有者走了 → 这段重新变成可淘汰
                    self.evictable_size += node.length
                    self.protected_size -= node.length
                node = node.parent
        else:
            while not node.is_root():
                if node.ref_count == 0:  # 0 → 1：这段从可淘汰变成受保护
                    self.evictable_size -= node.length
                    self.protected_size += node.length
                node.ref_count += 1
                node = node.parent

    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        """查能匹配多长，返回句柄（匹配长度 + 末端节点）；一个都没命中就是 root 句柄。

        root 句柄天然是个"空结果"：cached_len = 0，get_matched_indices() 返回空张量，
        lock/unlock 走到 is_root() 就停下、什么都不会发生。
        """
        node, prefix_len = self._tree_walk(input_ids)
        # RadixCacheHandle(cached_len=prefix_len, node=node)
        return MatchResult(RadixCacheHandle(prefix_len, node))

    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> InsertResult:
        """把 [0, insert_len) 这段"token → 物理槽位"的对应关系挂进树里。

        返回的 cached_len 是"插入之前就已经在缓存里的长度"（那段槽位归缓存持有，调用方
        要释放掉自己那份）；返回句柄的 cached_len 则是插入后的缓存总长度。
        """
        # 不足一页的尾巴不缓存：那一页剩下的槽位之后可能被别人占去，留在树里会读到脏数据
        insert_len = align_down(len(input_ids), self.page_size)
        input_ids, indices = input_ids[:insert_len], indices[:insert_len]
        node, prefix_len = self._tree_walk(input_ids)
        if prefix_len != insert_len:  # NOTE: prefix_len < insert_len
            # 走到哪儿算到哪儿：剩下的这截建个新节点接上。_tree_walk 返回的 node 一定
            # "正好结束于 prefix_len"（部分命中时它已经劈过了），所以直接挂就行
            new_node = RadixTreeNode(self.key_fn)
            # clone 是必须的：indices 是请求 page_table 那一行的切片（视图），直接存下来
            # 既会被那一行后续的写入带着变，又会让整行内存一直被引用着
            new_node.set_key_value(input_ids[prefix_len:], indices[prefix_len:].clone())
            new_node.set_parent(node)
            self.evictable_size += new_node.length  # 新进树的段默认可淘汰
            node = new_node
        return InsertResult(prefix_len, RadixCacheHandle(insert_len, node))

    def evict(self, size: int) -> torch.Tensor:
        """腾出至少 size 个 token 的空间，返回被淘汰的槽位（可能多给，见基类文档）。

        从所有"可淘汰的叶子"里挑最老的删（LRU：节点的 __lt__ + 小顶堆），删掉一个之后
        如果父节点变成光杆叶子，父节点也进堆 —— 于是回收会顺着树往根爬。
        """
        if size == 0:
            return self.empty_tensor
        assert (
            size <= self.evictable_size
        ), f"Cannot evict {size}, only {self.evictable_size} is evictable"

        leave_nodes = self._collect_leave_nodes_for_evict()
        heapq.heapify(leave_nodes)
        evicted_indices: List[torch.Tensor] = []
        evicted_size = 0

        while evicted_size < size:
            assert (
                leave_nodes
            ), f"Cannot evict enough cache, need {size}, only {evicted_size} evicted"
            node = heapq.heappop(leave_nodes)
            # 能进堆的必然满足这三条：被锁的（ref_count > 0）动不得
            assert node.ref_count == 0 and node.is_leaf() and not node.is_root()
            evicted_size += node.length
            evicted_indices.append(node.value)  # 槽位交回给调用方去复用
            self.evictable_size -= node.length
            parent = node.parent
            del parent.children[self.key_fn(node._key)]
            # NOTE: root is always protected, so won't be evicted
            # 父节点被删成光杆了就也变成候选（自己还被锁着的那种不算）
            if parent.is_leaf() and parent.ref_count == 0:
                heapq.heappush(leave_nodes, parent)

        return torch.cat(evicted_indices)

    def reset(self) -> None:
        raise NotImplementedError("RadixManager.reset is not implemented")

    @property
    def size_info(self) -> SizeInfo:
        """给上层（CacheManager.available_size）看的容量信息。"""
        return SizeInfo(
            evictable_size=self.evictable_size,
            protected_size=self.protected_size,
        )

    def check_integrity(self) -> None:
        pass  # 前缀树这边不做自检；CacheManager.check_integrity 只查页数是否配平

    def _collect_leave_nodes_for_evict(self) -> List[RadixTreeNode]:
        """收集所有"可淘汰的叶子"：没有孩子、且 ref_count == 0。

        用显式栈做 DFS（不用递归：树深了会爆栈）。被锁着的叶子直接跳过，它上面的祖先也
        不会成为候选 —— 整条被锁的路径都动不得。
        """
        nodes: List[RadixTreeNode] = [self.root_node]
        leave_nodes: List[RadixTreeNode] = []

        while len(nodes) > 0:
            node = nodes.pop()
            if node.is_leaf():
                if node.ref_count == 0:
                    leave_nodes.append(node)
            else:
                for child in node.children.values():
                    nodes.append(child)

        return leave_nodes

    def _tree_walk(self, input_ids: torch.Tensor) -> Tuple[RadixTreeNode, int]:
        """从 root 往下尽量走，返回 (末端节点, 匹配到的 token 数)。

        保证末端节点"正好结束于匹配长度处"：某段只匹配了一半就当场劈开、返回前半段，
        这样 insert_prefix 才能直接往上挂新节点。

        副作用：沿途访问到的节点会刷新 timestamp（LRU 里的"最近使用"）。
        """
        prefix_len = 0
        indice_len = len(input_ids)
        node = self.root_node
        tic = time.monotonic_ns()  # 这一轮统一用一个时间戳，省得每层都取一次

        while prefix_len < indice_len:
            # 用"剩余 token 的开头一页"查孩子，和 set_parent 登记时用的是同一套算法
            child_node = node.children.get(self.key_fn(input_ids[prefix_len:]))
            if child_node is None:
                return node, prefix_len
            node = child_node  # walk to child node

            # NOTE: at least 1 page is matched, so match_len >= page_size
            # 能查到孩子就说明开头整整一页对上了（key 就是一页 token），所以下面
            # align_down 之后不会是 0，split_at 里 0 < pos 的断言才站得住
            match_len = node.get_match_len(input_ids[prefix_len:])
            match_len = align_down(match_len, self.page_size)  # 只认整页的匹配
            prefix_len += match_len

            # need to split the node if not fully matched
            if match_len != node.length:
                # 只匹配了前半段：劈开并返回前半段（后半段留着以后给人匹配）
                node = node.split_at(match_len)
                # 前半段刚被用上，刷新成"最近使用"；后半段保留旧时间戳 —— 没被用到的部分
                # 本来就该更早被淘汰
                node.timestamp = tic
                return node, prefix_len

            # update timestamp for accessed node
            node.timestamp = tic  # 整段都用上了，刷新

        return node, prefix_len


def _get_key_fn(page_size: int) -> KEY_FN:
    """key 函数：拿开头一页 token 当 children 的 key。

    只要保证"开头一页相同的段一定同 key"就够了，不必拿整段去比（一段可能上千 token，
    当字典 key 太贵）。page_size == 1 时退化成单个 int，比元组更省。
    """
    if page_size == 1:
        return lambda x: x[0].item()
    return lambda x: tuple(x[:page_size].tolist())
