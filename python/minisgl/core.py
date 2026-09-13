from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Literal

import torch

if TYPE_CHECKING:
    from minisgl.attention import BaseAttnBackend, BaseAttnMetadata
    from minisgl.kvcache import BaseCacheHandle, BaseKVCachePool
    from minisgl.moe import BaseMoeBackend


@dataclass
class SamplingParams:
    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    ignore_eos: bool = False
    max_tokens: int = 1024

    @property
    def is_greedy(self) -> bool:
        return (self.temperature <= 0.0 or self.top_k == 1) and self.top_p == 1.0


@dataclass(eq=False)
class Req:
    """后端（调度器侧）的请求对象：由 PendingReq 构造，随一轮接一轮的 prefill / decode 推进，
    直到请求结束或被 abort。

    序列长度由三个数字描述，恒满足 cached_len <= device_len <= max_device_len：

        [0, cached_len)               前缀：KV 已算好并留在 cache 中，attention 直接读
        [cached_len, device_len)      本次 forward 要计算的部分，长度即 extend_len
        [device_len, max_device_len)  本次之后还没上 device 的部分

    每做完一次 forward，complete_one() 会把 cached_len 追平到 device_len，再把 device_len
    +1（腾出一个位置放刚采样出的 token），如此循环往复。

    eq=False 表示按对象身份比较和哈希，因此 Req 可以直接放进 Set
    （例如 DecodeManager.running_reqs）；prefill 被切块时构造的是子类 ChunkedReq
    （见 scheduler/prefill.py），它不参与采样。
    """

    # input_ids 是 host 侧的副本：prompt + 已采出的 token，由 append_host() 逐个追加
    input_ids: torch.Tensor  # cpu tensor
    table_idx: int  # page_table / token_pool 的行号，由 TableManager 分配与回收
    cached_len: int  # 已算好 KV 的前缀长度，本次 forward 只需要算它之后的部分
    output_len: int  # 最多还能生成多少 token（取自 sampling_params.max_tokens）
    uid: int  # 全局唯一的请求 id，用于结果回传和 abort 匹配
    sampling_params: SamplingParams  # 该请求的采样配置（temperature、top_k、top_p 等）
    cache_handle: BaseCacheHandle  # 前缀缓存句柄：锁住已命中的前缀，记录插入缓存前已有的长度

    def __post_init__(self) -> None:
        assert self.input_ids.is_cpu
        # device_len 不是构造参数：它表示"已经落在 device 上"的长度，随运行过程增长
        self.device_len = len(self.input_ids)  # 起始时输入的 token 都还没算，全部要算一遍
        self.max_device_len = len(self.input_ids) + self.output_len  # 序列长度上限
        # cached_len < device_len：至少要留 1 个 token 给这次 forward，否则这一轮无事可做
        assert 0 <= self.cached_len < self.device_len <= self.max_device_len

    @property
    def remain_len(self) -> int:
        """距离长度上限还剩多少 token 可以生成。"""
        return self.max_device_len - self.device_len

    @property
    def extend_len(self) -> int:
        """本次 forward 真正要计算的 token 数，即 [cached_len, device_len) 的长度。"""
        return self.device_len - self.cached_len

    def complete_one(self) -> None:
        """每次 forward 结束后调用（见 Engine.forward_batch）。

        此时 [0, device_len) 的 KV 都已写进 cache，于是让 cached_len 追平 device_len；
        同时刚采样出的 token 已经占住 device 上的下一个位置，device_len 再 +1。
        """
        self.cached_len = self.device_len
        self.device_len += 1

    def append_host(self, next_token: torch.Tensor) -> None:
        """把新采样出的 token 追加到 host 侧的 input_ids（在采样结果拷回 CPU 之后调用）。"""
        self.input_ids = torch.cat([self.input_ids, next_token])

    @property
    def can_decode(self) -> bool:
        """还有生成余量、可以继续 decode；False 表示已达长度上限。

        调度器据此决定是否把请求留在 decode 队列，以及采样结果是否写回
        （见 scheduler._make_write_tuple 中的 -1 哨兵）。注意它只看长度，不看 EOS。
        """
        return self.remain_len > 0

    def __repr__(self) -> str:  # 只打印长度记账信息，便于看日志时快速定位请求
        return (
            f"{type(self)}(table_idx={self.table_idx}, "
            f"cached_len={self.cached_len}, device_len={self.device_len}, "
            f"max_device_len={self.max_device_len})"
        )


@dataclass
class Batch:
    """一次 forward 要处理的请求集合：由调度器（prefill / decode manager）构造，由 engine 消费。

    这里只有 reqs 和 phase 是构造时就确定的，其余字段都是同一条流水线上不同角色依次填好的临时数据：

        reqs / phase      调度器构造时给出
        padded_reqs       GraphRunner.pad_batch()：CUDA graph 要求固定 batch size，用 dummy_req 补齐
        positions         Scheduler._prepare_batch()：每个 token 的绝对位置，供 RoPE 使用
        out_loc           Scheduler._prepare_batch()：每个 token 的 K/V 写进 cache 的哪个物理槽位
        input_ids         Scheduler._forward()：每个 token 的 id，直接喂给 embedding
        attn_metadata     attention backend 的 prepare_metadata() / prepare_for_replay()

    input_ids / positions / out_loc 三者形状相同：把所有 padded_reqs 的
    [cached_len, device_len) 区间按顺序拼成一条扁平数组，长度 = 各 req 的 extend_len 之和。
    """

    reqs: List[Req]  # 本批次真正要处理的请求；decode 时按 uid 排序，保证各 TP rank 的请求顺序一致
    phase: Literal["prefill", "decode"]  # 本批次属于哪个阶段，决定 q/k 的长度取法
    # these fields should be set by scheduler
    input_ids: torch.Tensor = field(init=False)  # 本批次所有待计算 token 的 id
    # positions 是每个 token 的绝对位置；prefill 命中前缀缓存时从 cached_len 起算
    positions: torch.Tensor = field(init=False)
    out_loc: torch.Tensor = field(init=False)  # 与之对应的 cache 物理槽位，store_kv 按它写入 K/V
    # reqs + 若干 dummy_req：CUDA graph 要求固定的 batch size，不足的用 dummy 补齐
    padded_reqs: List[Req] = field(init=False)
    # this field should be set by attention backend
    attn_metadata: BaseAttnMetadata = field(init=False)  # 后端自定义的元数据（cu_seqlens 等）

    @property
    def is_prefill(self) -> bool:
        return self.phase == "prefill"

    @property
    def is_decode(self) -> bool:
        return self.phase == "decode"

    @property
    def size(self) -> int:
        """真实请求数；logits 只取前 size 行（补位的 dummy_req 结果要丢掉）。"""
        return len(self.reqs)

    @property
    def padded_size(self) -> int:
        """补齐后的 batch 大小；CUDA graph 按它选图，也是 capture buffer 的切片长度。"""
        return len(self.padded_reqs)


@dataclass
class Context:
    """每个进程（也就是每个 TP rank）一份的运行时上下文，由 Engine 构造后注册为全局单例。

    为什么要做成全局的：模型各层（attention / moe / lm_head）的 forward 只接收张量参数，
    不接收 batch，但每层又都要用到"当前这一批请求"和 KV cache。与其把 batch 一路透传
    穿过整个 module 树，不如挂在全局上下文里，层内用 get_global_ctx() 取用
    （见 layers/attention.py、layers/moe.py）。CUDA graph 抓取时这一点尤其关键：
    GraphRunner 用 forward_batch() 把 dummy batch 挂上去，再调用同样签名的
    model.forward() 即可完成抓图。

    字段是分步填的：Engine.__init__ 里先 Context(page_size)，再依次补上 kv_cache、page_table、
    attn_backend、moe_backend。
    """

    page_size: int  # KV cache 的页大小；大于 1 时相邻的若干 token 共用一个物理页
    # NOTE: this table always treat page_size = 1
    # [max_running_req + 1, aligned_max_seq_len] 的 int32：行 = 请求的 table_idx
    # （最后一行留给 dummy_req），列 = 请求内的 token 位置，值 = 该 token 的 K/V 在
    # kv_cache 里的物理槽位。始终按 page_size = 1 逐 token 记录，页的折叠由 attention
    # 后端按 page_size 步长切片完成。
    page_table: torch.Tensor = field(init=False)
    attn_backend: BaseAttnBackend = field(init=False)  # attention 后端：拼元数据、选 kernel 等
    moe_backend: BaseMoeBackend = field(init=False)  # MoE 后端；非 MoE 模型不会设置这个字段
    kv_cache: BaseKVCachePool = field(init=False)  # K/V 的物理存储池，按 page_table 里的槽位寻址
    _batch: Batch | None = field(default=None, init=False)  # 只在 forward_batch() 中非空

    @property
    def batch(self) -> Batch:
        """当前正在 forward 的 batch，只能在 forward_batch() 的作用域内调用。"""
        assert self._batch is not None, "No active batch in context"
        return self._batch

    @contextmanager
    def forward_batch(self, batch: Batch):
        """把 batch 挂到全局上下文上，这样模型各层才能通过 ctx.batch 读到它。

        不允许嵌套（一次 forward 只能有一个当前 batch）。退出时无论正常返回还是抛异常都会清空，
        以免下一次 forward 读到已经过期的 batch。
        """
        assert self._batch is None, "Nested forward_batch is not allowed"
        try:
            self._batch = batch
            yield
        finally:
            self._batch = None


_GLOBAL_CTX: Context | None = None


def set_global_ctx(ctx: Context):
    """注册全局上下文；断言只能设置一次，因此一个进程里只能有一个 Engine。

    测试里请用 fixture 提前把 _GLOBAL_CTX 置空，结束后再还原
    （见 tests/core/test_cache_allocate.py）。
    """
    global _GLOBAL_CTX
    assert _GLOBAL_CTX is None, "Global context is already set"
    _GLOBAL_CTX = ctx


def get_global_ctx() -> Context:
    """取全局上下文；模型各层在 forward 期间靠它拿到当前 batch 和 KV cache。"""
    assert _GLOBAL_CTX is not None, "Global context is not set"
    return _GLOBAL_CTX
