"""FlashInfer 后端（fi）：prefill / decode 都交给 flashinfer。

和 fa.py 最本质的区别是接口形状 —— flashinfer 是**两段式**的：

    plan()  在 **host 侧**读一堆 CPU 张量（indptr / indices / last_page_len /
            seq_lens），把 kernel 的调度参数算好，再异步 H2D 拷进 device 缓冲
    run()   只负责启动 kernel，参数都取自 plan 留下的那份缓冲

所以这个文件里到处是 `*_cpu` / pinned 张量（plan 要读），而 fa.py 那边全是 device 张量。

第二件事：FI 眼里**页大小恒为 1**。全局 page_table 本来就是按 token 记槽位的
（page_table[row, i] = 第 i 个 token 的物理槽位，见 scheduler/table.py），
所以这里既不用按页取样也不用除 page_size，直接把
`page_table[req.table_idx, : device_len]` 逐请求拼成一条 ragged 槽位表，
再把 KV 池 `view(-1, 1, heads, dim)` 拍平即可 —— 与配置的 page_size 无关。

CUDA graph 路径：decode 用 CUDAGraphBatchDecodeWithPagedKVCacheWrapper，它把
indptr / indices / last_page_len 绑在 capture 那批**地址固定**的缓冲上；plan 的结果
写在那里，图里读的也是那里（见 prepare_for_capture / prepare_for_replay）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Dict, List, Literal

import torch
from minisgl.core import Batch, get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.env import ENV
from minisgl.utils import div_even, init_logger

from .base import BaseAttnBackend, BaseAttnMetadata
from .utils import BaseCaptureData

# 只在类型注解里用到（运行时再局部 import）：不装 flashinfer 时，
# import minisgl.attention 依然要能跑（只用 fa / trtllm 的话根本不需要它）
if TYPE_CHECKING:
    from flashinfer import (
        BatchDecodeWithPagedKVCacheWrapper,
        BatchPrefillWithPagedKVCacheWrapper,
        CUDAGraphBatchDecodeWithPagedKVCacheWrapper,
    )
    from minisgl.models import ModelConfig


def _next_power_of_2(n: int) -> int:
    """向上取到 2 的幂（_get_ones_cpu 用它决定缓冲扩容到多大）。"""
    if n <= 1:
        return 1
    return 1 << math.ceil(math.log2(n))


logger = init_logger(__name__)


@dataclass
class FICaptureData(BaseCaptureData):
    """抓图用的固定缓冲。

    基类已经按 (max_bs, max_seq_len) 开好了 seq_lens / cu_seqlens_k / page_table 等；
    这里做的只是**改名对齐 flashinfer 的形参**：

        one_tensor（就是 seq_lens）   当 last_page_len 用（page_size=1 时恒为 1）
        indices  （就是 page_table）  当 ragged 的槽位表用（注意会被拍成一维）
    """

    @property
    def one_tensor(self) -> torch.Tensor:
        return self.seq_lens

    @property
    def indices(self) -> torch.Tensor:
        return self.page_table


@dataclass
class FIMetadata(BaseAttnMetadata):
    """一批 forward 用的全部 flashinfer 元数据。

    字段名沿用 flashinfer 的 plan() 形参，方便对照。`_cpu` / `_gpu` 后缀就是
    "plan 读 host、kernel 读 device"这件事的直接体现：除了给 LM head 用的
    cu_seqlens_q_gpu，其余都在 host 上（而且是 pinned，见 prepare_metadata）。

    page_size 固定为 1：一个 indices 条目 = 一个 token 的槽位（见模块注释）。
    """

    # fmt: off
    cu_seqlens_q_cpu:   torch.Tensor  # on cpu
    cu_seqlens_k_cpu:   torch.Tensor  # on cpu
    cu_seqlens_q_gpu:   torch.Tensor  # on gpu
    indices:            torch.Tensor  # on gpu
    last_page_len_cpu:  torch.Tensor  # on cpu
    num_qo_heads:       int
    num_kv_heads:       int
    head_dim:           int
    page_size:          Literal[1] # currently only support page_size=1
    pos_encoding_mode:  str
    seq_lens_cpu:       torch.Tensor  # on cpu
    dtype:              torch.dtype
    wrapper:            BatchPrefillWithPagedKVCacheWrapper | BatchDecodeWithPagedKVCacheWrapper
    initialized:        bool = False
    # fmt: on

    def __post_init__(self) -> None:
        # 一个 assert 干两件事：把"FI 只支持 page_size=1"钉死，并检查每个张量
        # 都真的在它该在的设备上（plan 读 host、kernel 读 device，放错了
        # flashinfer 会用很绕的方式报错，不如在这里就拦住）
        assert self.page_size == 1, "Currently only page_size=1 is supported."
        assert (
            self.cu_seqlens_k_cpu.is_cpu
            and self.cu_seqlens_q_cpu.is_cpu
            and self.cu_seqlens_q_gpu.is_cuda
            and self.indices.is_cuda
            and self.last_page_len_cpu.is_cpu
            and self.seq_lens_cpu.is_cpu
        )

    def get_last_indices(self, bs: int) -> torch.Tensor:
        # 每个请求最后一个 token 在扁平 q 里的下标 = q 侧前缀和减一
        # （prefill 时 LM head 只算这些位置，调用方见 layers/embedding.py）
        return self.cu_seqlens_q_gpu[1 : 1 + bs] - 1


class FlashInferBackend(BaseAttnBackend):
    def __init__(self, config: ModelConfig) -> None:
        # 延迟导入：flashinfer 只在真正用这个后端时才需要装
        from flashinfer import (
            BatchDecodeWithPagedKVCacheWrapper,
            BatchPrefillWithPagedKVCacheWrapper,
        )

        self.config = config
        self.kvcache = get_global_ctx().kv_cache
        self.device = self.kvcache.device
        # flashinfer 要求的通用 float workspace（128 MiB）：plan 的中间结果和
        # kernel 的暂存都在里面，两个 wrapper 共用这一块（必须常驻，不能释放）
        self.float_workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=self.device
        )
        # prefill 与 decode 各一个 wrapper：走 flashinfer 自己的 fa2 实现
        # （代码里注明它家的 fa3 反而更慢）。注意上图的 decode **不用**下面这个
        # wrapper，而是每个档位另建 graph wrapper（见 graph_wrappers）
        self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.float_workspace_buffer,
            kv_layout="NHD",  # NHD = token 在头维之前，和 KV 池 [..., heads, dim] 一致
            backend="fa2",  # flashinfer fa3 is slow, use fa2 instead
        )
        self.decode_wrappers = BatchDecodeWithPagedKVCacheWrapper(
            self.float_workspace_buffer,
            use_tensor_cores=self.use_tensor_cores,
            kv_layout="NHD",
            backend="fa2",  # flashinfer fa3 is slow, use fa2 instead
        )

        # NOTE: some hack to reuse the int_workspace_buffer
        # 小 hack：int workspace 由 prefill wrapper 建好，这里直接把它共享给 decode
        # wrapper，省掉再开一块（两个 wrapper 不会同时跑）。存一份到 self 上，
        # 是为了后面每个 graph wrapper 也能共用它
        self.int_workspace_buffer = self.prefill_wrapper._int_workspace_buffer
        self.decode_wrappers._int_workspace_buffer = self.int_workspace_buffer

        # initialize some data members
        # 传给 plan 的是**本 rank 的**头数（TP 之后），切法与 qkv_proj / KV 池一致
        tp_size = get_tp_info().size
        self.qo_head_local = div_even(self.config.num_qo_heads, tp_size)
        self.kv_head_local = div_even(self.config.num_kv_heads, tp_size, allow_replicate=True)

        # last_page_len 恒为 1（page_size=1），所以缓存一份"全 1"的 pinned 张量反复用；
        # 起手是空张量，第一次用到时再按 2 的幂扩容（见 _get_ones_cpu）
        self.cached_ones_cpu: torch.Tensor = torch.tensor([], dtype=torch.int32, pin_memory=True)
        # for cuda graph
        # 抓图状态：capture_bs 是档位表，graph_wrappers 每档一个 graph wrapper，
        # capture 是所有档位共用的固定缓冲
        self.capture_bs: List[int] = []
        self.max_graph_bs = 0
        self.graph_wrappers: Dict[int, CUDAGraphBatchDecodeWithPagedKVCacheWrapper] = {}
        self.capture: FICaptureData | None = None
        # 用来串行化 plan 的异步 H2D：机制见 _initialize_metadata_once。
        # 先记一次，这样第一次 plan 之前那次 synchronize 也能正常对上
        self.last_event = torch.cuda.Event()
        self.last_event.record()

    def _initialize_metadata_once(self, metadata: FIMetadata) -> None:
        """同一份 metadata 只 plan 一次（靠 initialized 标记）。

        为什么只做一次：plan 是 host 侧的重活（按这批的长度分布算 kernel 调度参数），
        而这批数据在 metadata 的生命周期里不会再变 —— 重算纯属浪费。
        """
        if metadata.initialized:
            return

        # 局部导入：decode 的 graph wrapper 是它的子类，所以下面那次 isinstance
        # 能把"普通 decode wrapper"和"graph wrapper"两个分支一起覆盖到
        from flashinfer import BatchDecodeWithPagedKVCacheWrapper

        # 先置位再 plan：万一 plan 抛异常，也不会留下做了一半的状态被再次进入
        metadata.initialized = True
        # FlashInfer planning reuses a pinned host staging buffer and launches an
        # async H2D copy. Wait here before the next plan mutates that host buffer.
        # ↑ flashinfer 的 plan 是异步发往 device 的，而它内部复用同一块 pinned 中转
        #   内存：下一次 plan 会覆写那块 host 内存，所以先等上一次的拷贝做完
        #   （last_event 在本函数末尾 record）
        self.last_event.synchronize()
        if isinstance(metadata.wrapper, BatchDecodeWithPagedKVCacheWrapper):
            # ---- decode：ragged 布局 ----
            # indptr    = cu_seqlens_k（各请求 KV 长度的前缀和），告诉 kernel
            #             indices 里哪一段属于哪个请求
            # indices   = 每个 token 的物理槽位（page_size=1）
            # last_page_len = 全 1（每页只装 1 个 token）
            metadata.wrapper.plan(
                indptr=metadata.cu_seqlens_k_cpu,
                indices=metadata.indices,
                last_page_len=metadata.last_page_len_cpu,
                num_qo_heads=metadata.num_qo_heads,
                num_kv_heads=metadata.num_kv_heads,
                head_dim=metadata.head_dim,
                page_size=metadata.page_size,
                pos_encoding_mode=metadata.pos_encoding_mode,
                seq_lens=metadata.seq_lens_cpu,
                data_type=metadata.dtype,
                q_data_type=metadata.dtype,
                kv_data_type=metadata.dtype,
                non_blocking=True,
            )
        else:
            # ---- prefill：多一项 q 侧的 qo_indptr（每个请求这次算几个 token），
            #      并且必须 causal=True（因果掩码）----
            # 注意这里传的是 **paged_kv_** 前缀的那套形参名
            metadata.wrapper.plan(
                qo_indptr=metadata.cu_seqlens_q_cpu,
                paged_kv_indptr=metadata.cu_seqlens_k_cpu,
                paged_kv_indices=metadata.indices,
                paged_kv_last_page_len=metadata.last_page_len_cpu,
                num_qo_heads=metadata.num_qo_heads,
                num_kv_heads=metadata.num_kv_heads,
                head_dim_qk=metadata.head_dim,
                page_size=metadata.page_size,
                pos_encoding_mode=metadata.pos_encoding_mode,
                seq_lens=metadata.seq_lens_cpu,
                q_data_type=metadata.dtype,
                kv_data_type=metadata.dtype,
                non_blocking=True,
                causal=True,
            )
        # 记下"这次 plan 的 H2D 已经发出"，供下一次 plan 前等待
        self.last_event.record()

    def _get_ones_cpu(self, bs: int) -> torch.Tensor:
        """取长度为 bs、全 1 的 pinned int32 CPU 张量（当 last_page_len 用）。

        按 2 的幂扩容：decode 每一步都要用它，每步新开一块太贵（分配 + pin 都不便宜），
        而要的长度又随真实请求数浮动，所以留一块足够大的，用切片返回。切片是视图、
        不拷数据，返回的仍是 pinned 内存 —— plan 的异步 H2D 才有效。
        """
        if bs <= len(self.cached_ones_cpu):
            return self.cached_ones_cpu[:bs]
        # padding to next pow of 2
        next_len = _next_power_of_2(bs)
        self.cached_ones_cpu = torch.ones(next_len, dtype=torch.int32, pin_memory=True)
        return self.cached_ones_cpu[:bs]

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        # page_size=1 的视角：把 [num_pages, page_size, heads, dim] 拍成
        # [num_pages*page_size, 1, heads, dim]，下标正好是 page_table 里记的槽位号
        def _flatten_cache(cache: torch.Tensor) -> torch.Tensor:  # treat page = 1
            return cache.view(-1, 1, cache.shape[2], cache.shape[3])

        metadata = batch.attn_metadata
        assert isinstance(metadata, FIMetadata)
        # plan 只做一次：第一次碰到这份 metadata 时在这里完成，之后是空操作
        self._initialize_metadata_once(metadata)
        # 与 fa.py 一样，先落库再算：k/v 按 out_loc 写进 KV 池，attention 读的是池子，
        # 这里的 k/v 入参只用于写入（所以它们在 AttentionLayer 里保持 2D）
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
        kv_cache = (self.kvcache.k_cache(layer_id), self.kvcache.v_cache(layer_id))
        kv_cache = (_flatten_cache(kv_cache[0]), _flatten_cache(kv_cache[1]))
        # run 只负责启动 kernel：indptr / indices / 头数 / page_size 都在 plan 里定了
        return metadata.wrapper.run(q=q, paged_kv_cache=kv_cache)

    def prepare_metadata(self, batch: Batch) -> None:
        # padded_reqs 里包含为了凑 CUDA graph 档位补出来的 dummy req —— 它们也要
        # 出现在 cu_seqlens 里（长度 0 或 1），否则行数对不上图
        reqs = batch.padded_reqs

        padded_size = len(reqs)
        # 三个长度的含义见 core.Req 的注释：
        #   extend_len 本次要算几个 token（= device_len - cached_len）
        #   device_len 这次算完后 device 上的总长度（KV cache 覆盖的长度）
        #   cached_len 前缀缓存命中的长度
        seqlens_q = [req.extend_len for req in reqs]
        seqlens_k = [req.device_len for req in reqs]
        cached_lens = [req.cached_len for req in reqs]
        max_seqlen_q = max(seqlens_q)
        # 全部开在 pinned CPU 上：plan 在 host 侧读它们，pinned 才能让随后的 H2D 异步
        CPU_KWARGS = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}

        device = self.device
        seq_len_cpu = torch.tensor(seqlens_k, **CPU_KWARGS)
        # cu_seqlens = 前缀和（前面补个 0）：第 i 个请求对应扁平数组的 [起, 止)
        cu_seqlens_k_cpu = torch.tensor([0] + seqlens_k, **CPU_KWARGS).cumsum_(dim=0)
        if max_seqlen_q == 1:  # decode with all extend_len = 1
            # decode：每个请求恰好出 1 个 token，前缀和就是 0,1,2,...
            cu_seqlens_q_cpu = torch.arange(0, padded_size + 1, **CPU_KWARGS)
        elif all(l == 0 for l in cached_lens):  # prefill with no cache hit
            # 完全没命中缓存 → q 长度 = kv 长度，直接复用上面那份（同一张量）
            cu_seqlens_q_cpu = cu_seqlens_k_cpu
        else:  # normal extend prefill, with partial cache hit
            # 前缀部分命中：q 只覆盖 [cached_len, device_len)，各请求长度不等
            cu_seqlens_q_cpu = torch.tensor([0] + seqlens_q, **CPU_KWARGS).cumsum_(dim=0)

        page_table = get_global_ctx().page_table
        batch.attn_metadata = FIMetadata(
            cu_seqlens_q_cpu=cu_seqlens_q_cpu,
            cu_seqlens_k_cpu=cu_seqlens_k_cpu,
            # 唯一要落到 device 的一份（get_last_indices 要给 LM head 用）；
            # 其余 CPU 张量都是给 plan 读的
            cu_seqlens_q_gpu=cu_seqlens_q_cpu.to(device, non_blocking=True),
            # 逐 token 取槽位、按请求顺序拼成一条 ragged 表（靠 indptr 切段）。
            # 这里**不**按页取样也不除 page_size —— 那是 fa.py 的算法
            indices=torch.cat([page_table[req.table_idx, : req.device_len] for req in reqs]),
            # 全 1：page_size=1 时每页正好 1 个 token，页永远是满的
            last_page_len_cpu=self._get_ones_cpu(padded_size),
            num_qo_heads=self.qo_head_local,
            num_kv_heads=self.kv_head_local,
            head_dim=self.config.head_dim,
            page_size=1,
            # RoPE 已由 AttentionLayer 原地做过（见 layers/attention.py），
            # 这里必须置 NONE，否则会被旋两遍
            pos_encoding_mode="NONE",
            seq_lens_cpu=seq_len_cpu,
            dtype=self.kvcache.dtype,
            # 上不上图决定用哪个 wrapper：prepare_for_capture / prepare_for_replay
            # 之后会被换成对应档位的 graph wrapper
            wrapper=self.decode_wrappers if batch.is_decode else self.prefill_wrapper,
        )

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        assert self.capture is None, "Capture already initialized."
        max_bs = max(bs_list)
        # 与 fa.py 的差别：那边按页开缓冲（max_seq_len // page_size），这边是
        # "每个 token 一个索引"，所以直接用 max_seq_len
        capture = FICaptureData.create(max_bs, max_seq_len, self.kvcache.device)
        # 基类开的是按行的 [max_bs, max_seq_len] 页表；FI 要的是一整条 ragged 的
        # 一维表（用 indptr 切段），所以这里拍成一维，整块当作 indices 缓冲
        capture.page_table = capture.page_table.view(-1)  # use 1D as ragged indices
        self.max_graph_bs = max_bs
        self.capture = capture
        self.capture_bs = sorted(bs_list)

    @cached_property
    def use_tensor_cores(self) -> bool:
        # 环境变量优先（对齐/调试用），否则按 GQA 比例自动决定：
        # kv 头被越多 q 头共享，decode 越值得走 tensor core 版 kernel。
        # cached_property：只求值一次 —— __init__ 建 decode wrapper 时就已经定型
        if (overriden_value := ENV.FLASHINFER_USE_TENSOR_CORES.value) is not None:
            logger.warning(f"Overriding FlashInfer tensor core usage to {overriden_value}")
            return overriden_value
        GQA = self.config.num_qo_heads // self.config.num_kv_heads
        return GQA >= 4

    def prepare_for_capture(self, batch: Batch) -> None:
        # GraphRunner 在**开始录图之前**调用（见 engine/graph.py）：给这一档 bs 建一个
        # 绑好固定缓冲的 graph wrapper，并用 dummy batch 先 plan 一遍 —— 于是录进图里的
        # 读地址就是下面这几个 capture 缓冲
        from flashinfer import CUDAGraphBatchDecodeWithPagedKVCacheWrapper

        bs = batch.size
        assert bs in self.capture_bs and bs not in self.graph_wrappers and self.capture
        capture = self.capture
        self.graph_wrappers[bs] = CUDAGraphBatchDecodeWithPagedKVCacheWrapper(
            self.float_workspace_buffer,
            kv_layout="NHD",
            use_tensor_cores=self.use_tensor_cores,
            # 三个缓冲都切自 capture（地址固定，才能被录进图）：
            #   indptr_buffer        cu_seqlens_k 的前 bs+1 项
            #   indices_buffer       整块一维槽位表（用多少由 indptr 决定）
            #   last_page_len_buffer 全 1 的前 bs 项
            indptr_buffer=capture.cu_seqlens_k[: bs + 1],
            indices_buffer=capture.indices,
            last_page_len_buffer=capture.one_tensor[:bs],
        )
        # 这两个只能构造完再塞进去（graph wrapper 的构造参数里没有它们），
        # 与 __init__ 里那个 decode wrapper 共用同一块 int workspace
        self.graph_wrappers[bs]._backend = "fa2"
        self.graph_wrappers[bs]._int_workspace_buffer = self.int_workspace_buffer
        # 先给这批 dummy 数据生成元数据，再把 wrapper 换成刚建的 graph wrapper，
        # plan 的结果就落进 capture 的固定缓冲了
        self.prepare_metadata(batch)
        metadata = batch.attn_metadata
        assert isinstance(metadata, FIMetadata)
        metadata.wrapper = self.graph_wrappers[bs]
        self._initialize_metadata_once(metadata)

    def prepare_for_replay(self, batch: Batch) -> None:
        # GraphRunner.replay 在 g.replay() **之前**调用（见 engine/graph.py）：
        # 这一步的长度信息必须先写进那批固定缓冲，因为图里读的就是那些地址
        metadata, bs = batch.attn_metadata, batch.padded_size
        # not initialized：说明这份 metadata 是本步 prepare_metadata 刚造出来、
        # 还没 plan 过（plan 正是下面这句要做的事）
        assert isinstance(metadata, FIMetadata) and not metadata.initialized
        assert self.capture is not None and bs in self.capture_bs
        # 用 padded_size（不是 batch.size）取档位：图是按补位后的档位抓的
        metadata.wrapper = self.graph_wrappers[bs]
        self._initialize_metadata_once(metadata)
