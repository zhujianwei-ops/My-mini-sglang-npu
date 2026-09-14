"""GraphRunner：用 CUDA graph 把 decode 的一整段 GPU 工作"录"下来，之后每步只 replay。

一个 decode step 的 GPU 工作量很小（每个请求一个 token），但 Python 侧要拼元数据、
发几十上百次 kernel launch，CPU 开销反而占大头。CUDA graph 把这段录成一张图，
重放时只发一次 launch，把 CPU 侧开销压到接近零。

代价是"地址必须固定"：录进图里的 kernel 记的是张量的虚拟地址，所以
  - 输入（input_ids / positions / out_loc）要放在固定的 GraphCaptureBuffer 里，
    replay 前把真数据 copy 进去、跑完从固定 logits 缓冲里切结果；
  - attention 后端要用的 page_table / cu_seqlens 等元数据也得放固定缓冲，
    这就是 BaseAttnBackend 里 init_capture_graph / prepare_for_capture / prepare_for_replay
    三件套的用途。

因此只有形状规整的 **decode** 批能上图（prefill 的 seqlen 太活），而且每种 batch size
各抓一张图：真请求不足时用 dummy_req 补到最近的档位（pad_batch），让形状落在某一档上。
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List

import torch
from minisgl.core import Batch, Req, get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.utils import init_logger
from tqdm import tqdm

if TYPE_CHECKING:
    from minisgl.attention import BaseAttnBackend
    from minisgl.models import BaseLLMModel

logger = init_logger(__name__)


@dataclass
class GraphCaptureBuffer:
    """上图批次的固定输入/输出缓冲，按**最大档位**分配，每档只用前 padded_size 行。

    replay 的完整链路：
        copy_from(batch)  把本批真实的 input_ids / positions / out_loc 写进来
        graph.replay()    图里的 kernel 就是从这四个地址读写的
        切 logits[:size]  结果也在这里
    """

    input_ids: torch.Tensor
    out_loc: torch.Tensor
    positions: torch.Tensor
    logits: torch.Tensor

    @classmethod
    def init(cls, bs: int, vocab_size: int, device: torch.device) -> GraphCaptureBuffer:
        # 三个索引张量都是 int32（和 token_pool / page_table 一致）；
        # logits 是 float32 且形状 (bs, vocab) —— 这是抓图占显存的大头，
        # 也解释了 _determine_cuda_graph_bs 为什么要按空闲显存挑档位上限
        return GraphCaptureBuffer(
            input_ids=torch.zeros(bs, dtype=torch.int32, device=device),
            out_loc=torch.zeros(bs, dtype=torch.int32, device=device),
            positions=torch.zeros(bs, dtype=torch.int32, device=device),
            logits=torch.empty(bs, vocab_size, dtype=torch.float32, device=device),
        )

    def set_batch(self, batch: Batch) -> None:
        """把 batch 的输入张量**换成这块缓冲的视图**（抓图时用，让录进去的地址是缓冲）。"""
        _slice = slice(batch.padded_size)
        batch.input_ids = self.input_ids[_slice]
        batch.out_loc = self.out_loc[_slice]
        batch.positions = self.positions[_slice]

    def copy_from(self, batch: Batch) -> None:
        """把本批真实数据**写进缓冲**（重放前用，与 set_batch 正好相反）。"""
        _slice = slice(batch.padded_size)
        self.input_ids[_slice] = batch.input_ids
        self.out_loc[_slice] = batch.out_loc
        self.positions[_slice] = batch.positions


def _determine_cuda_graph_bs(
    cuda_graph_bs: List[int] | None,
    cuda_graph_max_bs: int | None,
    free_memory: int,
) -> List[int]:
    """定下要抓哪些 batch size 的图。

    用户显式给了列表就用它；否则按显存定上限（这里传进来的是**建模型之前**的空闲显存，
    所以 >80 GiB 实际就是在认 H200 这类 141G 的卡，其余按 H100 80G 处理），
    档位取 1, 2, 4 然后 8 起步每 8 一档 —— 小批稀疏、大批均匀，兼顾覆盖与显存。
    `cuda_graph_max_bs < 1` 表示禁用（返回空列表）。
    """
    if cuda_graph_bs is not None:
        return cuda_graph_bs

    free_memory_gb = free_memory / (1 << 30)
    if cuda_graph_max_bs is None:
        if free_memory_gb > 80:  # H200
            cuda_graph_max_bs = 256
        else:
            cuda_graph_max_bs = 160

    if cuda_graph_max_bs < 1:
        return []

    return [1, 2, 4] + list(range(8, cuda_graph_max_bs + 1, 8))


def mem_GB(size: int) -> str:
    """把字节数印成 GiB 字符串（日志用）。"""
    return f"{size / (1024**3):.2f} GiB"


def get_free_memory(device: torch.device) -> int:
    """驱动视角的空闲显存（不是 PyTorch 分配器视角的），mem_get_info 返回 (free, total)。"""
    return torch.cuda.mem_get_info(device)[0]


class GraphRunner:
    def __init__(
        self,
        stream: torch.cuda.Stream,
        device: torch.device,
        model: BaseLLMModel,
        attn_backend: BaseAttnBackend,
        cuda_graph_bs: List[int] | None,
        cuda_graph_max_bs: int | None,
        free_memory: int,
        max_seq_len: int,
        vocab_size: int,
        dummy_req: Req,
    ) -> None:
        # 先算出档位表，再由档位表决定最大 batch size
        cuda_graph_bs = _determine_cuda_graph_bs(
            cuda_graph_bs=cuda_graph_bs,
            cuda_graph_max_bs=cuda_graph_max_bs,
            free_memory=free_memory,
        )
        self.attn_backend = attn_backend
        # max_graph_bs = 0 表示禁用（见 _capture_graphs 开头）
        self.max_graph_bs = max(cuda_graph_bs) if cuda_graph_bs else 0
        # 升序：pad_batch 用 next(...) 找"第一个装得下的档位"，所以必须有序
        self.graph_bs_list = sorted(cuda_graph_bs)
        self.dummy_req = dummy_req
        self.stream = stream
        self.device = device
        self._capture_graphs(max_seq_len, vocab_size, model)

    def _capture_graphs(self, max_seq_len: int, vocab_size: int, model: BaseLLMModel):
        self.graph_map: Dict[int, torch.cuda.CUDAGraph] = {}
        if self.max_graph_bs == 0:
            return logger.info_rank0("CUDA graph is disabled.")

        # 让后端按上限预分配元数据缓冲（一次分配，所有档位共用），
        # 之后抓图/重放都往这块固定地址里填数据
        self.attn_backend.init_capture_graph(max_seq_len=max_seq_len, bs_list=self.graph_bs_list)

        # 抓图前清干净：把分配器缓存还给驱动，后面 pbar 里显示的 avail_mem 才是真实可用量
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

        logger.info_rank0(f"Start capturing CUDA graphs with sizes: {self.graph_bs_list}")
        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory before capturing CUDA graphs: {mem_GB(free_memory)}")

        # 输入/输出缓冲按最大档位分配一次，各档位共用（用前 bs 行）
        self.buffer = GraphCaptureBuffer.init(self.max_graph_bs, vocab_size, self.device)

        pbar = tqdm(
            sorted(self.graph_bs_list, reverse=True),
            desc="Preparing for capturing CUDA graphs...",
            unit="batch",
            disable=not get_tp_info().is_primary(),  # disable for non-primary ranks
        )
        pool = None
        # 从大到小抓：第一张（最大）抓完把它的显存池留下来，后面的小图全部复用
        for bs in pbar:
            free_memory = get_free_memory(self.device)
            pbar.desc = f"Capturing graphs: bs = {bs:<3} | avail_mem = {mem_GB(free_memory)}"
            pbar.refresh()
            graph = torch.cuda.CUDAGraph()
            # 抓图用的假 batch：全是 dummy_req，而且是 decode 阶段。
            # padded_reqs 要手动置成 reqs —— 这个批本来就是"满的"，没有补位
            batch = Batch(reqs=[self.dummy_req] * bs, phase="decode")
            batch.padded_reqs = batch.reqs
            # 后端据此填好这批的元数据（指向它自己的 capture 缓冲），并挂到 batch 上
            self.attn_backend.prepare_for_capture(batch)
            # 把 batch 的 input_ids/positions/out_loc 换成 capture 缓冲的视图，
            # 这样下面录进图里的读地址就是那块固定缓冲
            self.buffer.set_batch(batch)
            with get_global_ctx().forward_batch(batch):
                # 先热身跑一遍（不录）：cuBLAS workspace、autotune、分配器的惰性动作
                # 都在这一遍里做掉，免得被录进图、或者污染抓图
                self.buffer.logits[:bs] = model.forward()
                with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                    # 这一遍被录进图。录在当前流上，重放也在同一条流（Engine.forward_batch
                    # 里 assert 过），所以录/放的时序天然一致
                    self.buffer.logits[:bs] = model.forward()
            if pool is None:
                pool = graph.pool()  # reuse cuda graph handle to reduce memory
            self.graph_map[bs] = graph

        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory after capturing CUDA graphs: {mem_GB(free_memory)}")

    def can_use_cuda_graph(self, batch: Batch) -> bool:
        """判据只有两条：是 decode 批，且真实请求数不超上限。

        不看 padded_size —— 那是 pad_batch 的活：只要 size ≤ max_graph_bs，
        按上面的档位表一定能找到一档装下它。
        """
        return batch.is_decode and batch.size <= self.max_graph_bs

    def replay(self, batch: Batch) -> torch.Tensor:
        """把本批数据灌进固定缓冲，然后整段重放。调用前必须已经 pad_batch 过。"""
        assert self.can_use_cuda_graph(batch)
        # ① 数据：真实 input_ids / positions / out_loc 写进 capture 缓冲
        self.buffer.copy_from(batch)
        # ② 找到尺寸正好对上的一档（pad_batch 保证了 padded_size 一定在表里）
        g = self.graph_map[batch.padded_size]
        # ③ 元数据：page_table / cu_seqlens 等 copy 进后端缓冲 —— 必须在 replay 之前，
        #    因为图里读的就是那些地址
        self.attn_backend.prepare_for_replay(batch)
        # ④ 重放：一次 launch 放出整段 kernel
        g.replay()
        # 只切真实请求那几行；补位 dummy 行的 logits 是垃圾
        # （Engine.forward_batch 里还会再切一次 logits[:batch.size]，是幂等的）
        return self.buffer.logits[: batch.size]

    def pad_batch(self, batch: Batch) -> None:
        """把 batch 补到档位表里的某一档：reqs 后面接若干个 dummy_req。

        由调度器在组装 positions / out_loc 之前调用（见 Scheduler._prepare_batch），
        所以后面那些按 padded_reqs 拍的输入天然就是补齐后的长度。
        不能上图（prefill、或超过上限）时就不补，padded_size 就是 size。
        """
        padded_size = (  # choose the first available batch size
            next(bs for bs in self.graph_bs_list if bs >= batch.size)
            if self.can_use_cuda_graph(batch)
            else batch.size
        )
        batch.padded_reqs = batch.reqs + [self.dummy_req] * (padded_size - batch.size)

    # NOTE: This must be called before freeing NCCL resources to prevent program hang
    def destroy_cuda_graphs(self) -> None:
        """拆图。del + gc.collect() 是为了让引用计数立刻归零、显存池马上还给驱动
        （CUDA graph 持有的显存不走 PyTorch 分配器）。顺序要求见上面那行 NOTE：
        必须在释放 NCCL 资源之前调用，否则会挂（Engine.shutdown 就是按这个顺序）。
        """
        del self.graph_map
        gc.collect()
