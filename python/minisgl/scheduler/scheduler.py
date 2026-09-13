from __future__ import annotations

from typing import TYPE_CHECKING, List, NamedTuple, NoReturn, Set, Tuple, TypeAlias

import torch
from minisgl.core import Batch, Req
from minisgl.env import ENV
from minisgl.message import (
    AbortBackendMsg,
    BaseBackendMsg,
    BatchBackendMsg,
    DetokenizeMsg,
    ExitMsg,
    UserMsg,
)
from minisgl.utils import init_logger, load_tokenizer

from .cache import CacheManager
from .config import SchedulerConfig
from .decode import DecodeManager
from .io import SchedulerIOMixin
from .prefill import ChunkedReq, PrefillManager
from .table import TableManager

if TYPE_CHECKING:
    from minisgl.engine import BatchSamplingArgs, ForwardOutput


logger = init_logger(__name__)

Indice2D: TypeAlias = Tuple[torch.Tensor, torch.Tensor]


# For overlap scheduling, we also need to cache some other data to avoid IMA
class ForwardInput(NamedTuple):
    batch: Batch
    sample_args: BatchSamplingArgs
    input_tuple: Indice2D  # (token_mapping, positions)
    write_tuple: Indice2D  # (req_mapping, seq_lens or -1)


ForwardData: TypeAlias = "Tuple[ForwardInput, ForwardOutput]"


class Scheduler(SchedulerIOMixin):
    def __init__(self, config: SchedulerConfig):
        # 函数内 import：只有真正构造调度器时才把整个 engine 包导进来
        from minisgl.engine import Engine

        self.engine = Engine(config)  # 模型权重、KV cache、CUDA graph 都在这一步建好

        # 用另一个 stream 来 overlap 元数据处理和模型计算：调度侧（拼元数据、H2D 拷贝）跑在
        # self.stream 上，模型前向跑在 engine.stream 上，两者用 wait_stream 同步（见 overlap_loop）
        self.device = self.engine.device
        self.stream = torch.cuda.Stream(device=self.device)  # 调度器自己的 stream
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)  # 进入 engine stream 的 ctx
        torch.cuda.set_stream(self.stream)  # 此后调度器的 CUDA 操作默认发往 self.stream

        # 初始化各个管理器
        # page_table 的行号（slot）：每个在跑的请求占一行，请求结束再归还
        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
        # KV cache 的物理页 + 前缀缓存（匹配 / 插入 / 驱逐），显存主要花在这里
        self.cache_manager = CacheManager(
            self.engine.num_pages, config.page_size, self.engine.page_table, config.cache_type
        )
        # decode 阶段正在跑的请求集合（内部是 Set[Req]，按 can_decode 过滤）
        self.decode_manager = DecodeManager(config.page_size)
        # 待 prefill 的请求队列：按 token 预算把长 prompt 切成分批 prefill 的 chunk
        self.prefill_manager = PrefillManager(
            self.cache_manager, self.table_manager, self.decode_manager
        )

        # 一些取用方便的别名
        # 上一轮刚结束的请求：overlap 调度下用来跳过重复释放（Req 按身份哈希，可直接入 Set）
        self.finished_reqs: Set[Req] = set()
        self.tokenizer = load_tokenizer(config.model_path)  # 只用来取 eos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.token_pool = self.table_manager.token_pool  # device 上 token id 的存放处
        self.prefill_budget = config.max_extend_tokens  # 单次 prefill 的 token 预算
        # self.config = config

        # 初始化 I/O mixin：建好与 tokenizer 进程之间的 zmq 通道；多 TP rank 时还负责广播消息
        super().__init__(config, self.engine.tp_cpu_group)

    def run_when_idle(self) -> None:
        """Called when the scheduler is idle to perform background tasks."""
        logger.info_rank0("Scheduler is idle, waiting for new reqs...")
        self.cache_manager.check_integrity()

    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """
        The main loop of overlapping scheduling and execution.

        It will overlap the execution of current batch and processing of last batch's results,
        which can effectively hide CPU latency and improve GPU utilization.
        """
        # 只有真的没活干时才阻塞等消息：还有在飞的批次要收尾，或有 prefill / decode 可以
        # 调度，都必须立刻返回进入下面的流程，否则 GPU 会空转
        blocking = not (
            last_data is not None  # don't block if we have a batch to be processed
            or self.prefill_manager.runnable
            or self.decode_manager.runnable
        )
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)  # 新请求（UserMsg）、abort 都在这里入队 / 释放

        forward_input = self._schedule_next_batch()  # 拼元数据、分配页、H2D 拷贝（CPU 重活）
        ongoing_data = None
        if forward_input is not None:
            with self.engine_stream_ctx:  # run the batch in the engine's stream
                # 先让 engine stream 等 self.stream 上刚备好的元数据和拷贝，再提交 forward；
                # 这里只是把 kernel 排进队列，不等它算完（async）
                self.engine.stream.wait_stream(self.stream)
                ongoing_data = (forward_input, self._forward(forward_input))

        # 关键：上面刚提交的批次还在 GPU 上跑，这里处理的是上一轮的结果，两者天然重叠。
        # _process_last_data 开头的 copy_done.synchronize() 是整条流水线上唯一等 GPU 的地方
        self._process_last_data(last_data)
        return ongoing_data  # 交给下一轮当 last_data

    def normal_loop(self) -> None:
        blocking = not (self.prefill_manager.runnable or self.decode_manager.runnable)
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(ongoing_data)

    @torch.inference_mode()
    def run_forever(self) -> NoReturn:
        """调度器的主循环，永不正常返回，所以返回类型是 NoReturn。

        两种退出方式都是靠异常"炸"出来的：
          - 在线模式：收到 ExitMsg 时 _process_one_msg 抛 KeyboardInterrupt，
            由启动方捕获（见 server/launch.py）
          - 离线模式：请求全部跑完且空闲时抛 RequestAllFinished，
            由 LLMEngine.generate() 捕获（见 llm/llm.py）

        inference_mode 下创建的张量都不带 autograd 记账，省时省显存。
        """
        if ENV.DISABLE_OVERLAP_SCHEDULING:  # 环境变量 MINISGL_DISABLE_OVERLAP_SCHEDULING
            # 保守模式：整个循环都跑在 engine 的 stream 上，调度与计算串行执行，
            # 所以只需进循环前同步一次；实现简单好调试，代价是 CPU 拼元数据时 GPU 闲置
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                while True:
                    self.normal_loop()
        else:
            # 前提：当前 stream 仍是调度器自己的（在 __init__ 里设过）
            assert torch.cuda.current_stream() == self.stream
            data = None  # 上一轮提交、可能还在 GPU 上跑的批次；首轮没有，传 None
            # 流水线深度为 2：本轮 overlap_loop 一边调度新批次，一边处理上一批的结果
            while True:
                data = self.overlap_loop(data)

    def shutdown(self) -> None:
        torch.cuda.synchronize(self.device)
        self.sync_all_ranks()
        self.engine.shutdown()

    def _process_last_data(self, last_data: ForwardData | None) -> None:
        """收取上一轮提交的那批 forward 的结果（此时 GPU 正在跑本轮的新批，两者重叠）。

        last_data 是上一轮 overlap_loop 的返回值 (ForwardInput, ForwardOutput)。
        这里做的是 CPU 侧的收尾：等采样结果拷回 CPU，然后记结果、推进请求、释放资源。
        """
        if last_data is None:  # 首轮还没有在飞的批次
            return

        # 拆开而非 last_data[0].batch / last_data[1] 这样混着索引，读起来清楚些
        forward_input, (_, next_tokens_cpu, copy_done) = last_data
        batch = forward_input.batch
        # 整条流水线上唯一等 GPU 的地方：等这批的 D2H 拷贝完成（此刻 GPU 在跑本轮新批）
        copy_done.synchronize()

        reply: List[DetokenizeMsg] = []
        new_finished_reqs: Set[Req] = set()
        with self.cache_manager.lazy_free_region():  # 期间释放的页先攒着，退出时一并归还
            for i, req in enumerate(batch.reqs):
                if isinstance(req, ChunkedReq):
                    continue  # 被切块的 prefill 不采样，结果无意义（见 ChunkedReq）
                next_token_tensor = next_tokens_cpu[i]  # 名字改掉：后面那个是 int
                # 追加到 host 侧序列，之后缓存前缀时要用
                req.append_host(next_token_tensor.unsqueeze(0))
                next_token = int(next_token_tensor.item())
                # 结束条件：已达长度上限（can_decode 变 False），或采到 EOS（除非要求忽略 EOS）
                finished = not req.can_decode
                if not req.sampling_params.ignore_eos:
                    finished = finished or next_token == self.eos_token_id
                reply.append(DetokenizeMsg(uid=req.uid, next_token=next_token, finished=finished))

                # NOTE: overlap 调度下请求可能被释放两次（上一轮已释放，这批还在飞），跳过第二次
                if finished and req not in self.finished_reqs:
                    self.decode_manager.remove_req(req)
                    self._free_req_resources(req)
                    new_finished_reqs.add(req)
                elif batch.is_prefill:  # for prefill, non-chunk req, cache the prefix
                    self.cache_manager.cache_req(req, finished=False)

        self.finished_reqs = new_finished_reqs  # 只记本轮新结束的，供下一轮判重
        self.send_result(reply)  # 回给 tokenizer 进程做 detokenize

    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        """处理一条后端消息（新请求 / 取消 / 退出等）。

        overlap_loop 会把本轮到手的消息全部处理完才调度下一批，所以这里的效果
        （新请求入队、abort 释放资源）在本次调度中就会生效。
        """
        if isinstance(msg, BatchBackendMsg):
            # 成批的消息（多 rank 广播、离线模式一次投递多条）展开逐条处理
            for msg in msg.data:
                self._process_one_msg(msg)
        elif isinstance(msg, ExitMsg):
            # 退出信号：直接抛异常跳出 run_forever 的 while True（见 run_forever）
            raise KeyboardInterrupt
        elif isinstance(msg, UserMsg):
            logger.debug_rank0("Received user msg: %s", msg)
            # 长度校验：输入已超出 max_seq_len 就丢弃，否则把 max_tokens 夹到剩余空间
            input_len, max_seq_len = len(msg.input_ids), self.engine.max_seq_len
            max_output_len = max_seq_len - input_len
            if max_output_len <= 0:
                # warning_rank0 返回 None，这里等价于"记条日志然后 return"
                return logger.warning_rank0(
                    f"Input sequence length {input_len} exceeds {max_seq_len}, "
                    f"request {msg.uid} is dropped."
                )
            if msg.sampling_params.max_tokens > max_output_len:
                # NOTE: 原地修改；离线模式下一个 SamplingParams 可能被所有请求共用
                # （见 llm.py 的 [sp] * len(prompts)），改一个会连带影响其他请求
                msg.sampling_params.max_tokens = max_output_len
                logger.warning_rank0(
                    f"Adjust max_tokens to {max_output_len} for request {msg.uid}."
                )
            # 只是进待 prefill 队列，此时还没占 table slot 和 KV cache
            self.prefill_manager.add_one_req(msg)
        elif isinstance(msg, AbortBackendMsg):
            # 取消：先在待 prefill 队列里找（可能还没轮到它），再到 decode 集合里找
            logger.debug_rank0("Aborting request %d", msg.uid)
            req_to_free = self.prefill_manager.abort_req(msg.uid)
            req_to_free = req_to_free or self.decode_manager.abort_req(msg.uid)
            # 开始 prefill 前被取消的请求没有 chunked_req，也就没有资源需要释放
            if req_to_free is not None:
                self._free_req_resources(req_to_free)
        else:
            # 未知消息类型宁可报错也不要静默忽略，否则请求会一直卡着没人处理
            logger.error(f"Unknown message type: {type(msg)}")
            raise NotImplementedError

    def _free_req_resources(self, req: Req) -> None:
        self.table_manager.free(req.table_idx)
        self.cache_manager.cache_req(req, finished=True)

    def _prepare_batch(self, batch: Batch) -> ForwardInput:
        """把选出来的 Req 拼装成一次 forward 的输入。全是 CPU 侧的拼装和 H2D 拷贝，
        正是 overlap 调度要藏起来的那部分耗时。

        步骤之间有顺序依赖：先补齐 batch，再分配 KV 页，之后才能按页表算出 out_loc。
        """
        # 1. CUDA graph 要求固定 batch size：用 dummy_req 把不足的位置补上
        #    （decode 命中 graph 时补齐；不命中或 prefill 时 padded_reqs == reqs）
        self.engine.graph_runner.pad_batch(batch)
        # 2. 为每个请求新覆盖的 [cached_len, device_len) 分配物理页，写进 page_table；
        #    用的是 reqs 而不是 padded_reqs：dummy_req 那一行早已指向 dummy page
        self.cache_manager.allocate_paged(batch.reqs)
        # 3. 每个 token 的绝对位置（RoPE 用）；按 padded_reqs 拼，dummy 各自贡献 1 个
        batch.positions = _make_positions(batch, self.device)
        # 4. 两张索引表（host 侧 pinned tensor，异步拷到 device）：
        #    input_tuple：（每个 token 的行号、位置）—— 用来索引 page_table / token_pool
        input_mapping = _make_input_tuple(batch, self.device)
        #    write_tuple：（每个请求的行号、写回位置）—— 只含真实请求，长度 = batch.size；
        #    位置为 -1 表示该请求已达长度上限，不写回（见 _make_write_tuple）
        write_mapping = _make_write_tuple(batch, self.device)
        # 5. 按第 4 步的索引从页表取出每个 token 的 KV 物理槽位（依赖第 2 步已写好页表）
        batch.out_loc = self.engine.page_table[input_mapping]
        # 6. 交给 attention 后端拼自己的元数据（cu_seqlens 等）；CUDA graph 路径下换成抓图那份
        self.engine.attn_backend.prepare_metadata(batch)
        # input_ids 不在这里取，等 _forward 里再按同一份 input_mapping 从 token_pool 取
        return ForwardInput(
            batch=batch,
            sample_args=self.engine.sampler.prepare(batch),  # 采样参数，顺序与 batch.reqs 一致
            input_tuple=input_mapping,
            write_tuple=write_mapping,
        )

    def _schedule_next_batch(self) -> ForwardInput | None:
        """选出下一批要跑什么：先让 prefill / decode manager 选出 Req，再交给 _prepare_batch
        补齐 CUDA graph 的 batch size、分配 KV 页、拼好送进 kernel 的张量。

        返回 None 表示这次没有可跑的批次（overlap_loop 会跳过 forward）。
        """
        # 调度策略：prefill 优先 —— 只要有待 prefill 的请求就先做 prefill，decode 排后面。
        # 能保护的是 KV 空间而不是延迟：PrefillAdder 用 decode_manager.inflight_tokens 预留了
        # 在飞 decode 所需的空间；但 prefill 队列一直非空时 decode 会持续让位。
        # TODO: support other policies: e.g. DECODE first
        # Batch 没有定义 __bool__ / __len__，所以下面这个 or 等价于"谁非 None 就取谁"
        batch = (
            self.prefill_manager.schedule_next_batch(self.prefill_budget)
            or self.decode_manager.schedule_next_batch()
        )
        # None 表示这批没活可干（没请求、预算用尽或没空间），交给上层决定是否阻塞
        return self._prepare_batch(batch) if batch else None

    def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
        """提交一次 forward：取输入 token → 跑模型 → 采样结果写回 token_pool → 更新 decode 集合。

        整个函数都在 engine stream 上执行（调用方 overlap_loop 用 engine_stream_ctx 包着），
        其中只有第 4 步是纯 Python 记账。函数立即返回，此时 GPU 上的活通常还没算完。
        """
        # ForwardInput 是 NamedTuple(batch, sample_args, input_tuple, write_tuple)，
        # 后两项这里改名为 in/out：分别负责"读 token"和"写回采样结果"
        batch, sample_args, input_mapping, output_mapping = forward_input
        # 1. 按 (table_idx, position) 从 token_pool 里 gather 出本批待计算 token 的 id；
        #    prompt 的 H2D 拷贝在 PrefillAdder 里就做过了，这里只是取出来拼成扁平数组
        batch.input_ids = self.token_pool[input_mapping]
        # 2. 跑模型：内部会对每个 req 调 complete_one() 推进 cached_len / device_len，
        #    并在最后采样；CUDA graph 路径下 replay 会先把输入拷进 capture buffer 再重放
        forward_output = self.engine.forward_batch(batch, sample_args)
        # 3. 把采样结果 scatter 回 token_pool，位置是 _prepare_batch 时算好的 req.device_len；
        #    下一步 decode 就把这个位置当输入读出来（-1 表示不写，落在 dummy 行）
        self.token_pool[output_mapping] = forward_output.next_tokens_gpu
        # 4. 更新 decode 集合：刚 prefill 完的请求从这里进入 decode，已到长度上限的被剔除；
        #    它依赖第 2 步的 complete_one 已更新 remain_len，所以必须放在 forward 之后
        self.decode_manager.filter_reqs(forward_input.batch.reqs)
        # 立刻返回：GPU 上的活还在排队；copy_done_event 留给下一轮 _process_last_data 等待
        return forward_output


def _make_positions(batch: Batch, device: torch.device) -> torch.Tensor:
    needed_size = sum(r.extend_len for r in batch.padded_reqs)
    indices_host = torch.empty(needed_size, dtype=torch.int32, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        torch.arange(
            req.cached_len,
            req.device_len,
            dtype=torch.int32,
            out=indices_host[offset : offset + length],
        )
        offset += length
    return indices_host.to(device, non_blocking=True)


def _make_input_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_host = torch.empty(len(batch.positions), dtype=torch.int64, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        mapping_host[offset : offset + length].fill_(req.table_idx)
        offset += length
    return mapping_host.to(device, non_blocking=True), batch.positions.to(torch.int64)


def _make_write_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_list = [req.table_idx for req in batch.reqs]
    mapping_host = torch.tensor(mapping_list, dtype=torch.int64, pin_memory=True)
    write_list = [(req.device_len if req.can_decode else -1) for req in batch.reqs]
    write_host = torch.tensor(write_list, dtype=torch.int64, pin_memory=True)
    return mapping_host.to(device, non_blocking=True), write_host.to(device, non_blocking=True)
