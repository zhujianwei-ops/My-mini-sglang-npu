"""TableManager：按"行号"管理每个在跑请求在 page_table / token_pool 里占的那一行。

page_table 和 token_pool 是两块形状相同的张量（都由 Engine 一次性开好），
行 = 一个在跑的请求，列 = token 位置：

    page_table[row, i]  = 第 i 个 token 的 KV 落在哪个物理槽位（attention 按它取 KV）
    token_pool[row, i]  = 第 i 个 token 的 token id（forward 时 gather 出来当 input_ids）

同一个行号（Req.table_idx）同时索引这两张表，所以本类只管"发行号 / 收行号"，
不碰表里的内容：

    page_table 的内容  ← CacheManager.allocate_paged / _write_page_table 填
    token_pool 的内容  ← PrefillAdder 填 prompt 部分，Scheduler._forward 写回采样结果

行数是硬上限：`available_size` 就是"还能同时跑几个请求"，它为 0 时 PrefillAdder
直接就不再接收新请求了（见 _try_allocate_one 开头那道闸）。

⚠ 两处易混（名字像，含义不同）：
  - 本类的 `_free_slots` 存的是**行号**（0 .. max_running_reqs-1 的整数），
    而 CacheManager.free_slots 存的是 **KV 物理槽位**（按页对齐、单位是 token）；
  - 本类的 `available_size` 数的是"还能接几个请求"，
    CacheManager.available_size 数的是"还能放多少 token 的 KV"。

最后一行（行号 max_running_reqs）属于 CUDA graph padding 用的 dummy 请求
（Engine.dummy_req，engine.py:89），不从这里分配 —— 但它也会去读 token_pool，
这就是 __init__ 里那条 NOTE 的来由。
"""

import torch


class TableManager:
    def __init__(self, max_running_reqs: int, page_table: torch.Tensor) -> None:
        # 行号池的容量：这里只是记下来备查，仓库里没有别处读它
        self._max_running_reqs = max_running_reqs
        # 空闲行号池，初始是 [0, 1, ..., max_running_reqs-1]，不含 dummy 那一行
        self._free_slots = list(range(max_running_reqs))
        # 只存引用：(max_running_req + 1, aligned_max_seq_len) 的 int32 张量由
        # Engine 持有（engine.py:69），本类只负责把它的行号借出去
        self.page_table = page_table
        # NOTE: dummy request also use this pool to get the input ids, so we need to
        # make sure the token pool is initialized with valid values (token_id = 0).
        # token id 池：和 page_table 同形状（含 dummy 那一行）、int32，多占一份
        # (行数 × 最大长度 × 4B) 的显存。必须 zeros 而不能 empty：forward 时 input_ids
        # 是按行号直接从 token_pool gather 出来的，没人写过的位置（dummy 行、padding 到
        # cuda graph batch size 的那些行）同样会被读到，垃圾值可能越过 vocab_size。
        self.token_pool = torch.zeros_like(page_table, dtype=torch.int32)

    @property
    def available_size(self) -> int:
        # 还能接几个请求（不是 token 数），见模块注释里的提醒
        return len(self._free_slots)

    def allocate(self) -> int:
        # 借一个行号：从尾部 pop，所以是 LIFO，先借到的行号最大
        return self._free_slots.pop()

    def free(self, slot: int) -> None:
        # 归还行号；唯一调用点是 Scheduler._free_req_resources（请求结束或被取消时），
        # 那里先还行号、再 cache_req(finished=True) 还 KV
        self._free_slots.append(slot)
