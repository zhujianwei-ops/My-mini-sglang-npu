from __future__ import annotations

from typing import List

import torch
from minisgl.message import TokenizeMsg
from transformers import PreTrainedTokenizerBase


class TokenizeManager:
    """输入侧：把前端送来的文本（或 chat 消息列表）编码成 token id。

    跑在 tokenizer 进程里，是整条链路的第一站：之后这些 id 会以 UserMsg 的形式
    送给 scheduler，真正占 GPU 的是它们。
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer

    def tokenize(self, msgs: List[TokenizeMsg]) -> List[torch.Tensor]:
        """逐个请求编码，返回与 msgs 一一对应的 1D int32 张量列表。"""
        results: List[torch.Tensor] = []
        # TODO: batch tokenization
        # 现在是逐个 encode；HF 的分词主要在 Python 侧循环，攒批收益有限，
        # 所以这里先不改（真要做的话得用 tokenizer(prompts, padding=True) 那种批量接口）
        for msg in msgs:
            if isinstance(msg.text, list):
                # chat 格式（[{role, content}, ...]）：先用模板拼成一段 prompt 文本。
                # tokenize=False 表示模板只负责拼字符串，真正的分词交给下面的 encode；
                # add_generation_prompt=True 会补上"轮到 assistant 说话"的标记，
                # 少了它模型不知道自己该续写什么
                prompt = self.tokenizer.apply_chat_template(
                    msg.text,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                assert isinstance(prompt, str)
            else:
                prompt = msg.text
            input_ids: torch.Tensor = (  # type: ignore
                self.tokenizer.encode(prompt, return_tensors="pt")
            )
            # 拍平成 1D 再转 int32：消息序列化只支持 1D 张量，后端也统一按 int32 用
            results.append(input_ids.view(-1).to(torch.int32))
        return results
