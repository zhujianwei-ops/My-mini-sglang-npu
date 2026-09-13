"""流式反分词：把 scheduler 一个一个送来的 token 变成"可以立刻发给用户的文本"。

难点在于模型吐的是 token 流，而用户要的是文本流：

  - 一个 token 可能只是某个字符的一部分（byte-level BPE 会把一个 UTF-8 字符的字节
    拆到多个 token 里），单独解码只会得到替换符 U+FFFD（也就是 "�"），所以有时
    必须等下一个 token 来了才能补全；
  - 反过来，把一个词拆成两半推给用户体验很差，能整词发就整词发。

这里的做法是给每个请求维护一个"滑窗"，每轮只重解码最近的一小段（见 DecodeStatus），
再按 sent_offset 切出增量。注意不能只解码最后一个 token：多字节字符会被拆散，
必须带上一点上下文。
"""

from dataclasses import dataclass
from typing import Dict, List

from minisgl.message import DetokenizeMsg
from transformers import PreTrainedTokenizerBase

# Borrowed from sglang


def _is_chinese_char(cp: int):
    """Checks whether CP is the codepoint of a CJK character."""
    # 只用来决定"输出能不能在这里断开"（汉字之间没有空格，不用等空格），不参与语义
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if (
        (cp >= 0x4E00 and cp <= 0x9FFF)
        or (cp >= 0x3400 and cp <= 0x4DBF)  #
        or (cp >= 0x20000 and cp <= 0x2A6DF)  #
        or (cp >= 0x2A700 and cp <= 0x2B73F)  #
        or (cp >= 0x2B740 and cp <= 0x2B81F)  #
        or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
        or (cp >= 0xF900 and cp <= 0xFAFF)
        or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
    ):  #
        return True

    return False


def find_printable_text(text: str):
    """Returns the longest printable substring of text that contains only entire words."""
    # 从左边尽量多吐一点，但别把半个词吐出去
    # （实际调用场景是"最后一个字符还没收全"，见 DetokenizeManager.detokenize）
    # Borrowed from https://github.com/huggingface/transformers/blob/061580c82c2db1de9139528243e105953793f7a2/src/transformers/generation/streamers.py#L99

    # After the symbol for a new line, we flush the cache.
    if text.endswith("\n"):
        return text  # 换行是天然边界，整段都能发
    # If the last token is a CJK character, we print the characters.
    elif len(text) > 0 and _is_chinese_char(ord(text[-1])):
        return text  # 汉字之间没有空格，不用为了等空格而卡住
    # Otherwise if the penultimate token is a CJK character, we print the characters except for the last one.
    elif len(text) > 1 and _is_chinese_char(ord(text[-2])):
        return text[:-1]  # 倒数第二个是汉字：最后一个先留着
    # Otherwise, prints until the last space char (simple heuristic to avoid printing incomplete words,
    # which may change with the subsequent token -- there are probably smarter ways to do this!)
    else:
        # 退到最后一个空格（含），把可能没写完的那个词留下；没有空格就返回空串，继续等
        return text[: text.rfind(" ") + 1]


@dataclass
class DecodeStatus:
    """一个在跑请求的滑窗状态，按 uid 存在 decode_map 里，请求结束就删掉。

    窗口取 decoded_ids[surr_offset:]，整体解码得到 read_str；其中
    [surr_offset, read_offset) 是上一轮在同一窗口起点上已经解过的部分，单独解码得到
    surr_str。本轮新增的文本 = read_str[len(surr_str):]——byte-level BPE 的 decode 就是
    "拼接字节再按 UTF-8 解"，所以 surr_str 一定是 read_str 的前缀，按长度切即可。

    关键约束：两个 offset 都只在"干净的一轮"里推进（见 detokenize），于是
    surr_offset 一定落在完整的字符边界上——否则 read_str 的开头会解出假的 "�"，
    被当成正文发出去。
    """

    decoded_ids: List[int]  # 已收到、要参与解码的 token（EOS 不入列）
    decoded_str: str  # 已经确认的完整文本（只含已经决定要发的部分）
    read_offset: int  # length of read ids  本轮已解码到的位置（decoded_ids 上的下标）
    surr_offset: int  # length of surr ids  窗口起点，也是"已确认"的边界
    sent_offset: int  # length of sent out string  已发给客户端的文本长度


class DetokenizeManager:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        # uid -> DecodeStatus
        self.decode_map: Dict[int, DecodeStatus] = {}
        self.tokenizer = tokenizer
        self.eos_token_id = self.tokenizer.eos_token_id

    def detokenize(self, msgs: List[DetokenizeMsg]) -> List[str]:
        """把一批新 token 变成每个请求的增量文本（结果与 msgs 一一对应）。

        分两趟：先更新状态、收集每个请求要解码的窗口，再一次性 batch_decode。
        HF tokenizer 的开销主要在 Python 侧，批量解码比逐个 decode 快得多。

        NOTE: 一个批次里每个 uid 至多出现一次（scheduler 每个请求每步只回一条），
        所以第二趟里 len(s.decoded_ids) 与第一趟收集窗口时的长度一致。
        """
        read_ids: List[List[int]] = []
        surr_ids: List[List[int]] = []
        for msg in msgs:
            if msg.uid not in self.decode_map:
                # 懒创建：某个请求的第一条消息到达时才建状态
                self.decode_map[msg.uid] = DecodeStatus(
                    decoded_ids=[],
                    decoded_str="",
                    read_offset=0,
                    surr_offset=0,
                    sent_offset=0,
                )
            s = self.decode_map[msg.uid]
            # EOS 不进 decoded_ids：它只是"到此为止"的信号，不该出现在输出文本里
            # （按长度上限结束的那条 next_token 是正常 token，照样入列）
            if not (msg.finished and msg.next_token == self.eos_token_id):
                s.decoded_ids.append(msg.next_token)
            read_ids.append(s.decoded_ids[s.surr_offset :])  # 整个窗口
            surr_ids.append(s.decoded_ids[s.surr_offset : s.read_offset])  # 其中的已知前缀

        read_texts = self.tokenizer.batch_decode(read_ids)
        surr_texts = self.tokenizer.batch_decode(surr_ids)

        incremental_strs: List[str] = []
        for msg, read_str, surr_str in zip(msgs, read_texts, surr_texts, strict=True):
            s = self.decode_map[msg.uid]
            new_text = read_str[len(surr_str) :]  # 本次新增的文本（前缀切掉）
            # Streaming chunk: update the decode status
            if len(new_text) > 0 and not new_text.endswith("�"):
                # 干净的一轮：有新文本，且结尾不是半个多字节字符（不是替换符 "�"）
                # 这时才敢确认文本、把窗口右移
                output_str = s.decoded_str + new_text
                s.decoded_str = output_str
                s.surr_offset = s.read_offset  # 窗口起点追到上一轮已解的位置
                s.read_offset = len(s.decoded_ids)  # 已解位置追到最新
            else:
                # 两种可能：这轮没有新文本（比如只来了一条 EOS），或者最后一个 token
                # 是半个多字节字符（解出来是 "�"）。这时【窗口不动、decoded_str 也不更新】，
                # 下一轮从同一位置重新解码就能拿到完整结果；find_printable_text 只是把
                # 现在就能确定的部分（最后一个空格之前）先垫发给客户端
                new_text = find_printable_text(new_text)
                output_str = s.decoded_str + new_text

            # 注意 sent_offset 可能已经超出 len(decoded_str)：else 分支"预支"了一段
            # 文本却没记进 decoded_str。这样做是安全的，因为下次重解码的结果一定以
            # 这段预支文本开头，它仍然是一个合法的"已发送长度"
            incremental_output = output_str[s.sent_offset :]
            s.sent_offset = len(output_str)
            incremental_strs.append(incremental_output)
            if msg.finished:  # 请求结束，状态可以释放了
                del self.decode_map[msg.uid]

        return incremental_strs
