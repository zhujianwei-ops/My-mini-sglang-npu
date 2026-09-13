"""消息对象的序列化工具。

把 dataclass 消息（可能内嵌张量）变成纯 dict，交给 msgpack 打包后在 zmq 上传；
收到后再反向还原成对象。相当于一个手写的、不依赖 pickle 的"反射式"序列化器。

编码出来的形状（`__type__` 记录类名，是还原时的唯一线索）：

    {
        "__type__": "UserMsg",
        "uid": 3,
        "input_ids": {"__type__": "Tensor", "buffer": b"...", "dtype": "torch.int32"},
        "sampling_params": {"__type__": "SamplingParams", "temperature": 0.0, ...},
    }

解码时用**定义模块的 globals()** 查回类对象（见各消息类的 decoder），所以消息里
嵌套用到的类型必须在那个模块里 import 过，否则会 KeyError。
"""

from __future__ import annotations

from typing import Any, Dict, Type

import numpy as np
import torch


def _serialize_any(value: Any) -> Any:
    """递归编码字段值：msgpack 原生支持的类型原样返回，其余当作嵌套消息再编码。"""
    if isinstance(value, dict):
        return {k: _serialize_any(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        return type(value)(_serialize_any(v) for v in value)  # 保持 list / tuple
    elif isinstance(value, (int, float, str, type(None), bool, bytes)):
        return value
    else:
        # 嵌套 dataclass（如 SamplingParams）走这里，会带上自己的 __type__
        return serialize_type(value)


def serialize_type(self) -> Dict:
    # find all member variables
    # 首参叫 self 只是习惯：它其实是模块级函数，总被当"方法"调用
    # （传进来的可能是消息对象，也可能是裸张量）
    serialized = {}

    if isinstance(self, torch.Tensor):
        # 张量单独处理：只支持 1D（本项目只用来传 input_ids），拆成裸字节 + dtype 字符串
        assert self.dim() == 1, "we can only serialize 1D tensor for now"
        serialized["__type__"] = "Tensor"
        serialized["buffer"] = self.numpy().tobytes()  # 需要 CPU 张量
        serialized["dtype"] = str(self.dtype)
        return serialized

    # normal type
    # 普通 dataclass：类名当标签；dataclass 的字段都在实例 __dict__ 里，直接遍历即可
    serialized["__type__"] = self.__class__.__name__
    for k, v in self.__dict__.items():
        serialized[k] = _serialize_any(v)
    return serialized


def _deserialize_any(cls_map: Dict[str, Type], data: Any) -> Any:
    """_serialize_any 的逆操作。"""
    if isinstance(data, dict):
        if "__type__" in data:
            return deserialize_type(cls_map, data)  # 嵌套消息
        else:
            return {k: _deserialize_any(cls_map, v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        # 注意：msgpack 里的数组解出来是 list，所以原来的 tuple 字段会被还原成 list
        return type(data)(_deserialize_any(cls_map, d) for d in data)
    elif isinstance(data, (int, float, str, type(None), bool, bytes)):
        return data
    else:
        raise ValueError(f"Cannot deserialize type {type(data)}")


def deserialize_type(cls_map: Dict[str, Type], data: Dict) -> Any:
    """按 `__type__` 还原成一个对象；cls_map 通常传定义模块的 globals()。"""
    type_name = data["__type__"]
    # we can only serialize 1D tensor for now
    if type_name == "Tensor":
        buffer = data["buffer"]
        dtype_str = data["dtype"].replace("torch.", "")  # "torch.int32" -> "int32"
        np_dtype = getattr(np, dtype_str)  # 要求 dtype 在 numpy 里有同名类型
        assert isinstance(buffer, bytes)
        np_tensor = np.frombuffer(buffer, dtype=np_dtype)
        # 必须 copy：from_numpy 是零拷贝、会共享 numpy 的内存（这里指向刚解出来的字节），
        # 不 copy 就是只读/悬空内存
        return torch.from_numpy(np_tensor.copy())

    cls = cls_map[type_name]
    kwargs = {}
    for k, v in data.items():
        if k == "__type__":
            continue  # 标签不是字段
        kwargs[k] = _deserialize_any(cls_map, v)
    return cls(**kwargs)  # dataclass 按关键字构造
