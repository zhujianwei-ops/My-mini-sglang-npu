"""架构名 → 模型类的注册表。

刻意存的是**字符串**而不是类对象：这样只有真正用到的那一个模型文件才会被 import
（各模型文件都会拉进 flashinfer 之类的重依赖，启动时没必要全加载）。
"""

import importlib

from .config import ModelConfig

_MODEL_REGISTRY = {
    "LlamaForCausalLM": (".llama", "LlamaForCausalLM"),
    "Qwen2ForCausalLM": (".qwen2", "Qwen2ForCausalLM"),
    "Qwen3ForCausalLM": (".qwen3", "Qwen3ForCausalLM"),
    "Qwen3MoeForCausalLM": (".qwen3_moe", "Qwen3MoeForCausalLM"),
    "MistralForCausalLM": (".mistral", "MistralForCausalLM"),
    # 多模态模型只跑它的文本部分：视觉塔的权重在 load_weight 里被跳过，
    # 所以这里直接复用 Mistral 的文本实现
    "Mistral3ForConditionalGeneration": (".mistral", "MistralForCausalLM"),
}


def get_model_class(model_architecture: str, model_config: ModelConfig):
    """按架构名 import 对应模块并**实例化**（不是返回类），所以调用方拿到的是模型对象。

    package=__package__ 让 ".llama" 这样的相对路径解析到 minisgl.models。
    """
    if model_architecture not in _MODEL_REGISTRY:
        raise ValueError(f"Model architecture {model_architecture} not supported")
    module_path, class_name = _MODEL_REGISTRY[model_architecture]
    module = importlib.import_module(module_path, package=__package__)
    model_cls = getattr(module, class_name)
    return model_cls(model_config)


__all__ = ["get_model_class"]
