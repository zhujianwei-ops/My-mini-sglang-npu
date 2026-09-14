"""models 包的门面。

    config.py    ModelConfig  把 HF 的 config 拍扁成本仓库要的字段
    register.py  架构名 → 模型类（延迟导入）
    weight.py    从 safetensors 流式加载并按 TP 切分
    llama.py / qwen2.py / qwen3.py / qwen3_moe.py / mistral.py   各模型的结构
    base.py      BaseLLMModel：forward() 不接受参数（输入在全局 ctx 里）
    utils.py     各模型共用的 GatedMLP / RopeAttn / QK-Norm
"""

from .base import BaseLLMModel
from .config import ModelConfig, RotaryConfig
from .register import get_model_class
from .weight import load_weight


def create_model(model_config: ModelConfig) -> BaseLLMModel:
    """按 HF config 里的 architectures[0]（如 "Qwen3ForCausalLM"）建模型实例。

    这里建出来的是 **meta 设备上的空骨架**（调用方 Engine 用 `with torch.device("meta")`
    包着），真正的权重随后由 load_weight + load_state_dict 灌进来。
    注意只认第一个架构名。
    """
    return get_model_class(model_config.architectures[0], model_config)


__all__ = ["create_model", "load_weight", "RotaryConfig"]
