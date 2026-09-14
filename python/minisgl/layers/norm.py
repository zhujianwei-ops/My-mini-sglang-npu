"""RMSNorm，两种形态：

    RMSNorm        朴素版：算归一化。用在 QK-Norm 上（对每个 head 的 head_dim 维做）
    RMSNormFused   融合版：residual 相加 + 归一化一个 kernel 搞定。用在主干上

主干上之所以要融合，是因为 transformer 的每一层都是
`x = x + sublayer(norm(x))` 这个形状，而"算残差"和"做归一化"要读同一份数据 ——
分成两步就得多写一次显存、多读一次（decode 时每个张量都很小，访存才是瓶颈）。
"""

from typing import Tuple

import torch

from .base import BaseOP


class RMSNorm(BaseOP):
    """标准 RMSNorm：y = x / rms(x) * weight。kernel 直接来自 flashinfer。"""

    def __init__(self, size: int, eps: float) -> None:
        # 延迟导入：flashinfer 的 import 有开销，放在模块顶层会拖慢整个包的导入
        from flashinfer import rmsnorm

        self.eps = eps
        self.weight = torch.empty(size)
        # 注意这俩是**函数**，被存在 __dict__ 里：BaseOP.state_dict 只认 Tensor 和
        # BaseOP，所以它们不会被误当成权重收走（不需要用下划线前缀来躲）
        self.rmsnorm = rmsnorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rmsnorm(x, self.weight, self.eps)

    def forward_inplace(self, x: torch.Tensor) -> None:
        """把结果写回 x（flashinfer 的 out= 参数），省一次分配和一次拷贝。"""
        self.rmsnorm(x, self.weight, self.eps, out=x)


class RMSNormFused(BaseOP):
    """把「残差相加」和「RMSNorm」融成一个 kernel（flashinfer 的 fused_add_rmsnorm）。

    语义（也是调用方 LlamaDecoderLayer 依赖的约定）：
        residual = residual + x      （x 为 None 时 residual 就是 x）
        x        = rmsnorm(residual)
        return x, residual
    两个张量都是**原地**改写的，所以调用方必须用返回值重新绑定变量。

    这样一层里只需一次融合 kernel 就推进了残差流，而不是「先 add 再 norm」两次访存。
    """

    def __init__(self, size: int, eps: float) -> None:
        from flashinfer import fused_add_rmsnorm, rmsnorm

        self.eps = eps
        self.weight = torch.empty(size)
        self.rmsnorm = rmsnorm
        self.fused_add_rmsnorm = fused_add_rmsnorm

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            # 第一次进入（没有历史残差）：只归一化，并把输入本身当作新的残差
            return self.rmsnorm(x, self.weight, self.eps), x
        # 原地更新：residual += x，x = rmsnorm(residual)；返回的就是这两个被改写的张量
        self.fused_add_rmsnorm(x, residual, self.weight, self.eps)
        return x, residual
