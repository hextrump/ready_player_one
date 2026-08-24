"""
lie_detector.cpu_preprocessor — UETrack 仓库 Preprocessor 的 CPU 版

仓库原版 mean/std 绑 .cuda() (torch1.11 时代), bot 环境 torch2.10 CPU 直接崩。
这里落 CPU; 与仓库逻辑一致: 6 通道 (RGB×2 复制) 用 mm_mean/mm_std, 3 通道用单 mean/std。
"""
from __future__ import annotations

import numpy as np
import torch


class CPUPreprocessor:
    """输入 HxWxC (C=3 或 6), 输出 (1,C,H,W) 归一化 tensor (CPU)。"""

    def __init__(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.mm_mean = torch.tensor([0.485, 0.456, 0.406] * 2).view(1, 6, 1, 1)
        self.mm_std = torch.tensor([0.229, 0.224, 0.225] * 2).view(1, 6, 1, 1)

    def process(self, arr: np.ndarray) -> torch.Tensor:
        mm = arr.shape[-1] == 6
        mean = self.mm_mean if mm else self.mean
        std = self.mm_std if mm else self.std
        t = torch.tensor(arr, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        return ((t / 255.0) - mean) / std
