"""独立 FP64 参考实现：与执行器走完全不同的代码路径。

用途是测试中交叉核对数值口径：直接按协议文字完整解包 INT4 底权重、
整体矩阵乘、整体量化，不经过工作区/流式逻辑，避免实现错误自我印证。
"""

from __future__ import annotations

import numpy as np


def reference_base_matrix(base_packet: dict) -> np.ndarray:
    """按协议把 INT4 包完整解成 float64[n, k]，含分组/补位/非零零点处理。"""
    n = int(base_packet["n"])
    k = int(base_packet["k"])
    group = int(base_packet["group"])
    packed = np.asarray(base_packet["packed"], dtype=np.uint8)
    scale = np.asarray(base_packet["scale"], dtype=np.float64)
    zero = np.asarray(base_packet["zero"], dtype=np.float64)
    n_groups = (k + group - 1) // group

    weights = np.zeros((n, k), dtype=np.float64)
    for ch in range(n):
        for ki in range(k):
            byte = packed[ch, ki // 2]
            if ki % 2 == 0:
                code = byte & 0x0F
            else:
                code = (byte >> 4) & 0x0F
            g = ki // group
            weights[ch, ki] = (float(code) - zero[ch, g]) * scale[ch, g]
    assert n_groups == scale.shape[1]
    return weights


def reference_project(
    base_packet: dict,
    adapter_packet: dict | None,
    x,
    bias,
    clip,
) -> tuple[np.ndarray, float, np.ndarray]:
    """返回 (INT8 码值, 行比例, 截断前实值)，全 FP64。"""
    w = reference_base_matrix(base_packet)
    x64 = np.asarray(x, dtype=np.float64)
    real = w @ x64
    if adapter_packet is not None:
        A = np.asarray(adapter_packet["A"], dtype=np.float64)
        B = np.asarray(adapter_packet["B"], dtype=np.float64)
        gain = float(adapter_packet.get("gain", 1.0))
        real = real + gain * (B @ (A @ x64))
    real = real + np.asarray(bias, dtype=np.float64)
    lo, hi = float(clip[0]), float(clip[1])
    clipped = np.clip(real, lo, hi)
    max_abs = float(np.max(np.abs(clipped)))
    if max_abs == 0.0:
        scale = 1.0
        codes = np.zeros(clipped.shape, dtype=np.int8)
        return codes, scale, real
    scale = max_abs / 127.0
    codes = np.clip(np.rint(clipped / scale), -127, 127).astype(np.int8)
    return codes, float(scale), real


def reference_downstream(next_matrix, codes, scale) -> np.ndarray:
    return np.asarray(next_matrix, dtype=np.float64) @ (
        codes.astype(np.float64) * scale
    )
