"""逐行投影数值引擎。

顺序固定为协议要求的：先在实值上得到底投影与低秩修正之和，再做
偏置、闭区间截断，最后按行量化为有符号 INT8；下游矩阵消费的是
交付码值按行比例恢复出来的实值，而不是未量化中间值。

为满足工作区预算，INT4 底权重从不整体解包：每个输出通道、每个 K
分组各解一个小块到 scratch，点积累加到持久输出。所有临时张量都
经过 :class:`Workspace` 记账，预算不足时抛 ``ResourceExhausted``，
不会留下半截码值。
"""

from __future__ import annotations

import numpy as np

from .models import INT8_MAX, INT8_MIN, Adapter, BaseWeight
from .workspace import Workspace


def project_row_fp64(
    base: BaseWeight,
    adapter: Adapter | None,
    x: np.ndarray,
    workspace: Workspace,
    out: np.ndarray,
) -> np.ndarray:
    """把一行实值投影写入调用方提供的持久缓冲 ``out``（float64[n]）。

    底投影与 ``gain * B @ (A @ x)`` 全部在 FP64 上累加，修正不会被
    提前量化丢弃。返回同一个 ``out`` 以便链式调用。
    """
    n, k, group = base.n, base.k, base.group
    if out.shape != (n,) or out.dtype != np.float64:
        raise ValueError("out 必须为 float64[n] 的持久缓冲")

    # ---- 底投影：按输出通道、按 K 分组流式解包，峰值只含一个分组 ----
    with workspace.lease("base_g", (group,), np.float64) as g_tile, \
            workspace.lease("base_xg", (group,), np.float64) as x_tile:
        for ch in range(n):
            acc = 0.0
            packed_row = base.packed[ch]
            for g in range(base.n_groups):
                start = g * group
                width = min(group, k - start)
                for j in range(width):
                    ki = start + j
                    byte = packed_row[ki >> 1]
                    nibble = byte & 0x0F if (ki & 1) == 0 else (byte >> 4) & 0x0F
                    g_tile[j] = (int(nibble) - int(base.zero[ch, g])) * float(
                        base.scale[ch, g]
                    )
                    x_tile[j] = float(x[ki])
                # Python 标量求和，保证 FP64 累加顺序且不产生隐式临时张量
                for j in range(width):
                    acc += g_tile[j] * x_tile[j]
            out[ch] = acc

    # ---- 低秩修正：gain * B @ (A @ x)，按秩分量流式，绝不跨版本缓存 ----
    if adapter is not None:
        r = adapter.rank
        # 利用 NumPy 的 dtype 提升把 FP32 系数直接累加进已记账的 FP64 缓冲，
        # 不产生 .astype() 隐式临时；峰值只含固定的几块逐分量/逐行缓冲。
        with workspace.lease("lora_ax", (1,), np.float64) as ax, \
                workspace.lease("lora_arow", (k,), np.float64) as a_row, \
                workspace.lease("lora_x", (k,), np.float64) as x_full, \
                workspace.lease("lora_b", (1,), np.float64) as b_scalar:
            np.copyto(x_full, x)  # fp32 -> fp64 走提升规则，无转换临时
            ax_values: list[float] = []
            for rr in range(r):
                np.copyto(a_row, adapter.A[rr])
                ax[0] = float(np.dot(a_row, x_full))
                ax_values.append(float(ax[0]))
            for ch in range(n):
                correction = 0.0
                for rr in range(r):
                    np.copyto(b_scalar, adapter.B[ch, rr])
                    correction += float(b_scalar[0]) * ax_values[rr]
                out[ch] += adapter.gain * correction

    return out


def quantize_row(
    real: np.ndarray,
    bias: np.ndarray,
    clip: tuple[float, float],
    workspace: Workspace,
) -> tuple[np.ndarray, float]:
    """偏置 → 闭区间截断 → 逐行 INT8 量化。

    比例为截断后最大绝对值除以 127，零行比例为 1；恰好半整数按
    银行家舍入（向偶数）处理，使用 IEEE-754 的 ``np.rint``。
    返回的码值是最终交付物，属于持久对象，不计入 scratch。
    """
    lo, hi = clip
    if lo > hi:
        raise ValueError("clip 下界不能大于上界")
    with workspace.lease("quant_v", real.shape, np.float64) as v:
        v[:] = real
        v += bias  # 原地，无额外临时张量
        np.clip(v, lo, hi, out=v)
        max_abs = float(np.max(np.abs(v)))
        if not np.isfinite(max_abs):
            raise ValueError("截断后出现非有限值，无法量化")
        if max_abs == 0.0:
            row_scale = 1.0
            v[:] = 0.0
        else:
            row_scale = max_abs / INT8_MAX
            v /= row_scale
        np.rint(v, out=v)  # half-even
        np.clip(v, INT8_MIN, INT8_MAX, out=v)
        codes = v.astype(np.int8, copy=True)  # 交付物：在工作区外独立存活
    return codes, float(row_scale)


def downstream_from_codes(
    next_matrix: np.ndarray,
    codes: np.ndarray,
    row_scale: float,
    workspace: Workspace,
) -> np.ndarray:
    """``next @ (codes * row_scale)``，与实际交付码值严格一致。"""
    m, n = next_matrix.shape
    with workspace.lease("next_rec", codes.shape, np.float64) as recovered, \
            workspace.lease("next_fp64", (m, n), np.float64) as next_fp64:
        np.copyto(recovered, codes)  # int8 -> fp64 提升，无转换临时
        recovered *= row_scale
        np.copyto(next_fp64, next_matrix)
        out = np.empty((m,), dtype=np.float64)
        for row in range(m):
            out[row] = float(np.dot(next_fp64[row], recovered))
    return out
