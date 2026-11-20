"""协议数据结构：底权重包、适配器发布包、请求与逐行交付结果。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

FP32_BYTES = 4
INT8_MIN, INT8_MAX = -127, 127


@dataclass(frozen=True)
class BaseWeight:
    """INT4 底权重包。

    打包布局不能按逻辑连续矩阵猜测：每个输出通道占 ``packed`` 的一行，
    低 nibble 在前，按 K 轴 ``group`` 分组；``scale``/``zero`` 形状为
    ``[n, ceil(k/group)]``，零点可以非零，K 尾部补位仅用于对齐存储。
    """

    n: int
    k: int
    group: int
    packed: np.ndarray  # uint8 [n, ceil(ceil(k/group)*group / 2)]
    scale: np.ndarray  # float32 [n, n_groups]
    zero: np.ndarray  # 与码值同口径的整数 [n, n_groups]

    @property
    def n_groups(self) -> int:
        return (self.k + self.group - 1) // self.group

    @classmethod
    def from_packet(cls, packet: dict) -> "BaseWeight":
        n = int(packet["n"])
        k = int(packet["k"])
        group = int(packet["group"])
        if group <= 0 or n <= 0 or k <= 0:
            raise ValueError("base 的 n、k、group 必须为正整数")
        n_groups = (k + group - 1) // group
        packed = np.asarray(packet["packed"], dtype=np.uint8)
        expect_cols = ((n_groups * group) + 1) // 2
        if packed.shape != (n, expect_cols):
            raise ValueError(
                f"packed 形状 {packed.shape} 与协议布局 (n={n}, 字节列={expect_cols}) 不符"
            )
        scale = np.asarray(packet["scale"], dtype=np.float32)
        zero = np.asarray(packet["zero"])
        if scale.shape != (n, n_groups) or zero.shape != (n, n_groups):
            raise ValueError("scale/zero 形状必须为 [n, ceil(k/group)]")
        if not np.all(np.isfinite(scale)) or np.any(scale == 0):
            raise ValueError("base scale 必须为有限非零值")
        zero = zero.astype(np.int64)
        return cls(n=n, k=k, group=group, packed=packed, scale=scale, zero=zero)

    def code_at(self, out_channel: int, k_index: int) -> int:
        """按分组边界与低 nibble 在前的约定取原始 4bit 码值。"""
        byte = self.packed[out_channel, k_index >> 1]
        nibble = byte & 0x0F if (k_index & 1) == 0 else (byte >> 4) & 0x0F
        return int(nibble)

    def decode_element(self, out_channel: int, k_index: int) -> float:
        """单个元素的实值 ``(码值-零点)*标度``，补位位置不参与投影。"""
        group_index = k_index // self.group
        code = self.code_at(out_channel, k_index)
        zero = int(self.zero[out_channel, group_index])
        return float((code - zero) * float(self.scale[out_channel, group_index]))


@dataclass(frozen=True)
class Adapter:
    """一个已发布的低秩适配器版本。

    修正为 ``gain * B @ (A @ x)``。系数按 FP32 驻留，
    ``resident_bytes`` 即协议的驻留口径；不同版本（即使秩相同）也是
    不同对象，执行器绝不跨版本共享修正结果。
    """

    name: str
    version: str
    gain: float
    A: np.ndarray  # float32 [r, k]
    B: np.ndarray  # float32 [n, r]

    @property
    def rank(self) -> int:
        return int(self.A.shape[0])

    @property
    def resident_bytes(self) -> int:
        return int(self.A.size * FP32_BYTES + self.B.size * FP32_BYTES)

    @classmethod
    def from_packet(cls, name: str, version: str, packet: dict, k: int, n: int) -> "Adapter":
        A = np.asarray(packet["A"], dtype=np.float32)
        B = np.asarray(packet["B"], dtype=np.float32)
        if A.ndim != 2 or B.ndim != 2:
            raise ValueError(f"适配器 {name} 的 A、B 必须为二维矩阵")
        r = A.shape[0]
        if A.shape != (r, k) or B.shape != (n, r):
            raise ValueError(f"适配器 {name} 几何应为 A[{r},{k}]、B[{n},{r}]")
        if not (np.all(np.isfinite(A)) and np.all(np.isfinite(B))):
            raise ValueError(f"适配器 {name} 系数必须全部有限")
        gain = float(packet.get("gain", 1.0))
        if not np.isfinite(gain):
            raise ValueError(f"适配器 {name} 的 gain 必须有限")
        return cls(name=name, version=version, gain=gain, A=A, B=B)


@dataclass(frozen=True)
class PendingRequest:
    """受理后被固定的请求行：适配器选择在此刻绑定，之后不再改变。"""

    seq: int
    request_id: str
    adapter: Optional[Adapter]  # None 表示显式禁用适配器（null）
    x: np.ndarray  # 持久输入，float32，不计入 scratch


@dataclass
class RowResult:
    """逐行交付/失败记录，字段足以按请求行回查全部口径。"""

    seq: int
    request_id: str
    status: str  # "ok" | "rejected" | "resource"
    adapter_version: Optional[str] = None
    codes: Optional[np.ndarray] = None  # 实际交付的 INT8 码值
    scale: Optional[float] = None  # 该行的量化比例
    downstream: Optional[np.ndarray] = None  # next @ (codes * scale)
    reason: Optional[str] = None
    detail: dict = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return self.status != "rejected"


@dataclass
class DrainReport:
    """一次 drain 的完整报告，行顺序与受理顺序（即输入 id 先后）一致。"""

    rows: list[RowResult]
    waves: list[dict]
    peak_scratch_bytes: int
    peak_resident_bytes: int

    def by_id(self, request_id: str) -> RowResult:
        for row in self.rows:
            if row.request_id == request_id:
                return row
        raise KeyError(request_id)

    @property
    def delivered(self) -> list[RowResult]:
        return [row for row in self.rows if row.status == "ok"]

    @property
    def resource_failures(self) -> list[RowResult]:
        return [row for row in self.rows if row.status == "resource"]

    @property
    def rejected(self) -> list[RowResult]:
        return [row for row in self.rows if row.status == "rejected"]
