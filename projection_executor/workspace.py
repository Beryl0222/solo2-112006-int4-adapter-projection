"""投影工作区（scratch）的显式记账。

协议规定 ``scratch_bytes`` 计投影过程中所有临时张量的同时峰值，
持久输入与最终交付值不计；资源限制同样适用于回退路径。因此执行器
不依赖隐式的 NumPy 临时对象，而是让每一块临时缓冲都经过本工作区
申请/释放，超限直接失败，保证不会产出截断码值。
"""

from __future__ import annotations

import numpy as np

_DTYPE_BYTES = {
    np.dtype(np.float64): 8,
    np.dtype(np.float32): 4,
    np.dtype(np.int64): 8,
    np.dtype(np.int32): 4,
    np.dtype(np.int8): 1,
    np.dtype(np.uint8): 1,
}


class Workspace:
    """按字节预算跟踪同时存活的临时 NumPy 张量。"""

    def __init__(self, budget_bytes: int):
        if budget_bytes < 0:
            raise ValueError("scratch 预算不能为负")
        self.budget = int(budget_bytes)
        self._live: dict[str, tuple[np.dtype, int]] = {}
        self._current = 0
        self.peak = 0

    def _item_size(self, dtype) -> int:
        dtype = np.dtype(dtype)
        if dtype not in _DTYPE_BYTES:
            raise TypeError(f"工作区暂不支持 dtype {dtype}")
        return _DTYPE_BYTES[dtype]

    @property
    def used(self) -> int:
        return self._current

    def used_by(self, name: str) -> int:
        dtype, size = self._live[name]
        return size * self._item_size(dtype)

    def allocate(self, name: str, shape, dtype) -> np.ndarray:
        """申请一块零初始化临时张量；超预算时不改变任何现有分配。"""
        dtype = np.dtype(dtype)
        size = int(np.prod(shape)) if isinstance(shape, tuple) else int(shape)
        need = size * self._item_size(dtype)
        if self._current + need > self.budget:
            from .errors import ResourceExhausted

            raise ResourceExhausted(
                f"scratch 超限：需要再分配 {need}B（当前 {self._current}B，"
                f"预算 {self.budget}B，缓冲 {name!r}）",
                kind="scratch",
                requested=self._current + need,
                budget=self.budget,
            )
        if name in self._live:
            raise KeyError(f"临时缓冲 {name!r} 已存活，不能重复申请")
        buffer = np.zeros(shape, dtype=dtype)
        self._live[name] = (dtype, size)
        self._current += need
        if self._current > self.peak:
            self.peak = self._current
        return buffer

    def release(self, name: str) -> None:
        dtype, size = self._live.pop(name)
        self._current -= size * self._item_size(dtype)

    def reset(self) -> None:
        """一波结束后归还全部临时缓冲（持久对象不在此管理）。"""
        self._live.clear()
        self._current = 0

    class _Lease:
        def __init__(self, workspace: "Workspace", name: str, shape, dtype):
            self._ws = workspace
            self.name = name
            self.array = workspace.allocate(name, shape, dtype)

        def __enter__(self) -> np.ndarray:
            return self.array

        def __exit__(self, *exc) -> None:
            self._ws.release(self.name)

    def lease(self, name: str, shape, dtype) -> "Workspace._Lease":
        return Workspace._Lease(self, name, shape, dtype)
