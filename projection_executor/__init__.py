"""共享 INT4 底权重、多低秩适配器的混合批次投影执行器。

公开入口见 :class:`projection_executor.ProjectionExecutor` 与
:func:`projection_executor.run_capture`。所有数值口径与资源口径以
``交接协议.md`` 为准。
"""

from .errors import (
    ExecutorError,
    UnknownAdapterError,
    DuplicateRequestError,
    BadRequestError,
    ResourceExhausted,
)
from .models import BaseWeight, Adapter, PendingRequest, RowResult, DrainReport
from .executor import ProjectionExecutor
from .capture import run_capture

__all__ = [
    "ExecutorError",
    "UnknownAdapterError",
    "DuplicateRequestError",
    "BadRequestError",
    "ResourceExhausted",
    "BaseWeight",
    "Adapter",
    "PendingRequest",
    "RowResult",
    "DrainReport",
    "ProjectionExecutor",
    "run_capture",
]
