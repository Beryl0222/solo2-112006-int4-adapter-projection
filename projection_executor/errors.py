"""执行器的异常类型。

受理阶段的错误（未知版本、重复 id、输入不合法）立即抛出并拒绝该行；
``drain`` 阶段单行资源不足不抛异常，而是以 :class:`RowResult` 的
``status="resource"`` 逐行返回，保证其它行仍能交付。
"""


class ExecutorError(Exception):
    """执行器领域错误的公共基类。"""


class UnknownAdapterError(ExecutorError):
    """显式引用了发布包中不存在的适配器版本。"""

    def __init__(self, request_id, version):
        self.request_id = request_id
        self.version = version
        super().__init__(
            f"请求 {request_id!r} 显式引用的适配器版本 {version!r} 不存在，该行未受理"
        )


class DuplicateRequestError(ExecutorError):
    """请求 id 与已受理但尚未 drain 的行重复。"""

    def __init__(self, request_id):
        self.request_id = request_id
        super().__init__(f"请求 id {request_id!r} 已存在，重复受理被拒绝")


class BadRequestError(ExecutorError):
    """请求输入不符合协议（维度或有限性）。"""


class ResourceExhausted(ExecutorError):
    """发布/驻留等前置动作本身超过资源预算。

    单行执行时的预算超限时不使用本异常（那属于逐行结果），它只用于
    适配器发布包本身无法驻留等整体性的前置失败。
"""

    def __init__(self, message, *, kind=None, requested=None, budget=None):
        self.kind = kind
        self.requested = requested
        self.budget = budget
        super().__init__(message)
