"""按交接采集的事件序列回放执行器。

事件协议：``["default", version]``、``["accept", id, choice, x]``、
``["drain"]``。受理期的未知版本等错误不会中断回放，而是按输入顺序
记成 ``rejected`` 行，最终与 drain 交付合并，保证每个请求 id 都可
逐条回查。
"""

from __future__ import annotations

from typing import Any

from .errors import ExecutorError
from .executor import ProjectionExecutor
from .models import DrainReport, RowResult


def _build_executor(capture: dict) -> ProjectionExecutor:
    executor = ProjectionExecutor(
        capture["base"],
        bias=capture["bias"],
        clip=capture["clip"],
        next_matrix=capture["next"],
        resident_bytes=int(capture["resident_bytes"]),
        scratch_bytes=int(capture["scratch_bytes"]),
    )
    for version, packet in capture.get("adapters", {}).items():
        executor.publish_adapter(version, packet)
    return executor


def run_capture(capture: dict) -> DrainReport:
    """回放一次采集，返回保持原始请求顺序的完整报告。"""
    events = capture.get("events")
    if not isinstance(events, list) or any(
        not isinstance(event, list) or not event or not isinstance(event[0], str)
        for event in events
    ):
        raise ValueError("采集资料含有无效的事件列表")

    executor = _build_executor(capture)
    rejected: list[RowResult] = []
    seq = 0
    drained = False

    for event in events:
        kind = event[0]
        if kind == "default":
            executor.set_default(event[1])
        elif kind == "accept":
            _, request_id, choice, x = event
            seq += 1
            try:
                executor.accept(request_id, choice, x, seq=seq)
            except ExecutorError as exc:
                rejected.append(
                    RowResult(
                        # 执行器只为成功受理行分配 seq；拒收行按事件位置补同一序号，
                        # 使合并排序后严格保持输入 id 的先后顺序
                        seq=seq,
                        request_id=request_id,
                        status="rejected",
                        adapter_version=None if choice is None else str(choice),
                        reason=type(exc).__name__,
                        detail={"message": str(exc), "choice": choice},
                    )
                )
        elif kind == "drain":
            report = executor.drain()
            drained = True
        else:
            raise ValueError(f"未知事件类型 {kind!r}")

    if not drained:
        raise ValueError("采集事件缺少 drain，无法交付")

    report.rows.extend(rejected)
    report.rows.sort(key=lambda row: row.seq)
    return report


def report_to_jsonable(report: DrainReport) -> dict[str, Any]:
    """把报告转成可 JSON 序列化的结构，便于离线核对与留档。"""
    import numpy as np

    def row_to_dict(row: RowResult) -> dict:
        data = {
            "seq": row.seq,
            "id": row.request_id,
            "status": row.status,
            "adapter_version": row.adapter_version,
            "reason": row.reason,
            "detail": row.detail,
        }
        if row.status == "ok":
            data.update(
                codes=np.asarray(row.codes).tolist(),
                scale=row.scale,
                downstream=np.asarray(row.downstream).tolist(),
            )
        return data

    return {
        "rows": [row_to_dict(row) for row in report.rows],
        "waves": report.waves,
        "peak_scratch_bytes": report.peak_scratch_bytes,
        "peak_resident_bytes": report.peak_resident_bytes,
    }
