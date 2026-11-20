"""投影执行器主体：发布、受理、按资源预算分批 drain。

关键语义（对应交接协议）：

* 每行的适配器选择在 ``accept`` 时固定；随后发布新版本只改变之后受理
  的未指定行，已受理行继续持有旧版本对象，版本替换不影响在途请求。
* 显式引用不存在版本属于受理失败，调用方能拿到具体是哪一行；``null``
  是单独的有效选择（禁用适配器），不走默认版本。
* 驻留预算只计当前已加载适配器的 FP32 系数；同一波放不下就拆波，
  波间卸载适配器。单行都放不下时该行以 ``status="resource"`` 返回，
  绝不交付截断码值，其它行照常执行。
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .errors import (
    BadRequestError,
    DuplicateRequestError,
    ResourceExhausted,
    UnknownAdapterError,
)
from .models import (
    Adapter,
    BaseWeight,
    DrainReport,
    PendingRequest,
    RowResult,
)
from .projection import downstream_from_codes, project_row_fp64, quantize_row
from .workspace import Workspace


class ProjectionExecutor:
    """可复用的投影执行器；线程安全不在本 CPU 交接版本的范围内。"""

    def __init__(
        self,
        base: BaseWeight | dict,
        *,
        bias,
        clip,
        next_matrix,
        resident_bytes: int,
        scratch_bytes: int,
    ):
        self.base = base if isinstance(base, BaseWeight) else BaseWeight.from_packet(base)
        bias_arr = np.asarray(bias, dtype=np.float64)
        if bias_arr.shape != (self.base.n,):
            raise ValueError("bias 形状必须为 [n]")
        if not np.all(np.isfinite(bias_arr)):
            raise ValueError("bias 必须全部有限")
        self.bias = bias_arr
        clip_arr = np.asarray(clip, dtype=np.float64)
        if clip_arr.shape != (2,) or not np.all(np.isfinite(clip_arr)):
            raise ValueError("clip 必须为含两个有限数的闭区间")
        self.clip = (float(clip_arr[0]), float(clip_arr[1]))
        next_arr = np.asarray(next_matrix, dtype=np.float64)
        if next_arr.ndim != 2 or next_arr.shape[1] != self.base.n:
            raise ValueError("next 矩阵形状必须为 [m, n]")
        self.next_matrix = next_arr

        self.resident_budget = int(resident_bytes)
        self.scratch_budget = int(scratch_bytes)
        if self.resident_budget < 0 or self.scratch_budget < 0:
            raise ValueError("资源预算不能为负")

        # 发布包：版本名 -> Adapter。适配器可以先存在于发布包但不驻留。
        self._catalog: dict[str, Adapter] = {}
        self._default_version: Optional[str] = None
        self._pending: list[PendingRequest] = []
        self._seq = 0

    # ------------------------------------------------------------------ 发布
    def publish_adapter(self, version: str, packet: dict) -> Adapter:
        """发布（或替换）一个适配器版本；替换不影响已受理行。"""
        adapter = Adapter.from_packet(
            name=version, version=version, packet=packet,
            k=self.base.k, n=self.base.n,
        )
        # 仅当有在途行仍引用旧对象时才需要保留；字典替换后旧对象
        # 仍被 PendingRequest 持有，在途行的绑定不受影响。
        self._catalog[version] = adapter
        return adapter

    def set_default(self, version: Optional[str]) -> None:
        """设置当前默认版本；只影响之后受理的未指定行。传 None 取消默认。"""
        if version is not None and version not in self._catalog:
            raise UnknownAdapterError(request_id="<default>", version=version)
        self._default_version = version

    @property
    def default_version(self) -> Optional[str]:
        return self._default_version

    @property
    def published_versions(self) -> tuple[str, ...]:
        return tuple(self._catalog)

    # ------------------------------------------------------------------ 受理
    def accept(self, request_id: str, adapter_choice, x, *, seq: int | None = None) -> PendingRequest:
        """受理一行。``adapter_choice`` 为 ``"default"``、版本名或 ``None``。

        选择在此刻解析并固定为具体适配器对象（或 None）；未知版本抛
        :class:`UnknownAdapterError`，该行不会进入在途队列。``seq``
        可由外部事件源显式指定，便于与未受理行共用同一顺序编号。
        """
        if not isinstance(request_id, str) or not request_id:
            raise BadRequestError("请求 id 必须为非空字符串")
        if any(p.request_id == request_id for p in self._pending):
            raise DuplicateRequestError(request_id)

        x_arr = np.asarray(x, dtype=np.float32)
        if x_arr.shape != (self.base.k,) or not np.all(np.isfinite(x_arr)):
            raise BadRequestError(
                f"请求 {request_id!r} 的输入必须为长度 k={self.base.k} 的有限 FP32 向量"
            )

        if adapter_choice is None:
            adapter: Optional[Adapter] = None
        elif adapter_choice == "default":
            if self._default_version is None:
                raise BadRequestError(
                    f"请求 {request_id!r} 要求默认适配器，但当前没有默认版本"
                )
            adapter = self._catalog[self._default_version]
        else:
            if adapter_choice not in self._catalog:
                # 明确告知调用方哪一行未受理
                raise UnknownAdapterError(request_id, adapter_choice)
            adapter = self._catalog[adapter_choice]

        if seq is None:
            self._seq += 1
            seq = self._seq
        else:
            self._seq = max(self._seq, seq)
        pending = PendingRequest(
            seq=seq, request_id=request_id, adapter=adapter, x=x_arr
        )
        self._pending.append(pending)
        return pending

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    # ------------------------------------------------------------- 波次规划
    def _plan_waves(self, requests: list[PendingRequest]) -> list[list[PendingRequest]]:
        """按受理顺序分波：同波驻留的适配器系数合计不超驻留预算。

        单个适配器自身超过预算时，规划层就把它单独成波并标记为不可驻留，
        drain 时该行直接得到资源原因，而不会假装它能与别的行同波运行。
        """
        waves: list[list[PendingRequest]] = []
        current: list[PendingRequest] = []
        loaded: dict[str, Adapter] = {}

        def resident_now() -> int:
            return sum(a.resident_bytes for a in loaded.values())

        for req in requests:
            if req.adapter is None or req.adapter.version in loaded:
                current.append(req)
                continue
            adapter = req.adapter
            if adapter.resident_bytes > self.resident_budget:
                # 单行都无法驻留：先封当前波，再把不可行行单独成波
                if current:
                    waves.append(current)
                    current, loaded = [], {}
                waves.append([req])
                continue
            if resident_now() + adapter.resident_bytes <= self.resident_budget:
                loaded[adapter.version] = adapter
                current.append(req)
            else:
                # 当前波装不下：封波并开新波（新波先清空驻留）
                waves.append(current)
                current = [req]
                loaded = {adapter.version: adapter}
        if current:
            waves.append(current)
        return waves

    # ------------------------------------------------------------------ drain
    def drain(
        self,
        *,
        resident_bytes: Optional[int] = None,
        scratch_bytes: Optional[int] = None,
    ) -> DrainReport:
        """执行全部在途行，按受理顺序返回逐行结果，并清空在途队列。"""
        resident_budget = (
            self.resident_budget if resident_bytes is None else int(resident_bytes)
        )
        scratch_budget = (
            self.scratch_budget if scratch_bytes is None else int(scratch_bytes)
        )
        if resident_budget < 0 or scratch_budget < 0:
            raise ValueError("资源预算不能为负")

        requests = self._pending
        self._pending = []
        # 本批次使用临时预算值，不修改执行器的长期默认预算
        waves_plan = self._replan_with_budget(requests, resident_budget)

        results: list[RowResult] = []
        wave_logs: list[dict] = []
        peak_scratch = 0
        peak_resident = 0

        for wave_index, wave_requests in enumerate(waves_plan, start=1):
            workspace = Workspace(scratch_budget)
            loaded: dict[str, Adapter] = {}
            wave_rows: list[RowResult] = []
            failed_in_wave: list[str] = []

            for req in wave_requests:
                # 驻留准入：单行所需适配器无法驻留即资源失败，不产码值
                if req.adapter is not None and req.adapter.version not in loaded:
                    if req.adapter.resident_bytes > resident_budget:
                        failed_in_wave.append(req.request_id)
                        wave_rows.append(
                            RowResult(
                                seq=req.seq,
                                request_id=req.request_id,
                                status="resource",
                                adapter_version=req.adapter.version,
                                reason="adapter_too_large",
                                detail={
                                    "needed_resident_bytes": req.adapter.resident_bytes,
                                    "resident_budget": resident_budget,
                                },
                            )
                        )
                        continue
                    loaded[req.adapter.version] = req.adapter
                    peak_resident = max(
                        peak_resident,
                        sum(a.resident_bytes for a in loaded.values()),
                    )

                row = self._run_one(req, workspace)
                if row.status == "ok":
                    peak_scratch = max(peak_scratch, workspace.peak)
                else:
                    # 失败后归还该行可能申请的全部临时缓冲，保证下一行干净起步
                    failed_in_wave.append(req.request_id)
                    workspace.reset()
                wave_rows.append(row)

            wave_logs.append(
                {
                    "wave": wave_index,
                    "request_ids": [r.request_id for r in wave_requests],
                    "resident_versions": sorted(loaded),
                    "resident_bytes": sum(a.resident_bytes for a in loaded.values()),
                    "scratch_peak_bytes": workspace.peak,
                    "resource_failed": failed_in_wave,
                }
            )
            results.extend(wave_rows)

        results.sort(key=lambda row: row.seq)
        return DrainReport(
            rows=results,
            waves=wave_logs,
            peak_scratch_bytes=peak_scratch,
            peak_resident_bytes=peak_resident,
        )

    def _replan_with_budget(
        self, requests: list[PendingRequest], resident_budget: int
    ) -> list[list[PendingRequest]]:
        saved = self.resident_budget
        self.resident_budget = resident_budget
        try:
            return self._plan_waves(requests)
        finally:
            self.resident_budget = saved

    def _run_one(self, req: PendingRequest, workspace: Workspace) -> RowResult:
        version = None if req.adapter is None else req.adapter.version
        try:
            real = np.zeros((self.base.n,), dtype=np.float64)  # 逐行持久结果
            project_row_fp64(self.base, req.adapter, req.x, workspace, real)
            codes, row_scale = quantize_row(
                real, self.bias, self.clip, workspace
            )
            downstream = downstream_from_codes(
                self.next_matrix, codes, row_scale, workspace
            )
        except ResourceExhausted as exc:
            # 只有资源口径走逐行失败；数值/协议类错误继续向上抛，不伪装成资源问题
            return RowResult(
                seq=req.seq,
                request_id=req.request_id,
                status="resource",
                adapter_version=version,
                reason=exc.kind or "resource",
                detail={
                    "requested": exc.requested,
                    "budget": exc.budget,
                    "message": str(exc),
                },
            )
        return RowResult(
            seq=req.seq,
            request_id=req.request_id,
            status="ok",
            adapter_version=version,
            codes=codes,
            scale=row_scale,
            downstream=downstream,
        )
