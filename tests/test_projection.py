"""投影执行器的完整交接测试。

数值口径用 tests/reference.py 的独立 FP64 实现交叉核对；
资源口径直接断言工作区峰值与协议预算；失败边界逐行检查。
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from projection_executor import (
    BadRequestError,
    BaseWeight,
    DuplicateRequestError,
    ProjectionExecutor,
    ResourceExhausted,
    UnknownAdapterError,
    run_capture,
)
from projection_executor.capture import report_to_jsonable
from projection_executor.workspace import Workspace
from read_capture import read_capture

from tests.reference import (
    reference_base_matrix,
    reference_downstream,
    reference_project,
)

ROOT = Path(__file__).resolve().parents[1]
CAPTURE = ROOT / "samples" / "capture.json"
ATOL = RTOL = 2e-5


def load_capture() -> dict:
    return read_capture(CAPTURE)


def make_executor(capture: dict, **overrides) -> ProjectionExecutor:
    kwargs = dict(
        base=capture["base"],
        bias=capture["bias"],
        clip=capture["clip"],
        next_matrix=capture["next"],
        resident_bytes=capture["resident_bytes"],
        scratch_bytes=capture["scratch_bytes"],
    )
    kwargs.update(overrides)
    return ProjectionExecutor(**kwargs)


class BaseDecodingTests(unittest.TestCase):
    def test_grouped_nibble_layout_and_nonzero_zero(self):
        capture = load_capture()
        base = BaseWeight.from_packet(capture["base"])
        matrix = reference_base_matrix(capture["base"])
        for ch in range(base.n):
            for ki in range(base.k):
                self.assertEqual(
                    base.decode_element(ch, ki), matrix[ch, ki],
                    f"解码不一致 ch={ch} k={ki}",
                )
        # 显式抽查：低 nibble 在前、组 1 零点为 3/5/7
        self.assertEqual(base.code_at(0, 0), 33 & 0x0F)
        self.assertEqual(base.code_at(0, 1), (33 >> 4) & 0x0F)
        self.assertEqual(base.zero[0, 1], 3)
        # 补位位置（k=5，group=4 -> 存储 8 列元素）也必须可按组解码且不越界
        self.assertEqual(base.code_at(2, 7), (238 >> 4) & 0x0F)

    def test_packed_shape_validated_against_protocol_layout(self):
        capture = load_capture()
        bad = json.loads(json.dumps(capture["base"]))
        bad["packed"] = bad["packed"][:-1]
        with self.assertRaises(ValueError):
            BaseWeight.from_packet(bad)


class NumericPathTests(unittest.TestCase):
    def setUp(self):
        self.capture = load_capture()

    def _check_row_against_reference(self, row, adapter_packet, x):
        self.assertEqual(row.status, "ok")
        ref_codes, ref_scale, _ = reference_project(
            self.capture["base"], adapter_packet, x,
            self.capture["bias"], self.capture["clip"],
        )
        np.testing.assert_array_equal(np.asarray(row.codes), ref_codes)
        self.assertAlmostEqual(row.scale, ref_scale, delta=ATOL)
        ref_next = reference_downstream(self.capture["next"], ref_codes, ref_scale)
        np.testing.assert_allclose(row.downstream, ref_next, atol=ATOL, rtol=RTOL)

    def test_all_capture_rows_match_independent_fp64_reference(self):
        report = run_capture(load_capture())
        events = [e for e in self.capture["events"] if e[0] == "accept"]
        by_id = {row.request_id: row for row in report.rows}
        for _, request_id, choice, x in events:
            row = by_id[request_id]
            if choice == "r9":
                self.assertEqual(row.status, "rejected")
                continue
            if choice is None:
                packet = None
            elif choice == "default":
                # 受理时的默认版本按事件顺序确定
                packet = None
            else:
                packet = self.capture["adapters"][choice]
            # 默认版本在回放中由事件决定，这里按报告记录回查
            if choice == "default":
                version = row.adapter_version
                packet = self.capture["adapters"][version]
            self._check_row_against_reference(row, packet, x)

    def test_order_preserved_and_every_row_traceable(self):
        report = run_capture(load_capture())
        accept_ids = [e[1] for e in self.capture["events"] if e[0] == "accept"]
        self.assertEqual([r.request_id for r in report.rows], accept_ids)
        for row in report.rows:
            if row.status == "ok":
                self.assertIsNotNone(row.codes)
                self.assertIsNotNone(row.scale)
                self.assertEqual(len(row.downstream), len(self.capture["next"]))

    def test_half_even_rounding_boundary(self):
        # 直接对量化函数注入精确实值，验证协议的半整数向偶数舍入。
        from projection_executor.projection import quantize_row

        ws = Workspace(1 << 20)
        # 单通道 v=127 -> s=1，商为整数 127，不触舍入，作为锚点
        real = np.array([127.0], dtype=np.float64)
        codes, s = quantize_row(real, np.zeros(1), (-1000.0, 1000.0), ws)
        self.assertEqual(int(codes[0]), 127)
        self.assertEqual(s, 1.0)

        # 两通道 [63.5, 127]：s=1，63.5 必须向偶数 64（而非 63）
        real = np.array([63.5, 127.0], dtype=np.float64)
        codes, _ = quantize_row(real, np.zeros(2), (-1000.0, 1000.0), ws)
        np.testing.assert_array_equal(codes, np.array([64, 127], dtype=np.int8))

        # [64.5, 127]：64.5 向偶数 64（而非四舍五入的 65）
        real = np.array([64.5, 127.0], dtype=np.float64)
        codes, _ = quantize_row(real, np.zeros(2), (-1000.0, 1000.0), ws)
        np.testing.assert_array_equal(codes, np.array([64, 127], dtype=np.int8))
        self.assertEqual(int(np.rint(63.5)), 64)
        self.assertEqual(int(np.rint(64.5)), 64)

    def test_zero_row_scale_is_one(self):
        capture = self.capture
        executor = make_executor(capture, bias=[0.0, 0.0, 0.0])
        executor.publish_adapter("r0", capture["adapters"]["r0"])
        executor.set_default("r0")
        executor.accept("zero", None, [0.0] * capture["base"]["k"])
        row = executor.drain().rows[0]
        self.assertEqual(row.status, "ok")
        self.assertEqual(row.scale, 1.0)
        np.testing.assert_array_equal(row.codes, np.zeros(capture["base"]["n"], np.int8))
        np.testing.assert_allclose(row.downstream, np.zeros(len(capture["next"])), atol=0)

    def test_quantization_happens_after_full_real_sum(self):
        # 适配器放大通道 3（B 中 20 倍），先量化底投影再补修正会丢失范围；
        # 执行器必须先实值求和，因此行 b 的码值达到 127 饱和而非被底投影限幅
        report = run_capture(load_capture())
        row_b = report.by_id("b")
        self.assertEqual(int(np.max(np.abs(row_b.codes))), 127)
        ref_codes, ref_scale, _ = reference_project(
            self.capture["base"], self.capture["adapters"]["r1"],
            [1, 0, 2, 1, 5], self.capture["bias"], self.capture["clip"],
        )
        np.testing.assert_array_equal(row_b.codes, ref_codes)
        self.assertAlmostEqual(row_b.scale, ref_scale, delta=ATOL)

    def test_pre_quantized_base_path_changes_delivered_codes(self):
        # 底投影 [0.05, 0.025]、修正 [0.3, 0.23]：先量化底投影会使第二通道
        # 码档翻转（93 -> 92）。执行器必须与“先实值求和”的参考一致。
        base = {
            "n": 2, "k": 2, "group": 2,
            # nibble 1 -> 码值 1，零点 0，标度 0.025
            "packed": [[0x11], [0x01]],
            "scale": [[0.025], [0.025]],
            "zero": [[0], [0]],
        }
        adapter = {
            "gain": 1.0,
            "A": [[1.0, 0.0]],
            "B": [[0.3], [0.23]],
        }
        executor = ProjectionExecutor(
            base=base, bias=[0.0, 0.0], clip=[-100, 100],
            next_matrix=np.eye(2).tolist(),
            resident_bytes=1 << 20, scratch_bytes=1 << 20,
        )
        executor.publish_adapter("a", adapter)
        executor.accept("r", "a", [1.0, 1.0])
        row = executor.drain().rows[0]

        base_real = np.array([0.05, 0.025])
        lora = np.array([0.3, 0.23])
        correct = base_real + lora
        scale_correct = np.max(np.abs(correct)) / 127.0
        expected = np.clip(np.rint(correct / scale_correct), -127, 127).astype(np.int8)
        np.testing.assert_array_equal(row.codes, expected)  # [127, 93]

        # 对照：naive 两阶段路径确实给出不同码值 [127, 92]
        s0 = np.max(np.abs(base_real)) / 127.0
        naive_real = np.rint(base_real / s0) * s0 + lora
        s_naive = np.max(np.abs(naive_real)) / 127.0
        naive_codes = np.clip(
            np.rint(naive_real / s_naive), -127, 127
        ).astype(np.int8)
        self.assertFalse(np.array_equal(naive_codes, expected))


class AdapterLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.capture = load_capture()

    def test_explicit_unknown_version_reports_which_row(self):
        executor = make_executor(self.capture)
        executor.publish_adapter("r0", self.capture["adapters"]["r0"])
        with self.assertRaises(UnknownAdapterError) as ctx:
            executor.accept("d", "r9", [0, 0, 0, 0, 0])
        self.assertEqual(ctx.exception.request_id, "d")
        self.assertEqual(ctx.exception.version, "r9")
        self.assertEqual(executor.pending_count, 0)

    def test_null_is_valid_and_uses_no_adapter(self):
        executor = make_executor(self.capture)
        executor.publish_adapter("r0", self.capture["adapters"]["r0"])
        executor.set_default("r0")
        executor.accept("e", None, [0.0] * 5)
        row = executor.drain().rows[0]
        self.assertEqual(row.adapter_version, None)
        ref_codes, ref_scale, _ = reference_project(
            self.capture["base"], None, [0.0] * 5,
            self.capture["bias"], self.capture["clip"],
        )
        np.testing.assert_array_equal(row.codes, ref_codes)

    def test_default_change_does_not_affect_accepted_rows(self):
        executor = make_executor(self.capture)
        executor.publish_adapter("r0", self.capture["adapters"]["r0"])
        executor.publish_adapter("r1", self.capture["adapters"]["r1"])
        executor.set_default("r0")
        executor.accept("a", "default", [1, 2, 3, -4, 5])
        # 发布新默认版本；已受理的 a 必须继续绑定 r0
        executor.set_default("r1")
        executor.accept("b", "default", [1, 0, 2, 1, 5])
        report = executor.drain()
        self.assertEqual(report.rows[0].adapter_version, "r0")
        self.assertEqual(report.rows[1].adapter_version, "r1")

    def test_republished_version_keeps_inflight_rows_on_old_object(self):
        executor = make_executor(self.capture)
        executor.publish_adapter("r0", self.capture["adapters"]["r0"])
        executor.set_default("r0")
        executor.accept("a", "default", [1, 2, 3, -4, 5])
        old = executor._catalog["r0"]
        # 同名发布包内容替换
        replacement = json.loads(json.dumps(self.capture["adapters"]["r1"]))
        executor.publish_adapter("r0", replacement)
        self.assertIsNot(executor._catalog["r0"], old)
        report = executor.drain()
        # 在途行仍用旧对象数值（与原始 r0 参考一致）
        ref_codes, _, _ = reference_project(
            self.capture["base"], self.capture["adapters"]["r0"],
            [1, 2, 3, -4, 5], self.capture["bias"], self.capture["clip"],
        )
        np.testing.assert_array_equal(report.rows[0].codes, ref_codes)

    def test_same_rank_versions_never_share_corrections(self):
        executor = make_executor(self.capture)
        p0 = json.loads(json.dumps(self.capture["adapters"]["r0"]))
        p1 = json.loads(json.dumps(self.capture["adapters"]["r0"]))
        # 同秩但内容不同的两个版本
        p1["B"][0][0] = 5
        executor.publish_adapter("v0", p0)
        executor.publish_adapter("v1", p1)
        x = [1, 1, 1, 1, 1]
        executor.accept("one", "v0", x)
        executor.accept("two", "v1", x)
        report = executor.drain()
        self.assertFalse(np.array_equal(report.rows[0].codes, report.rows[1].codes))
        ref_codes_1, _, _ = reference_project(
            self.capture["base"], p1, x, self.capture["bias"], self.capture["clip"]
        )
        np.testing.assert_array_equal(report.rows[1].codes, ref_codes_1)

    def test_duplicate_id_and_bad_input_rejected_at_accept(self):
        executor = make_executor(self.capture)
        executor.publish_adapter("r0", self.capture["adapters"]["r0"])
        executor.accept("a", None, [0.0] * 5)
        with self.assertRaises(DuplicateRequestError):
            executor.accept("a", None, [0.0] * 5)
        with self.assertRaises(BadRequestError):
            executor.accept("b", None, [0.0] * 4)
        with self.assertRaises(BadRequestError):
            executor.accept("c", None, [float("nan")] + [0.0] * 4)
        self.assertEqual(executor.pending_count, 1)


class ResourceAccountingTests(unittest.TestCase):
    def setUp(self):
        self.capture = load_capture()

    def test_workspace_tracks_peak_and_rejects_overbudget(self):
        ws = Workspace(64)
        buf = ws.allocate("a", (4,), np.float64)
        self.assertEqual(ws.used, 32)
        with self.assertRaises(ResourceExhausted) as ctx:
            ws.allocate("b", (5,), np.float64)  # 再 40B -> 72 > 64
        self.assertEqual(ctx.exception.kind, "scratch")
        self.assertEqual(ws.used, 32)  # 失败不改变现有分配
        self.assertEqual(ws.peak, 32)
        with ws.lease("c", (4,), np.float64):
            self.assertEqual(ws.used, 64)
        self.assertEqual(ws.used, 32)
        self.assertEqual(buf.shape, (4,))

    def test_capture_resident_budget_forces_multiple_waves(self):
        report = run_capture(load_capture())
        # r0=32B、r1=64B，预算 64B，两者不能同波 -> 至少 3 波
        self.assertGreaterEqual(len(report.waves), 3)
        for wave in report.waves:
            self.assertLessEqual(wave["resident_bytes"], self.capture["resident_bytes"])
            self.assertLessEqual(
                wave["scratch_peak_bytes"], self.capture["scratch_bytes"]
            )
        self.assertLessEqual(
            report.peak_resident_bytes, self.capture["resident_bytes"]
        )
        self.assertLessEqual(report.peak_scratch_bytes, self.capture["scratch_bytes"])

    def test_base_weights_never_fully_unpacked_into_scratch(self):
        # 完整解包成 fp32 需要 n*k*4 = 60B；给一个低于该值、但够流式的预算
        executor = make_executor(self.capture, scratch_bytes=48)
        executor.accept("e", None, [0.0] * 5)
        report = executor.drain()
        # 流式底投影峰值为 g(32)+xg(32)=64B，48B 应明确失败而非静默截断
        self.assertEqual(report.rows[0].status, "resource")
        self.assertEqual(report.rows[0].reason, "scratch")
        self.assertIsNone(report.rows[0].codes)

    def test_single_oversized_adapter_gives_resource_row(self):
        capture = self.capture
        # 驻留预算 32B：r0 可跑（32B），r1 需 64B -> 单行资源失败
        executor = make_executor(capture, resident_bytes=32)
        executor.publish_adapter("r0", capture["adapters"]["r0"])
        executor.publish_adapter("r1", capture["adapters"]["r1"])
        executor.accept("ok-row", "r0", [1, 1, 1, 1, 1])
        executor.accept("big-row", "r1", [1, 0, 2, 1, 5])
        report = executor.drain()
        statuses = {r.request_id: r.status for r in report.rows}
        self.assertEqual(statuses["ok-row"], "ok")
        self.assertEqual(statuses["big-row"], "resource")
        big = report.by_id("big-row")
        self.assertEqual(big.reason, "adapter_too_large")
        self.assertIsNone(big.codes)
        # 顺序不被失败打乱
        self.assertEqual([r.request_id for r in report.rows], ["ok-row", "big-row"])

    def test_tight_scratch_budget_still_runs_rows_that_fit(self):
        capture = self.capture
        # 96B 正好是执行器实测峰值，应全部成功；95B 则全部资源失败
        executor = make_executor(capture, scratch_bytes=96)
        for name, packet in capture["adapters"].items():
            executor.publish_adapter(name, packet)
        executor.set_default("r0")
        executor.accept("a", "default", [1, 2, 3, -4, 5])
        report = executor.drain()
        self.assertEqual(report.rows[0].status, "ok")

        executor.set_default("r0")
        executor.accept("a2", "default", [1, 2, 3, -4, 5])
        tight = executor.drain(scratch_bytes=95)
        self.assertEqual(tight.rows[0].status, "resource")
        self.assertIsNone(tight.rows[0].codes)

    def test_accepted_rows_remain_bound_when_budget_shrinks_at_drain(self):
        capture = self.capture
        executor = make_executor(capture, resident_bytes=capture["resident_bytes"])
        executor.publish_adapter("r0", capture["adapters"]["r0"])
        executor.publish_adapter("r1", capture["adapters"]["r1"])
        executor.accept("a", "r0", [1, 2, 3, -4, 5])
        executor.accept("b", "r1", [1, 0, 2, 1, 5])
        # drain 时驻留预算收紧到 0：行不丢失、不换版本，逐行给资源原因
        report = executor.drain(resident_bytes=0)
        self.assertEqual([r.request_id for r in report.rows], ["a", "b"])
        self.assertTrue(all(r.status == "resource" for r in report.rows))
        self.assertEqual(
            [r.adapter_version for r in report.rows], ["r0", "r1"]
        )

    def test_oversized_adapter_isolated_in_own_wave_others_still_deliver(self):
        # 预算 32B：r0(32B) 与 null 行可跑，r1(64B) 被单独成波并资源失败，
        # 已绑定版本不丢失、顺序保持
        capture = self.capture
        executor = make_executor(capture, resident_bytes=32)
        executor.publish_adapter("r0", capture["adapters"]["r0"])
        executor.publish_adapter("r1", capture["adapters"]["r1"])
        executor.accept("p", "r0", [1, 1, 1, 1, 1])
        executor.accept("q", "r1", [1, 0, 2, 1, 5])
        executor.accept("z", None, [0.0] * 5)
        report = executor.drain()
        self.assertEqual([r.request_id for r in report.rows], ["p", "q", "z"])
        self.assertEqual(
            [r.status for r in report.rows], ["ok", "resource", "ok"]
        )
        self.assertEqual(report.by_id("q").reason, "adapter_too_large")
        self.assertIsNone(report.by_id("q").codes)
        # 三个波：p 单独波、q 不可行单独波、z 单独波
        self.assertEqual(
            [w["request_ids"] for w in report.waves],
            [["p"], ["q"], ["z"]],
        )
        self.assertEqual(report.waves[1]["resource_failed"], ["q"])

    def test_resource_limits_apply_to_fallback_path_too(self):
        # null（无适配器，回退路径）同样受 scratch 预算约束
        capture = self.capture
        executor = make_executor(capture, scratch_bytes=8)
        executor.accept("e", None, [0.0] * 5)
        report = executor.drain()
        self.assertEqual(report.rows[0].status, "resource")

    def test_mixed_batch_isolates_single_scratch_failure(self):
        # 预算只够 null 路径（底投影+量化+下游），不够 LoRA 路径：
        # 同一批中适配器行资源失败、null 行仍交付，顺序保持、无截断码值流出
        capture = self.capture
        # LoRA 路径峰值 96B；null 路径峰值 = 底投影64 + 量化24 + 下游40
        # 取预算 88B：LoRA 在第 5 块临时张量处超限，null 全流程可完成
        executor = make_executor(capture, scratch_bytes=88)
        executor.publish_adapter("r0", capture["adapters"]["r0"])
        executor.accept("x", "r0", [1, 1, 1, 1, 1])
        executor.accept("y", None, [0.0] * 5)
        report = executor.drain()
        self.assertEqual([r.request_id for r in report.rows], ["x", "y"])
        self.assertEqual(report.rows[0].status, "resource")
        self.assertIsNone(report.rows[0].codes)
        self.assertEqual(report.rows[1].status, "ok")
        self.assertIsNotNone(report.rows[1].codes)

    def test_downstream_consumes_delivered_codes_not_hidden_values(self):
        # 手工构造：让量化发生明显舍入，验证下游严格按交付码值*行比例计算
        capture = self.capture
        executor = make_executor(capture)
        executor.accept("e", None, [0.0] * 5)
        row = executor.drain().rows[0]
        manual = (
            np.asarray(capture["next"], dtype=np.float64)
            @ (row.codes.astype(np.float64) * row.scale)
        )
        np.testing.assert_allclose(row.downstream, manual, atol=0, rtol=0)


class CaptureReplayTests(unittest.TestCase):
    def test_full_capture_report_matches_event_semantics(self):
        report = run_capture(load_capture())
        ids = [r.request_id for r in report.rows]
        self.assertEqual(ids, ["a", "b", "c", "d", "e"])
        versions = {r.request_id: r.adapter_version for r in report.rows if r.status == "ok"}
        # a 在 default=r0 时受理；默认切 r1 后 b 用 r1；c 显式 r0；e 为 null
        self.assertEqual(versions["a"], "r0")
        self.assertEqual(versions["b"], "r1")
        self.assertEqual(versions["c"], "r0")
        self.assertIsNone(versions["e"])
        rejected = report.by_id("d")
        self.assertEqual(rejected.status, "rejected")
        self.assertEqual(rejected.reason, "UnknownAdapterError")
        payload = report_to_jsonable(report)
        json.dumps(payload, ensure_ascii=False)  # 可留档序列化

    def test_cli_runs_in_isolated_mode(self):
        result = subprocess.run(
            [sys.executable, "-I", str(ROOT / "projection_runner.py"), str(CAPTURE)],
            text=True, capture_output=True, check=True,
        )
        lines = result.stdout.splitlines()
        self.assertIn("[a] ok", lines[0])
        self.assertTrue(any("resource" not in line and "rejected" in line for line in lines))
        self.assertTrue(any(line.startswith("批次峰值") for line in lines))

    def test_capture_without_drain_is_an_error(self):
        capture = load_capture()
        capture["events"] = [e for e in capture["events"] if e[0] != "drain"]
        with self.assertRaisesRegex(ValueError, "drain"):
            run_capture(capture)

    def test_untrusted_capture_in_own_directory_with_isolated_python(self):        # 模拟交付包放在独立空目录；执行器以 -I 运行， planted json.py 不会被导入
        with tempfile.TemporaryDirectory() as raw_dir, tempfile.TemporaryDirectory() as out_dir:
            raw = Path(raw_dir) / "capture.json"
            raw.write_text(Path(CAPTURE).read_text(encoding="utf-8"), encoding="utf-8")
            (Path(raw_dir) / "json.py").write_text("raise RuntimeError('planted')")
            script = Path(out_dir) / "check.py"
            script.write_text(
                "import sys; sys.path.insert(0, %r);\n"
                "from projection_executor import run_capture;\n"
                "from read_capture import read_capture;\n"
                "report = run_capture(read_capture(sys.argv[1]));\n"
                "print(len(report.delivered))\n" % str(ROOT),
                encoding="utf-8",
            )
            result = subprocess.run(
                [sys.executable, "-I", str(script), str(raw)],
                text=True, capture_output=True, check=True,
            )
            self.assertEqual(result.stdout.strip(), "4")  # a/b/c/e 交付，d 拒收


if __name__ == "__main__":
    unittest.main()
