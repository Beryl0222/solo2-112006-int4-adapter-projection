"""随机化属性测试：更大几何、多分组、K 尾部补位与非零零点全覆盖。"""

from __future__ import annotations

import unittest

import numpy as np

from projection_executor import ProjectionExecutor
from tests.reference import reference_downstream, reference_project

ATOL = RTOL = 2e-5


def make_packet(rng: np.random.Generator, n: int, k: int, group: int) -> dict:
    n_groups = (k + group - 1) // group
    stored_k = n_groups * group
    byte_cols = (stored_k + 1) // 2
    packed = rng.integers(0, 256, size=(n, byte_cols), dtype=np.uint8)
    scale = rng.uniform(0.05, 1.5, size=(n, n_groups)).astype(np.float32)
    zero = rng.integers(0, 16, size=(n, n_groups)).astype(np.int64)
    return {
        "n": n, "k": k, "group": group,
        "packed": packed.tolist(),
        "scale": scale.tolist(),
        "zero": zero.tolist(),
    }


def make_adapter(rng, name, k, n, rank):
    return {
        "gain": float(rng.uniform(0.5, 2.0)),
        "A": rng.normal(size=(rank, k)).astype(np.float32).tolist(),
        "B": rng.normal(size=(n, rank)).astype(np.float32).tolist(),
    }


class RandomizedNumericTests(unittest.TestCase):
    def test_many_geometries_against_reference(self):
        rng = np.random.default_rng(20261120)
        for trial in range(12):
            n = int(rng.integers(1, 7))
            k = int(rng.integers(1, 21))
            group = int(rng.integers(1, 6))
            base = make_packet(rng, n, k, group)
            bias = rng.normal(size=n).tolist()
            clip = [-float(rng.integers(5, 50)), float(rng.integers(5, 50))]
            m = int(rng.integers(1, 4))
            next_matrix = rng.normal(size=(m, n)).tolist()

            adapters = {
                "lo": make_adapter(rng, "lo", k, n, 1),
                "hi": make_adapter(rng, "hi", k, n, int(rng.integers(1, 4))),
            }
            executor = ProjectionExecutor(
                base=base, bias=bias, clip=clip, next_matrix=next_matrix,
                resident_bytes=1 << 22, scratch_bytes=1 << 22,
            )
            for name, packet in adapters.items():
                executor.publish_adapter(name, packet)

            rows_x = [rng.normal(size=k).astype(np.float32) for _ in range(6)]
            choices = [None, "lo", "hi", "lo", None, "hi"]
            for idx, (choice, x) in enumerate(zip(choices, rows_x)):
                executor.accept(f"r{idx}", choice, x)
            report = executor.drain()

            for idx, (choice, x) in enumerate(zip(choices, rows_x)):
                row = report.rows[idx]
                self.assertEqual(row.status, "ok")
                packet = None if choice is None else adapters[choice]
                ref_codes, ref_scale, _ = reference_project(
                    base, packet, x, bias, clip
                )
                np.testing.assert_array_equal(row.codes, ref_codes)
                self.assertAlmostEqual(row.scale, ref_scale, delta=ATOL)
                ref_next = reference_downstream(next_matrix, ref_codes, ref_scale)
                np.testing.assert_allclose(
                    row.downstream, ref_next, atol=ATOL, rtol=RTOL
                )

    def test_streaming_peak_independent_of_k_within_group_budget(self):
        # 工作区峰值只取决于 group 与 n 的流式块，不随 k 增长（底权重不整包解包）
        rng = np.random.default_rng(7)
        small = make_packet(rng, 4, 5, 4)
        large = make_packet(rng, 4, 40, 4)  # k 扩大 8 倍
        common = dict(
            bias=[0.0] * 4, clip=[-10.0, 10.0],
            next_matrix=np.eye(4).tolist(),
            resident_bytes=1 << 20,
        )
        ex_small = ProjectionExecutor(small, scratch_bytes=1 << 20, **common)
        ex_large = ProjectionExecutor(large, scratch_bytes=1 << 20, **common)
        ex_small.accept("s", None, np.ones(5, np.float32))
        ex_large.accept("l", None, np.ones(40, np.float32))
        peak_small = ex_small.drain().peak_scratch_bytes
        peak_large = ex_large.drain().peak_scratch_bytes
        # 底投影部分同 group 同宽；差异只允许来自 LoRA（此处都为 null），故相等
        self.assertEqual(peak_small, peak_large)


if __name__ == "__main__":
    unittest.main()
