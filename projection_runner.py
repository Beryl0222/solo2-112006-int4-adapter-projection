"""命令行入口：回放交接采集并打印逐行交付与资源口径。

用法：
    python -I -m projection_runner samples/capture.json
"""

import argparse
import json
import sys
from pathlib import Path

# 支持 `python -I projection_runner.py ...`：隔离模式不自动把脚本目录加入路径
sys.path.insert(0, str(Path(__file__).resolve().parent))

from projection_executor.capture import report_to_jsonable, run_capture
from read_capture import read_capture


def main() -> None:
    parser = argparse.ArgumentParser(description="共享 INT4 多适配器投影执行器")
    parser.add_argument("capture", type=Path, help="交接采集 JSON 路径")
    parser.add_argument(
        "--json", action="store_true", help="以 JSON 输出完整报告（含码值与下游值）"
    )
    args = parser.parse_args()

    capture = read_capture(args.capture)
    report = run_capture(capture)

    if args.json:
        print(json.dumps(report_to_jsonable(report), ensure_ascii=False, indent=2))
        return

    for row in report.rows:
        if row.status == "ok":
            codes = ", ".join(f"{v:d}" for v in row.codes)
            downstream = ", ".join(f"{v:.6g}" for v in row.downstream)
            print(
                f"[{row.request_id}] ok 适配器={row.adapter_version} "
                f"scale={row.scale:.8g} 码值=[{codes}] 下游=[{downstream}]"
            )
        elif row.status == "rejected":
            print(f"[{row.request_id}] rejected 原因={row.reason} {row.detail.get('message', '')}")
        else:
            print(
                f"[{row.request_id}] resource 原因={row.reason} "
                f"适配器={row.adapter_version} {row.detail}"
            )

    for wave in report.waves:
        print(
            f"波次{wave['wave']}: 行={wave['request_ids']} "
            f"驻留={wave['resident_versions']}({wave['resident_bytes']}B) "
            f"scratch峰值={wave['scratch_peak_bytes']}B"
        )
    print(
        f"批次峰值: resident={report.peak_resident_bytes}B "
        f"scratch={report.peak_scratch_bytes}B"
    )


if __name__ == "__main__":
    main()
