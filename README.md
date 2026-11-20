# 共享 INT4 投影的多适配器批执行

同一份 INT4 底权重在设备上只驻留一份，多款产品经各自的低秩适配器改变投影
行为。本项目提供可在 CPU 上交接验证的 Python 投影执行器，数值与资源口径
均以 [交接协议](交接协议.md) 为准。

`samples/capture.json` 是离线合成的交接采集，包含底权重包、两个适配器
发布包、逐行推理请求与接收顺序。

## 口径要点

- **解码不猜布局**：每输出通道一行打包字节，低 nibble 在前，按 K 轴
  `group` 分组；标度/零点形状 `[n, ceil(k/group)]`，零点可非零，K 尾部
  补位只对齐存储、不参与投影。
- **选择在受理时固定**：`accept` 即把请求行绑定到具体适配器版本对象；
  之后发布/替换默认版本不影响在途行。显式引用不存在版本会抛出
  `UnknownAdapterError` 并标明哪一行未受理；`null` 是显式禁用适配器的
  有效选择。即使两个版本秩相同也不共享修正结果。
- **先实值后量化**：底投影与 `gain * B @ (A @ x)` 在 FP64 上求和，再做
  偏置、闭区间截断和逐行 INT8 量化（比例 `max|v|/127`、零行比例 1、
  半整数向偶数舍入）。下游矩阵消费的是交付码值按行比例恢复的实值。
- **资源口径**：驻留量只计活跃适配器的 FP32 系数；scratch 计投影过程中
  临时张量的同时峰值，持久输入与最终交付值不计。底权重按分组流式解码、
  从不整体解包进临时区。预算不足时单行以 `status="resource"` 返回原因，
  不产出截断码值，其它行照常交付；每行可回查实际版本、量化参数与下游值。

## 环境与命令

环境使用 Python 3.11 和锁定的 NumPy。

```bash
python -m pip install -r requirements.txt
python -m read_capture samples/capture.json          # 查看事件顺序
python -m unittest discover -s tests -v              # 常规模式跑测试
```

处理不受信任的交付包时，把包放在独立空目录并用隔离模式运行（ planted
`json.py` 等不会进入导入路径）：

```bash
python -I projection_runner.py samples/capture.json           # 文本摘要
python -I projection_runner.py samples/capture.json --json    # 完整 JSON 报告
python -I -m unittest discover -s tests -t .                  # 隔离模式测试
```

## 可复用调用方式

```python
from projection_executor import ProjectionExecutor, UnknownAdapterError

executor = ProjectionExecutor(
    base_packet, bias=bias, clip=clip, next_matrix=next_w,
    resident_bytes=64, scratch_bytes=104,
)
executor.publish_adapter("r0", adapter_packet)   # 发布/替换版本
executor.set_default("r0")                        # 默认只影响之后受理的行

try:
    executor.accept("a", "default", x_fp32)      # 选择此刻固定
    executor.accept("b", None, x_fp32)           # null = 显式禁用适配器
except UnknownAdapterError as exc:
    ...                                          # exc.request_id 指明未受理行

report = executor.drain()                        # 也可临时传入预算覆盖
for row in report.rows:                          # 顺序与受理顺序一致
    row.status            # ok / rejected / resource
    row.adapter_version   # 实际使用的版本（null 行为 None）
    row.codes, row.scale  # 实际交付的 INT8 码值与行比例
    row.downstream        # next @ (codes * scale)
report.waves              # 每波驻留版本、驻留字节、scratch 峰值
```

回放整份事件采集用 `projection_executor.run_capture(capture)`，
`report_to_jsonable(report)` 可序列化为 JSON 留档。
