本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。
本文件只用于 Lab research，不是 Stable 交易规则，不接 QMT，不生成 OrderIntent，不自动下单。

# AETF Q3 Lab Intraday Mock Tensor Smoke

## 任务定位

F-1 是 intraday / 5分钟K / PyTorch 执行研究的 mock tensor smoke。它基于 F-0 数据合同、状态机方案和 PyTorch 执行模型方案，只验证工程路径，不读取真实 intraday 数据，不训练正式模型，不解释模型效果。

## 验证内容

- 读取人工构造的 mock 5分钟K CSV fixture。
- 构造 sequence tensor，shape 为 `[batch, time_steps, features]`。
- 执行 forbidden feature scan。
- 运行 MLP / GRU / temporal CNN 的 2-step forward/backward smoke。
- 运行 Intraday Watch Engine 状态机 dry-run skeleton。

## 边界

- 不是训练，不做 formal model training。
- 不读取真实 5分钟K / tick / 盘口数据。
- 不运行 `torchrun`。
- 不接 QMT。
- 不生成 `OrderIntent`。
- 不自动下单。
- 不写 Stable `output/` 或 `runtime/`。
- 不创建 `lab_advisory/`，不生成 advisory 包。
- 不保存模型，不生成 checkpoint。
- `final_loss` 只代表 smoke 路径可运行，不能解释为模型效果或交易建议。

## 命令

Lab 命令必须使用仓库 `.venv` Python：

```powershell
.\.venv\Scripts\python.exe tools\lab\intraday_mock_tensor_smoke.py `
  --input tests\fixtures\aetfq3_lab\mock_intraday_5m_samples.csv `
  --out-dir .local_research_outputs\aetfq3_lab\intraday_mock_tensor_smoke\
```

```powershell
.\.venv\Scripts\python.exe tools\lab\intraday_watch_state_machine_dryrun.py `
  --events tests\fixtures\aetfq3_lab\mock_intraday_watch_events.json `
  --out-dir .local_research_outputs\aetfq3_lab\intraday_mock_tensor_smoke\
```

## 输出

仅允许写 ignored 本地目录：

```text
.local_research_outputs/aetfq3_lab/intraday_mock_tensor_smoke/
```

允许生成：

- `intraday_mock_tensor_smoke_report.md`
- `intraday_mock_tensor_smoke_report.json`
- `intraday_watch_dryrun_report.md`
- `intraday_watch_dryrun_report.json`

这些文件不是 Stable 输入，不是 advisory 包，不得进入正式交易流程。

