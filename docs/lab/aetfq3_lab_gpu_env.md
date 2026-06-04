# AETF Q3 Lab GPU Environment

## 文档定位

本文件记录 `aetfq3-lab / Lab` 当前台式机 GPU worker 的 Python、PyTorch、CUDA 与基础 ML 包环境。它只用于 Lab 研究复现和环境审阅，不是 V2.1 Stable 运行环境，不定义 Stable 生产依赖，也不授权任何正式交易执行。

本任务不在仓库根目录新生成 `requirements.txt`，避免把 Lab GPU venv 快照误解为 Stable 或生产运行依赖。完整 `pip freeze` 快照应写入本地 ignored 目录：

```text
.local_research_outputs/aetfq3_lab/gpu_env/requirements.freeze.txt
```

## 当前推荐环境

- Python 版本：`3.12.0`
- venv 路径：`E:\aetfq3-lab\.venv`
- Python executable：`E:\aetfq3-lab\.venv\Scripts\python.exe`
- 平台：`Windows-11-10.0.22631-SP0`
- PyTorch 版本：`2.11.0+cu128`
- PyTorch CUDA 版本：`12.8`
- CUDA 可用：`true`
- GPU 型号：`NVIDIA GeForce RTX 4060 Ti`
- GPU 显存：`8.0 GB`
- `numpy`：`2.4.4`
- `pandas`：`3.0.3`
- `scikit-learn`：`1.9.0`
- `pyarrow`：`24.0.0`
- `matplotlib`：`3.10.9`

NVIDIA 驱动检查摘要：

- `nvidia-smi` 可正常执行。
- Driver / KMD 版本：`610.47`
- CUDA UMD 版本：`13.3`
- GPU 可见，当前为 WDDM 模式。

## 安装命令记录

本环境使用以下命令建立：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m pip install numpy pandas scikit-learn pyarrow matplotlib
```

注意：如果 CUDA wheel 下载失败，不应自动改用 CPU 版；需要人工确认后再调整安装方案。

## 适用范围

该环境适合：

- PyTorch 小模型 smoke test 与原型验证。
- GRU / TCN / 小型 Transformer 执行模型研究。
- 5分钟K 执行模型原型。
- `pandas` / `scikit-learn` 表格研究。
- Lab-only 环境诊断、模型诊断和本地 ignored 研究报告生成。

## 不适用范围

该环境不适合：

- Stable 正式交易。
- QMT 实盘执行。
- 自动下单。
- 大模型训练。
- 直接影响 `final_buy_action`。
- 生成 Stable 正式 `OrderIntent`。
- 绕过 `RiskGate`。
- 将模型输出直接接入 Stable。

## 复现步骤

从空环境重建 Lab GPU worker：

1. 进入 Lab 仓库：

```powershell
cd E:\aetfq3-lab
```

2. 创建独立 venv：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

3. 安装 PyTorch CUDA wheel：

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

4. 安装基础 ML 包：

```powershell
python -m pip install numpy pandas scikit-learn pyarrow matplotlib
```

5. 验证 CUDA：

```powershell
$code = @'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_version", torch.version.cuda)
print("device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device_name", torch.cuda.get_device_name(0))
'@
$code | python -
```

6. 生成本地 freeze 快照：

```powershell
python -m pip freeze > .local_research_outputs\aetfq3_lab\gpu_env\requirements.freeze.txt
```

## 边界声明

- 本环境属于 `aetfq3-lab / Lab`，不属于 V2.1 Stable。
- 本环境不修改 Stable entry。
- 本环境不影响 `final_buy_action`。
- 本环境不修改 `target_weight`。
- 本环境不修改 BUY / PROBE 阈值。
- 本环境不生成 `OrderIntent`。
- 本环境不接 QMT。
- 本环境只用于 Lab 研究、环境复现和只读诊断。
- 本环境快照不应被 Stable 自动读取或解释为正式交易依赖。
