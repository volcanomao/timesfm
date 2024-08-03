# TimesFM

TimesFM（时间序列基础模型）是由 Google Research 开发的一个预训练时间序列基础模型，用于时间序列预测。

* 论文: [A decoder-only foundation model for time-series forecasting](https://arxiv.org/abs/2310.10688)，将出现在 ICML 2024 上。
* [Google Research 博客](https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/)
* [Hugging Face 检查点库](https://huggingface.co/google/timesfm-1.0-200m)

该库包含了加载公共 TimesFM 检查点并运行模型推理的代码。请访问我们的 
[Hugging Face 检查点库](https://huggingface.co/google/timesfm-1.0-200m)
下载模型检查点。

这不是 Google 官方支持的产品。

我们建议至少 16GB 的内存来加载 TimesFM 依赖项。

## 更新 - 2024 年 7 月 15 日

- 要安装 TimesFM，现在可以简单地使用: `pip install timesfm`。
- 推出了 [微调支持](https://github.com/google-research/timesfm/blob/master/notebooks/finetuning.ipynb)，允许您在自己的数据上微调预训练的 TimesFM 模型的权重。
- 推出了 [~零样本协变量支持](https://github.com/google-research/timesfm/blob/master/notebooks/covariates.ipynb)，支持外部回归因子。更多细节 [这里](https://github.com/google-research/timesfm?tab=readme-ov-file#covariates-support)。

## 检查点 timesfm-1.0-200m

timesfm-1.0-200m 是第一个开放模型检查点：

- 它对上下文长度最长为 512 个时间点的单变量时间序列进行预测，并支持任意的预测长度，具有可选的频率指示器。
- 它专注于点预测，不支持概率预测。我们实验性地提供了分位数头，但在预训练后尚未进行校准。
- 它要求上下文是连续的（即没有“空洞”），并且上下文和预测期的频率相同。

## 基准测试

请参阅我们的 [扩展基准测试](https://github.com/google-research/timesfm/tree/master/experiments/extended_benchmarks) 和 [长时间预测基准测试](https://github.com/google-research/timesfm/tree/master/experiments/long_horizon_benchmarks) 的结果表。

请查阅 `experiments/` 目录下各基准测试目录中的 README 文件，以获取在相应基准测试上运行 TimesFM 的说明。

## 安装

### 作为包安装

要将 TimesFM 安装为包，您可以运行以下命令，而无需克隆此库：

`pip install timesfm`

### 使用 conda 安装

为了调用 TimesFM，我们提供了两个环境文件。在 `timesfm` 目录下，对于 GPU 安装（假设已设置 CUDA 12），您可以通过以下命令从基础文件夹创建 conda 环境 `tfm_env`：

```
conda env create --file=environment.yml
```

对于 CPU 安装，请使用：

```
conda env create --file=environment_cpu.yml
```
to create the environment instead.

接下来使用：

```
conda activate tfm_env
pip install -e .
```
安装包。

**注意**：

1. 运行提供的基准测试将需要额外的依赖项。请使用 `experiments` 下的环境文件。

2. 依赖项 `lingvo` 不支持 ARM 架构，该代码不适用于 Apple Silicon 的机器。我们知道这个问题，并正在寻找解决方案。敬请关注。

### 使用 poetry 进行本地安装

要从当前存储库/本地版本安装（就像您之前使用 `pip -e .` 一样），您可以运行以下命令：

```
pip install poetry # optional
poetry install
```

这将安装环境到本地的 .venv 文件夹（取决于配置），并将 python 命令匹配到 poetry 环境。如果不是这样，您可以使用 `poetry run python` 来使用本地环境。

### 注意事项

1. 运行提供的基准测试将需要额外的依赖项。请使用 `experiments` 下的环境文件。

2. 依赖项 `lingvo` 不支持 ARM 架构，该代码不适用于 Apple Silicon 的机器。我们知道这个问题，并正在寻找解决方案。敬请关注。

#### 构建包并发布到 PyPI

可以使用命令 `poetry build` 来构建包。

要构建并发布到 PyPI，可以使用命令 `poetry publish`。该命令需要用户具有发布到 PyPI 仓库的必要权限。

## 使用 

### 初始化模型并加载检查点
然后可以像这样加载基类：

```python
import timesfm

tfm = timesfm.TimesFm(
    context_len=<context>,
    horizon_len=<horizon>,
    input_patch_len=32,
    output_patch_len=128,
    num_layers=20,
    model_dims=1280,
    backend=<backend>,
)
tfm.load_from_checkpoint(repo_id="google/timesfm-1.0-200m")
```

注意，四个参数已固定以加载 200m 模型：

```python
input_patch_len=32,
output_patch_len=128,
num_layers=20,
model_dims=1280,
```

1. `context_len` 可以设置为模型的最大上下文长度。**它需要是 `input_patch_len` 的倍数，即 32 的倍数。** 您可以向 `tfm.forecast()` 函数提供更短的时间序列，模型将会处理它。目前，模型处理的最大上下文长度为 512，可以在后续版本中增加。输入的时间序列可以具有 **任何上下文长度**。如果需要，推理代码会处理填充/截断。

2. 预测范围长度可以设置为任何值。我们建议将其设置为您在预测任务中所需的最大预测范围长度。我们通常建议预测范围长度 <= 上下文长度，但这不是函数调用的要求。

3. `backend` 可以是 "cpu"、"gpu" 或 "tpu" 之一，区分大小写。

### 执行推理

我们提供了 API 来从数组输入或 `pandas` 数据框进行预测。两种预测方法都需要（1）输入时间序列上下文，（2）以及它们的频率。请查看 `tfm.forecast()` 和 `tfm.forecast_on_df()` 函数的文档，以获取详细说明。

特别地，关于频率，TimesFM 期望的类别指示符值为 {0, 1, 2}：

- **0**（默认）：高频率，长预测时间序列。我们建议用于日粒度的时间序列。
- **1**：中频率时间序列。我们建议用于每周和每月的数据。
- **2**：低频率，短预测时间序列。我们建议用于超过每月的数据，例如季度或年度。

该类别值应直接提供给数组输入。对于数据框输入，我们将传统的频率字母编码转换为我们期望的类别，其中

- **0**：T, MIN, H, D, B, U
- **1**：W, M
- **2**：Q, Y

请注意，您不必严格遵循我们的建议。尽管这是我们在模型训练期间的设置，并且我们期望它提供最佳的预测结果，但您也可以将频率输入视为一个自由参数，并根据具体用例进行调整。

示例：

数组输入，分别设置为低、中和高频率。

```python
import numpy as np
forecast_input = [
    np.sin(np.linspace(0, 20, 100)),
    np.sin(np.linspace(0, 20, 200)),
    np.sin(np.linspace(0, 20, 400)),
]
frequency_input = [0, 1, 2]

point_forecast, experimental_quantile_forecast = tfm.forecast(
    forecast_input,
    freq=frequency_input,
)
```

`pandas` dataframe, with the frequency set to "M" monthly.

```python
import pandas as pd

# e.g. input_df is
#       unique_id  ds          y
# 0     T1         1975-12-31  697458.0
# 1     T1         1976-01-31  1187650.0
# 2     T1         1976-02-29  1069690.0
# 3     T1         1976-03-31  1078430.0
# 4     T1         1976-04-30  1059910.0
# ...   ...        ...         ...
# 8175  T99        1986-01-31  602.0
# 8176  T99        1986-02-28  684.0
# 8177  T99        1986-03-31  818.0
# 8178  T99        1986-04-30  836.0
# 8179  T99        1986-05-31  878.0

forecast_df = tfm.forecast_on_df(
    inputs=input_df,
    freq="M",  # monthly
    value_name="y",
    num_jobs=-1,
)
```

## 协变量支持

我们现在在 TimesFM 之上有一个外部回归库，可以支持静态协变量以及未来可用的动态协变量。我们在 [notebooks/covariates.ipynb](https://github.com/google-research/timesfm/blob/master/notebooks/covariates.ipynb) 中提供了一个使用示例。

让我们以一个预测超市销售的玩具示例为例：

**任务：** 给定本周（7天）的每日销售数据，预测下周（7天）的每日销售。

```
Product: ice cream
Daily_sales: [30, 30, 4, 5, 7, 8, 10]
Category: food
Base_price: 1.99
Weekday: [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6]
Has_promotion: [Yes, Yes, No, No, No, Yes, Yes, No, No, No, No, No, No, No]
Daily_temperature: [31.0, 24.3, 19.4, 26.2, 24.6, 30.0, 31.1, 32.4, 30.9, 26.0, 25.0, 27.8, 29.5, 31.2]
```

```
Product: sunscreen
Daily_sales: [5, 7, 12, 13, 5, 6, 10]
Category: skin product
Base_price: 29.99
Weekday: [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6]
Has_promotion: [No, No, Yes, Yes, No, No, No, Yes, Yes, Yes, Yes, Yes, Yes, Yes]
Daily_temperature: [31.0, 24.3, 19.4, 26.2, 24.6, 30.0, 31.1, 32.4, 30.9, 26.0, 25.0, 27.8, 29.5, 31.2]
```

在这个示例中，除了 `Daily_sales`，我们还有协变量 `Category`、`Base_price`、`Weekday`、`Has_promotion` 和 `Daily_temperature`。让我们介绍一些概念：

**静态协变量** 是针对每个时间序列的协变量。
- 在我们的例子中，`Category` 是一个 **静态分类协变量**，
- `Base_price` 是一个 **静态数值协变量**。

**动态协变量** 是针对每个时间戳的协变量。
- 与日期/时间相关的特征通常可以视为动态协变量。
- 在我们的例子中，`Weekday` 和 `Has_promotion` 是 **动态分类协变量**。
- `Daily_temperature` 是一个 **动态数值协变量**。

**注意：** 这里我们规定动态协变量必须覆盖预测的上下文和预测范围。例如，示例中的所有动态协变量都有 14 个值：前 7 个值对应于观察到的 7 天，最后 7 个值对应于接下来的 7 天。

我们现在可以将两个产品的历史数据以及静态和动态协变量作为批量输入提供给 TimesFM，并生成考虑了协变量的预测。要了解更多信息，请查看 [notebooks/covariates.ipynb](https://github.com/google-research/timesfm/blob/master/notebooks/covariates.ipynb) 中的示例。

## 微调

我们在 [notebooks/finetuning.ipynb](https://github.com/google-research/timesfm/blob/master/notebooks/finetuning.ipynb) 中提供了在新数据集上微调模型的示例。

## 贡献风格指南

如果您想提交 PR，请确保您使用我们的格式化风格。我们使用 [yapf](https://github.com/google/yapf) 进行格式化，使用以下选项，

```
[style]
based_on_style = google
# Add your custom style rules here
indent_width = 2
spaces_before_comment = 2

```

Please run `yapf --in-place --recursive <filename>` on all affected files.
