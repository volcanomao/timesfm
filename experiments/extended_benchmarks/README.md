# 扩展基准测试

基准测试设置借鉴了Nixtla最初的[基准测试](https://github.com/AzulGarza/nixtla/tree/main/experiments/amazon-chronos)，它将时间序列基础模型与强大的统计集成进行了对比。后来，Chronos团队在这个[拉取请求](https://github.com/shchur/nixtla/tree/chronos-full-eval/experiments/amazon-chronos)中添加了更多的数据集。我们在这些扩展的基准测试中对所有数据集进行了比较。

## 在基准测试上运行TimesFM

按照主README中详细说明的步骤安装环境和软件包，然后从基本目录开始执行以下步骤。

```
conda activate tfm_env
TF_CPP_MIN_LOG_LEVEL=2 XLA_PYTHON_CLIENT_PREALLOCATE=false python3 -m experiments.extended_benchmarks.run_timesfm --model_path=<model_path> --backend="gpu"
```
在上述命令中，`<model_path>` 应指向可以从HuggingFace下载的检查点目录。

注意：在当前版本的TimesFM中，我们专注于点预测，因此mase和smape是使用中位数对应的分位数头即0.5分位数计算的。我们确实提供了10个分位数头，但在预训练后尚未校准。我们建议在您的应用中谨慎使用它们，或者在保持的情况下对其进行校准/一致化。后续版本会有更多更新。

## 基准测试结果

![基准测试结果表](./tfm_extended_new.png)

__更新：__ 我们已将TimeGPT-1添加到基准测试结果中。由于我们无法在此基准测试中运行TimeGPT-1，因此不得不移除Dominick数据集。注意，包括Dominick在内的先前结果仍然可在`./tfm_results.png`中找到。要重现TimeGPT-1的结果，请运行`run_timegpt.py`。

_备注：_ 除涉及TimeGPT的基准外，所有基准测试均在[g2-standard-32](https://cloud.google.com/compute/docs/gpus)上进行。由于TimeGPT-1只能通过API访问，时间列可能无法反映模型的真实速度，因为它还包括通信成本。此外，我们不确定TimeGPT的确切后端硬件。

我们可以看到，TimesFM在mase和smape方面表现最佳。更重要的是，它比其他方法快得多，特别是比StatisticalEnsemble快600倍以上，比Chronos（大型）快80倍。

注意：此基准测试仅在长时间序列数据集（如ETT每小时和15分钟）的`一个`小时间窗口上进行比较。我们在长期滚动验证任务上的更深入比较请见我们的长期基准测试。