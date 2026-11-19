# 环境与复现说明

## 1. 两种复现范围

**仅使用公开仓库：** 阅读完整源码、检查冻结指标、核对研究图表和运行不依赖真实行情的自动化测试。

**完整模型/收益复现：** 额外需要原始 Qlib A股 `cn_data` 数据快照、训练期环境、V1/V2中间缓存和V4/V7/V8冻结模型检查点。当前仓库不分发原始行情和权重。`archive_metadata/model_registry_sha256.json` 记录当次冻结模型哈希；公开结果表格来自真实实验运行。

## 2. 建议环境

原实验环境为 macOS / Apple Silicon M1 Max、Python 3.11、Qlib 0.9.7，PyTorch可使用MPS。`requirements.txt` 列出直接依赖，**不是精确锁定的完整依赖图**；在其他系统和CPU上需验证包兼容性。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -q
```

如Qlib安装在特定平台出现兼容性问题，优先参照该版本Qlib的官方部署要求。开发过程中测试全部通过并不等于有完整行情就可以无条件复现相同数值；数据快照与训练依赖版本也需要匹配。

## 3. 运行阶段及对应入口

以下命令仅供**在数据和依赖齐全、允许重新生成研究产物的独立复现实验**使用。既有研究中的最终评价权重和结果已经冻结，不应在已公布的最终结果上事后调参。

```bash
python 00_audit.py
python 01_build_factor_panel.py
python 02_prepare_sequences.py
python 03_run_model_pilot.py --device auto
python 04_test_history_hypothesis.py --experiment both --device auto
python 05_lstm_window_study.py --device auto
python 06_order_perturbation.py --experiment both --device auto
python 07_compare_representations.py --stage all --device auto
python 08_test_a30_temporal_value.py --experiment all --device auto
python 09_development_evaluation.py --stage audit
python 09_development_evaluation.py --stage evaluate --device auto
python 10_economic_diagnostics.py --stage audit
python 10_economic_diagnostics.py --stage evaluate
python 11_data_validity_audit.py --stage audit
python 12_final_evaluation.py --stage audit
python 12_final_evaluation.py --stage evaluate --device auto
```

入口脚本默认从运行目录的 `output/` 加载上一阶段文件，外部行情预期位于 `~/.qlib/qlib_data/cn_data`。请核对源码中的来源路径、交易日历、标签截止条件，避免与其他数据集混合。对于冻结模型的对照复核，应首先检查本地模型文件SHA-256与登记清单是否一致：

```bash
python scripts/verify_local_sources.py
```

仓库只有源码时，上述命令因缺少检查点而返回非零状态，是预期行为，不代表公开仓库损坏。

## 4. 私有证据保全

研究机器上可使用 `scripts/export_local_evidence.py --output <private-path>.zip` 导出冻结模型及部分小型结果；**该压缩包包含权重与本地环境信息，仅用于私人备份，不应上传公开GitHub**。原始Qlib行情及数据使用授权需要另行保存。

## 5. 样本和推断限制

实验报告从模型训练、受控探索、跨时期评价到最终评价的各阶段结果；样本由完整窗口及未来标签可观测性共同决定。V11发现2023—2024年有47条未来标签不可用候选被排除，V12最终期为128条。参阅 [`DATA_LIMITATIONS.md`](DATA_LIMITATIONS.md)。
