# 研究结果索引

[`results/verified_summaries/`](../results/verified_summaries/) 保存从本机真实实验结果核对后收录的轻量 CSV / JSON。目录保留各阶段结果标识，便于从报告中的数值回溯到实验产出。

| 实验 | 核心证据文件 |
|---|---|
| V3 首轮对照 | [`model_valid_summary.csv`](../results/verified_summaries/model_valid_summary.csv) |
| V4 窗口与容量 | [`v4_model_summary.csv`](../results/verified_summaries/v4_model_summary.csv) · [`v4_paired_rankic.json`](../results/verified_summaries/v4_paired_rankic.json) |
| V5 LSTM窗口 | [`v5_window_summary.csv`](../results/verified_summaries/v5_window_summary.csv) · [`v5_paired_rankic.json`](../results/verified_summaries/v5_paired_rankic.json) |
| V6 顺序扰动 | [`v6_summary.csv`](../results/verified_summaries/v6_summary.csv) · [`v6_paired_rankic.json`](../results/verified_summaries/v6_paired_rankic.json) |
| V7 表达方案 | [`v7_representation_summary.csv`](../results/verified_summaries/v7_representation_summary.csv) · [`v7_paired_rankic.json`](../results/verified_summaries/v7_paired_rankic.json) |
| V8 控制实验 | [`v8_model_summary.csv`](../results/verified_summaries/v8_model_summary.csv) · [`v8_paired_rankic.json`](../results/verified_summaries/v8_paired_rankic.json) |
| V9 跨时期评价 | [`v9_model_summary.csv`](../results/verified_summaries/v9_model_summary.csv) · [`v9_yearly_rankic.csv`](../results/verified_summaries/v9_yearly_rankic.csv) |
| V10 理论经济分析 | [`v10_quintile_summary.csv`](../results/verified_summaries/v10_quintile_summary.csv) · [`v10_phase_summary.csv`](../results/verified_summaries/v10_phase_summary.csv) |
| V11 数据审计 | [`v11_verdict.json`](../results/verified_summaries/v11_verdict.json) · [`v11_eligibility_audit.json`](../results/verified_summaries/v11_eligibility_audit.json) |
| V12 最终评价 | [`v12_model_summary.csv`](../results/verified_summaries/v12_model_summary.csv) · [`v12_paired_rankic.json`](../results/verified_summaries/v12_paired_rankic.json) · [`v12_quintile_summary.csv`](../results/verified_summaries/v12_quintile_summary.csv) |

## 结果复核要点

- 最终日期：2025-01-02至2026-10-23（有效信号日，共436天）。
- 最终期单期滚动因子 Ridge：RankIC 0.037561；30日基础量价 LSTM：0.019391；同信息集 Ridge：−0.005239。
- V7基础量价LSTM 30日减1日的 RankIC差值：+0.027115；V8一致顺序扰动减正常排列：−0.020663。
- V12最终期30日滚动因子LSTM的Q5−Q1平均理论5日收益差：+0.355个百分点。

模型权重不在公开仓库，原始模型SHA-256请查阅 [`model_registry_sha256.json`](../archive_metadata/model_registry_sha256.json)。公开统计结果不构成可从零重训后的逐位一致性保证；若需完整复现，须获得相同Qlib原始快照和模型权重。
