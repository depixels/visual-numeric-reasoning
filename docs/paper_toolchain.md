# Paper Toolchain Commands

这份文档面向当前仓库的论文实验链路，目标是把命令使用方式、关键可选参数、受限取值和推荐调用顺序整理清楚。

覆盖范围：
- controlled Blender OOD benchmark 生成
- API 单次评测
- repeated eval 稳定性分析
- GT + prediction join
- 按物理因素聚合统计
- tilt / photometric 可视化
- 主表 / 消融表构建
- error taxonomy
- Stage2 `answer_only` vs `grounded_rationale` 对照数据生成

## 0. 推荐执行顺序

1. 生成 benchmark
2. 跑单次 eval
3. 跑 repeated eval
4. join GT 与 per-sample predictions
5. 做 bucket aggregation
6. 画 tilt 曲线
7. 画 specular / blur 曲线
8. 生成 ablation table
9. 生成 error taxonomy
10. 跑 Stage2 `target_format` 对照

---

## 1. 生成 Controlled OOD Benchmark

脚本：
- `tools/generate/make_ood_blender_benchmark.py`

用途：
- 生成论文用 Blender OOD benchmark
- 默认主 split:
  - `clean`: moderate physical OOD
  - `noisy`: severe physical OOD
- 可选 factorized split:
  - `viewpoint_only`
  - `illumination_only`

### 最小命令

```bash
python tools/generate/make_ood_blender_benchmark.py \
  --out_dir data/bench/clock_ood_v1 \
  --clean_n 500 \
  --noisy_n 500 \
  --resolution 512 \
  --seed 2026
```

### 带 factorized split 的命令

```bash
python tools/generate/make_ood_blender_benchmark.py \
  --out_dir data/bench/clock_ood_v1 \
  --clean_n 500 \
  --noisy_n 500 \
  --viewpoint_only_n 200 \
  --illumination_only_n 200 \
  --resolution 512 \
  --seed 2026 \
  --validate
```

### 关键参数

| 参数 | 说明 | 默认值 | 受限取值 |
|---|---|---:|---|
| `--out_dir` | 输出目录 | 无 | 任意路径 |
| `--clean_n` | clean split 样本数 | `100` | 正整数 |
| `--noisy_n` | noisy split 样本数 | `100` | 正整数 |
| `--viewpoint_only_n` | viewpoint-only split 样本数 | `0` | 非负整数 |
| `--illumination_only_n` | illumination-only split 样本数 | `0` | 非负整数 |
| `--resolution` | 图像分辨率 | `512` | 正整数 |
| `--seed` | 随机种子 | `2026` | 整数 |
| `--style_bank_dir` | Blender style bank | `tools/generate/blender/assets/styles` | 现有 style 目录 |
| `--blender_bin` | Blender 可执行文件 | `blender` | 可执行命令路径 |
| `--validate` | 生成后自动校验 | 关闭 | flag |

### 生成结果

每个 split 目录下会包含：
- `samples.jsonl`
- `images/*.png`

`samples.jsonl` 中会显式写入：
- `split`
- `meta.benchmark_split`
- `meta.ood_severity`
- `meta.source`
- `meta.view_bucket`
- `meta.tilt_bucket`
- `meta.specular_bucket`
- `meta.blur_bucket`
- `meta.lighting_env_id`

### 论文叙事对应关系

- `clean`: moderate physical OOD，主要是较明显 viewpoint shift，辅以轻微 photometric change
- `noisy`: severe physical OOD，联合 viewpoint / illumination / blur / specular shift
- `viewpoint_only`: 尽量 isolate viewpoint
- `illumination_only`: 尽量 isolate lighting / photometric change

---

## 2. 单次 API Eval

脚本：
- `tools/eval/eval_clock_api.py`

用途：
- 调用 API 或 OpenAI-compatible 视觉模型
- 对 analog clock readout 做单次评测
- 输出统一 per-sample 结果和 overall metrics

### 最小命令

```bash
python tools/eval/eval_clock_api.py \
  --gt_jsonl data/bench/clock_ood_v1/clean/samples.jsonl \
  --images_root data/bench/clock_ood_v1/clean \
  --provider vllm_qwen \
  --model molmo \
  --base_url http://127.0.0.1:8003/v1 \
  --api_key EMPTY \
  --output_dir data/results/molmo_clean
```

### 关键参数

| 参数 | 说明 | 默认值 | 受限取值 |
|---|---|---:|---|
| `--gt_jsonl` | GT annotation 文件 | 无 | JSONL 路径 |
| `--images_root` | 图像根目录 | `gt_jsonl` 所在目录 | 路径 |
| `--provider` | API provider | 无 | `vllm_qwen`, `gemini_3_pro`, `azure_gpt`, `qwen_dashscope` |
| `--model` | 模型名 / deployment 名 | Qwen 本地默认值 | provider 对应字符串 |
| `--base_url` | OpenAI-compatible endpoint | `http://127.0.0.1:8001/v1` | URL |
| `--api_key` | API key | 无 | 字符串 |
| `--timeout` | 请求超时秒数 | `3600` | 正整数 |
| `--developer_prompt` | developer prompt | `None` | 字符串 |
| `--system_prompt` | system prompt | `None` | 字符串 |
| `--user_prompt` | 用户侧评测 prompt | 内置 answer-only prompt | 字符串 |
| `--max_tokens` | 生成上限 | `1024` | 正整数 |
| `--temperature` | 采样温度 | `0.01` | 浮点数 |
| `--max_retries` | 失败重试次数 | `5` | 正整数 |
| `--retry_sleep` | 重试间隔 | `0.8` | 浮点数 |
| `--start_index` | 从第几个样本开始 | `0` | 非负整数 |
| `--limit` | 只跑多少个样本 | `None` | 正整数 |
| `--save_every` | 每多少条写盘 | `20` | 正整数 |
| `--output_dir` | 输出目录 | 无 | 路径 |
| `--pred_json` | 预测列表 json 文件名 | `predictions.json` | 文件名 |
| `--pred_jsonl` | per-sample jsonl 文件名 | `per_sample_results.jsonl` | 文件名 |
| `--pred_csv` | per-sample csv 文件名 | `per_sample_results.csv` | 文件名 |
| `--metrics_json` | overall metrics 文件名 | `metrics.json` | 文件名 |

### 输出文件

- `predictions.json`
- `per_sample_results.jsonl`
- `per_sample_results.csv`
- `metrics.json`

### per-sample 核心字段

- `id`
- `split`
- `image`
- `gt_time_hhmm`
- `gt_time_minutes`
- `pred_time_hhmm`
- `pred_time_minutes`
- `is_exact`
- `tol_1`
- `tol_5`
- `abs_err_minutes`
- `hour_correct`
- `minute_correct`
- `second_correct`
- `raw_output`
- `parsed_ok`

### metrics.json 核心字段

- `parsed_rate`
- `exact_acc`
- `tol1_acc`
- `tol5_acc`
- `hour_acc`
- `minute_acc`
- `second_acc`
- `mae`
- `median_abs_error_minutes`

---

## 3. Repeated Eval 与 Sampling Variance

脚本：
- `tools/eval/eval_clock_api_repeat.py`

用途：
- 同一模型、同一数据、多次重复采样评测
- 支持 run-level mean/std
- 支持 per-sample stability
- 支持 majority vote
- 支持 oracle best-of-n

### 最小命令

```bash
python tools/eval/eval_clock_api_repeat.py \
  --num_runs 5 \
  --gt_jsonl data/bench/clock_ood_v1/noisy/samples.jsonl \
  --images_root data/bench/clock_ood_v1/noisy \
  --provider vllm_qwen \
  --model molmo \
  --base_url http://127.0.0.1:8003/v1 \
  --api_key EMPTY \
  --temperature 0.7 \
  --output_dir data/results/molmo_noisy_repeat5
```

### 关键参数

除了继承 `eval_clock_api.py` 的主要参数外，还新增：

| 参数 | 说明 | 默认值 | 受限取值 |
|---|---|---:|---|
| `--num_runs` | 重复运行次数 | `5` | 正整数 |

### 输出文件

每次运行会落到：
- `run_01/`
- `run_02/`
- ...

汇总层输出：
- `summary.json`
- `majority_vote_results.jsonl`
- `majority_vote_results.csv`
- `majority_vote_metrics.json`
- `oracle_best_of_n_metrics.json`
- `per_sample_stability.jsonl`
- `per_sample_stability.csv`

### 适用分析

- sampling variance
- run-level mean/std
- majority-vote gain
- oracle best-of-n upper bound
- per-sample agreement rate

---

## 4. GT 与 Prediction Join

脚本：
- `tools/analysis/join_preds_with_gt.py`

用途：
- 读取 GT jsonl 和 per-sample pred
- merge GT label + GT meta + pred
- 输出后续统计和画图统一输入

### 最小命令

```bash
python tools/analysis/join_preds_with_gt.py \
  --gt_jsonl data/bench/clock_ood_v1/clean/samples.jsonl \
  --pred_path data/results/molmo_clean/per_sample_results.jsonl \
  --output_jsonl data/results/molmo_clean/joined.jsonl
```

### 关键参数

| 参数 | 说明 | 默认值 | 受限取值 |
|---|---|---:|---|
| `--gt_jsonl` | GT 样本 jsonl | 无 | 路径 |
| `--pred_path` | `jsonl/json` 形式的 per-sample predictions | 无 | 路径 |
| `--output_jsonl` | 输出 joined jsonl | 无 | 路径 |
| `--split` | 强制覆盖 split 字段 | `None` | 字符串 |

### joined 输出会包含

- 预测字段与指标字段
- `style_id`
- `source`
- `yaw`
- `pitch`
- `roll`
- `view_bucket`
- `tilt_bucket`
- `specular`
- `specular_bucket`
- `motion_blur`
- `defocus`
- `blur_bucket`
- `lighting_env_id`
- `gt_meta`
- `gt_label`

---

## 5. 按 Bucket 聚合统计

脚本：
- `tools/analysis/aggregate_metrics.py`

用途：
- 对 joined jsonl 按任意字段分组聚合
- 输出 csv，适合做表和后续绘图

### 最小命令

```bash
python tools/analysis/aggregate_metrics.py \
  --input_jsonl data/results/molmo_clean/joined.jsonl \
  --group_by tilt_bucket \
  --output_csv data/results/molmo_clean/tilt_metrics.csv
```

### 多字段分组

```bash
python tools/analysis/aggregate_metrics.py \
  --input_jsonl data/results/molmo_noisy/joined.jsonl \
  --group_by split tilt_bucket \
  --output_csv data/results/molmo_noisy/split_tilt_metrics.csv
```

### 关键参数

| 参数 | 说明 | 默认值 | 受限取值 |
|---|---|---:|---|
| `--input_jsonl` | joined jsonl | 无 | 路径 |
| `--output_csv` | 聚合结果 csv | 无 | 路径 |
| `--group_by` | 分组字段，可多个 | 无 | 任意字段名 |
| `--split` | 只统计某个 split | `None` | 字符串 |

### 推荐分组字段

- `tilt_bucket`
- `specular_bucket`
- `blur_bucket`
- `style_id`
- `lighting_env_id`

### 输出统计

- `n`
- `exact_acc`
- `tol1_acc`
- `tol5_acc`
- `hour_acc`
- `minute_acc`
- `mae`

---

## 6. Tilt Robustness 曲线

脚本：
- `tools/eval/plot_acc_vs_tilt.py`

用途：
- 画论文主图 `accuracy vs tilt`
- 支持多模型对比
- 支持 clean / noisy 分开作图
- 当前 `tilt_bucket` 默认基于 `abs(yaw)`

### 最小命令

```bash
python tools/eval/plot_acc_vs_tilt.py \
  --input baseline=data/results/baseline_clean/joined.jsonl \
  --input "ours full"=data/results/ours_full_clean/joined.jsonl \
  --split clean \
  --metric exact_acc \
  --output_prefix data/figs/acc_vs_tilt_clean
```

### 多模型命令

```bash
python tools/eval/plot_acc_vs_tilt.py \
  --input baseline=data/results/baseline_noisy/joined.jsonl \
  --input "ours stage1+2"=data/results/ours_stage12_noisy/joined.jsonl \
  --input "ours full"=data/results/ours_full_noisy/joined.jsonl \
  --input molmo=data/results/molmo_noisy/joined.jsonl \
  --split noisy \
  --metric tol5_acc \
  --output_prefix data/figs/acc_vs_tilt_noisy_tol5
```

### 关键参数

| 参数 | 说明 | 默认值 | 受限取值 |
|---|---|---:|---|
| `--input` | `label=joined.jsonl`，可重复多次 | 无 | 多个 |
| `--output_prefix` | 输出前缀 | 无 | 路径前缀 |
| `--metric` | 使用哪种准确率指标 | `exact_acc` | `exact_acc`, `tol1_acc`, `tol5_acc` |
| `--split` | 只画某个 split | `None` | `clean`, `noisy` 或自定义 |
| `--title` | 自定义标题 | `None` | 字符串 |

### 当前 bucket 顺序

- `Front`
- `10-20`
- `20-30`
- `30-40`
- `40-50`
- `50-60`
- `60-70`
- `70+`

### 输出

- `${output_prefix}.png`
- `${output_prefix}.pdf`

### 说明

- 当前主图按 `abs(yaw)` 分桶，不再使用旧脚本中的 `90 - pitch`
- 图中默认带数值标注
- 推荐 clean / noisy 分开出图

---

## 7. Photometric Robustness 曲线

脚本：
- `tools/analysis/plot_photometric_curve.py`

用途：
- 画 `accuracy vs specular_bucket`
- 画 `accuracy vs blur_bucket`
- 适合正文补图或 appendix

### specular 曲线

```bash
python tools/analysis/plot_photometric_curve.py \
  --input baseline=data/results/baseline_noisy/joined.jsonl \
  --input "ours full"=data/results/ours_full_noisy/joined.jsonl \
  --field specular_bucket \
  --split noisy \
  --metric exact_acc \
  --output_prefix data/figs/specular_curve_noisy
```

### blur 曲线

```bash
python tools/analysis/plot_photometric_curve.py \
  --input baseline=data/results/baseline_noisy/joined.jsonl \
  --input "ours full"=data/results/ours_full_noisy/joined.jsonl \
  --field blur_bucket \
  --split noisy \
  --metric tol5_acc \
  --output_prefix data/figs/blur_curve_noisy
```

### 关键参数

| 参数 | 说明 | 默认值 | 受限取值 |
|---|---|---:|---|
| `--input` | `label=joined.jsonl`，可重复多次 | 无 | 多个 |
| `--field` | 横轴 bucket 类型 | 无 | `specular_bucket`, `blur_bucket` |
| `--output_prefix` | 输出前缀 | 无 | 路径前缀 |
| `--metric` | 纵轴指标 | `exact_acc` | `exact_acc`, `tol1_acc`, `tol5_acc` |
| `--split` | 只画某个 split | `None` | 字符串 |

### 当前 bucket 定义

`specular_bucket`:
- `0.0`
- `0.0-0.1`
- `0.1-0.3`
- `0.3-0.6`
- `0.6+`

`blur_bucket`:
- `0.0`
- `0.0-0.05`
- `0.05-0.15`
- `0.15-0.30`
- `0.30+`

### 输出

- `${output_prefix}.png`
- `${output_prefix}.pdf`

---

## 8. 主表 / 消融表

脚本：
- `tools/analysis/build_ablation_table.py`

用途：
- 从多个结果目录读取 metrics
- 组装 baseline / stage1 / stage2 / stage1+2 / full 等表格

### 最小命令

```bash
python tools/analysis/build_ablation_table.py \
  --setting baseline=data/results/baseline \
  --setting stage1=data/results/stage1 \
  --setting stage2=data/results/stage2 \
  --setting stage1+2=data/results/stage1_stage2 \
  --setting full=data/results/full \
  --output_csv data/tables/ablation.csv \
  --output_json data/tables/ablation.json
```

### 关键参数

| 参数 | 说明 | 默认值 | 受限取值 |
|---|---|---:|---|
| `--setting` | `name=result_dir`，可重复多次 | 无 | 多个 |
| `--output_csv` | 表格 csv | 无 | 路径 |
| `--output_json` | 原始 json 汇总 | `None` | 路径 |

### 读取逻辑

优先尝试：
- `${result_dir}/metrics.json`
- `${result_dir}/clean/metrics.json`
- `${result_dir}/noisy/metrics.json`
- `${result_dir}/clean/majority_vote_metrics.json`
- `${result_dir}/noisy/majority_vote_metrics.json`

### 表中常见指标

- `exact_acc`
- `tol1_acc`
- `tol5_acc`
- `hour_acc`
- `minute_acc`
- `mae`
- `parsed_rate`
- `clean_*`
- `noisy_*`

---

## 9. Error Taxonomy

脚本：
- `tools/analysis/error_taxonomy.py`

用途：
- 从 joined jsonl 自动总结错误类型
- 用于论文错误分析表

### 最小命令

```bash
python tools/analysis/error_taxonomy.py \
  --input_jsonl data/results/ours_full_noisy/joined.jsonl \
  --output_json data/results/ours_full_noisy/error_taxonomy.json \
  --output_csv data/results/ours_full_noisy/error_taxonomy.csv \
  --output_jsonl data/results/ours_full_noisy/error_taxonomy_samples.jsonl
```

### 关键参数

| 参数 | 说明 | 默认值 | 受限取值 |
|---|---|---:|---|
| `--input_jsonl` | joined jsonl | 无 | 路径 |
| `--output_json` | 汇总 json | 无 | 路径 |
| `--output_csv` | 汇总 csv | `None` | 路径 |
| `--output_jsonl` | 带 category 的逐样本 jsonl | `None` | 路径 |

### 当前支持的错误类别

- `unparsed`
- `hour_only_wrong`
- `minute_only_wrong`
- `both_wrong`
- `near_miss_1to5`
- `large_error_5plus`
- `second_only_wrong`
- `correct`

---

## 10. Stage2 对照数据生成

脚本：
- `tools/generate/train_stage2_sft_single.py`
- `tools/generate/train_stage2_sft_pair.py`

用途：
- 生成 Stage2 SFT 数据
- 支持关键对照：
  - `answer_only`
  - `grounded_rationale`

### 单图 answer-only

```bash
python tools/generate/train_stage2_sft_single.py \
  --out_dir data/trainsets/stage2_single_answer_only \
  --reuse_pools_dir data/pools \
  --n_samples 50000 \
  --target_format answer_only
```

### 单图 grounded rationale

```bash
python tools/generate/train_stage2_sft_single.py \
  --out_dir data/trainsets/stage2_single_grounded \
  --reuse_pools_dir data/pools \
  --n_samples 50000 \
  --target_format grounded_rationale
```

### 双图 answer-only

```bash
python tools/generate/train_stage2_sft_pair.py \
  --out_dir data/trainsets/stage2_pair_answer_only \
  --reuse_pools_dir data/pools \
  --n_pairs 50000 \
  --target_format answer_only
```

### 关键参数

`train_stage2_sft_single.py`

| 参数 | 说明 | 默认值 | 受限取值 |
|---|---|---:|---|
| `--out_dir` | 输出目录 | 无 | 路径 |
| `--seed` | 随机种子 | `1` | 整数 |
| `--resolution` | 渲染分辨率 | `512` | 正整数 |
| `--n_samples` | 样本数 | 无 | 正整数 |
| `--reuse_pools_dir` | 复用样本池 | `None` | 路径 |
| `--pool_blender_splits` | 复用哪些 blender split | `both` | `clean`, `noisy`, `both` |
| `--pool_use_matplot` | 是否混入 matplot pool | 关闭 | flag |
| `--target_format` | assistant target 形式 | `grounded_rationale` | `answer_only`, `grounded_rationale` |
| `--dry_run` | 只跑少量样本并打印 | `0` | 非负整数 |
| `--resume` | 断点续跑 | 关闭 | flag |

`train_stage2_sft_pair.py`

| 参数 | 说明 | 默认值 | 受限取值 |
|---|---|---:|---|
| `--out_dir` | 输出目录 | 无 | 路径 |
| `--seed` | 随机种子 | `1` | 整数 |
| `--resolution` | 分辨率 | `512` | 正整数 |
| `--n_pairs` | pair 数量 | 无 | 正整数 |
| `--reuse_pools_dir` | 复用样本池 | `None` | 路径 |
| `--pool_blender_splits` | blender split 选择 | `both` | `clean`, `noisy`, `both` |
| `--pool_use_matplot` | 是否用 matplot | 关闭 | flag |
| `--target_format` | assistant target 形式 | `grounded_rationale` | `answer_only`, `grounded_rationale` |
| `--resume` | 断点续跑 | 关闭 | flag |

### target_format 语义

- `answer_only`
  - assistant target 只包含最终答案
  - 单图格式如 `03:25`
  - 双图格式如 `-15`

- `grounded_rationale`
  - assistant target 保留 `<think>...</think>\n<answer>...</answer>`
  - 对应 grounded rationale supervision

---

## 11. 常见论文工作流示例

### A. clean / noisy 主表

1. 分别对 `clean`、`noisy` 跑 eval
2. 各自生成 `metrics.json`
3. 用 `build_ablation_table.py` 汇总

### B. 主图 `accuracy vs tilt`

1. 对每个模型结果运行 `join_preds_with_gt.py`
2. 调用 `plot_acc_vs_tilt.py`
3. clean / noisy 分开出图

### C. photometric robustness

1. join
2. 按需先 `aggregate_metrics.py`
3. 用 `plot_photometric_curve.py` 分别画 `specular_bucket` 和 `blur_bucket`

### D. repeated-eval 稳定性

1. 跑 `eval_clock_api_repeat.py`
2. 查看：
   - `summary.json`
   - `majority_vote_metrics.json`
   - `oracle_best_of_n_metrics.json`
   - `per_sample_stability.jsonl`

### E. 错误分析

1. 先 join
2. 再跑 `error_taxonomy.py`
3. 如果需要按 bucket 细分，可先用 `aggregate_metrics.py`

---

## 12. 当前实现边界

- `tilt` 当前统一按 `abs(yaw)` 分桶，不再使用旧版 `pitch` 近似
- photometric 曲线当前支持：
  - `specular_bucket`
  - `blur_bucket`
- `build_ablation_table.py` 当前是轻量汇总器，适合先生成 csv，再在论文表格里做最终排版
- repeated eval 的 `oracle best-of-n` 是样本级上界分析，不是实际部署策略
