
# 三阶段数据生成 - 使用指南

## 目录结构
```
trainsets_generation_code/
├── make_trainsets_qwen3vl.py          # 主入口
├── train_stage1_contrastive.py       # Stage1: 对比学习 triplets
├── train_stage2_sft_single.py        # Stage2: 单图 SFT
├── train_stage2_sft_pair.py          # Stage2: 双图 delta SFT
└── train_stage3_prefs.py             # Stage3: 真实图 DPO
```

## 快速开始

### 1. 复制代码到你的项目
```bash
cp trainsets_generation_code/*.py /path/to/your/project/tools/generate/
```

### 2. 运行主脚本（一键生成所有数据）
```bash
python tools/generate/make_trainsets_qwen3vl.py \
  --out_dir /data/rege_trainsets_qwen3vl \
  --seed 1 \
  --resolution 512 \
  --n_stage1_triplets 200000 \
  --n_stage2_sft_single 50000 \
  --n_stage2_sft_pair 50000 \
  --n_stage3_prefs 1000
```

### 3. 或者分阶段运行

#### Stage1: 对比学习 triplets
```bash
python tools/generate/train_stage1_contrastive.py \
  --out_dir data/stage1_contrastive \
  --seed 1 \
  --resolution 512 \
  --n_triplets 20
```

输出格式：
```json
{
  "id": "stage1_000001",
  "anchor": "images/triplet_000001_anchor.png",
  "positive": "images/triplet_000001_positive.png",
  "negative": "images/triplet_000001_negative.png",
  "label": {
    "anchor_time_minutes": 125,
    "positive_time_minutes": 125,
    "negative_time_minutes": 128,
    "negative_delta": 3
  },
  "meta": {
    "anchor_source": "blender",
    "positive_source": "matplot",
    "negative_source": "blender",
    "triplet_type": "cross_style_hard_neg"
  }
}
```

#### Stage2: SFT 数据
```bash
# 单图
python tools/generate/train_stage2_sft_single.py \
  --out_dir /data/stage2_sft_single \
  --seed 1 \
  --resolution 512 \
  --n_samples 50000

# 双图
python tools/generate/train_stage2_sft_pair.py \
  --out_dir /data/stage2_sft_pair \
  --seed 1 \
  --resolution 512 \
  --n_pairs 50000
```

输出格式（Qwen3-VL 标准格式）：
```json
{
  "id": "stage2_single_000001",
  "images": ["images/sample_000001.png"],
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "images/sample_000001.png"},
        {"type": "text", "text": "Read the exact time..."}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "<think>...</think>\n<answer>03:25</answer>"}
      ]
    }
  ],
  "label": {"time_minutes": 205, "time_hhmm": "03:25"},
  "meta": {"source": "blender_clean", "style_id": "style_01"}
}
```

#### Stage3: Preference 数据
```bash
python tools/generate/train_stage3_prefs.py \
  --out_dir /data/stage3_prefs \
  --seed 1 \
  --n_prefs 1000 \
  --stage2_single_dir /data/stage2_sft_single \
  --stage2_pair_dir /data/stage2_sft_pair \
  --mode synthetic
```

## 数据验证

### 验证 Stage1 triplets
```python
import json

with open('stage1_contrastive/annotations.jsonl') as f:
    for line in f:
        row = json.loads(line)
        # 检查 positive 和 anchor 时间相同
        assert row['label']['anchor_time_minutes'] == row['label']['positive_time_minutes']
        # 检查 negative 和 anchor 时间不同
        assert row['label']['anchor_time_minutes'] != row['label']['negative_time_minutes']
        # 检查 triplet 类型分布
        print(row['meta']['triplet_type'])
```

### 验证 Stage2 CoT 质量
```python
import json

with open('stage2_sft_single/annotations.jsonl') as f:
    sample = json.loads(f.readline())
    cot = sample['messages'][1]['content'][0]['text']
    print(cot)
    # 应该看到详细的推理步骤，不是硬编码
```

## 重要说明

### Stage1 的采样策略
- **40% cross_style_hard_neg**: anchor(Blender) + positive(Matplot, 同时间) + negative(Blender, 1-5分钟差)
- **30% same_style_hard_neg**: 都是 Blender，positive 同时间不同环境，negative 1-5分钟差
- **30% easy_neg**: anchor(Blender) + positive(Matplot, 同时间) + negative(Blender, 30-180分钟差)

### Stage2 的 CoT 生成
- Blender 数据：用 metadata 的 hour_angle/minute_angle 生成详细推理
- Matplot 数据：用简化推理（因为没有 metadata）
- 不要硬编码！每个样本的 CoT 都是根据 GT 动态生成的

### Stage3 的两种模式
1. **synthetic mode**（临时方案）：从 Stage2 数据生成错误答案
2. **real mode**（推荐方案）：用 Stage2 模型预测真实图，构造 preference pairs
   - 需要先训练 Stage2 模型
   - 需要准备真实图数据集

## 依赖项

这些脚本假设你有以下工具：
- `blender`：用于渲染 3D 时钟
- `tools/generate/blender/render_batch.py`：Blender 批量渲染脚本
- `tools/generate/matplot/render_matplot_batch.py`：Matplotlib 渲染脚本

如果你还没有这些工具，需要先实现它们。

## 故障排除

### 问题 1: Blender 找不到
```bash
# 检查 Blender 是否安装
which blender

# 如果没有，安装 Blender
sudo apt install blender  # Ubuntu
brew install blender      # macOS
```

### 问题 2: 图像池太小
如果看到 "Failed to sample triplet" 错误，说明图像池太小，增加 pool size：
```python
# 在 _pool_size() 函数里调整
def _pool_size(n_triplets):
    return min(50000, max(5000, int(n_triplets * 0.1)))  # 增加到 10%
```

### 问题 3: CoT 看起来还是硬编码
检查 Blender 输出的 metadata 是否包含 hour_angle 和 minute_angle：
```python
import json
with open('_pool_blender_clean/clean/samples.jsonl') as f:
    sample = json.loads(f.readline())
    print(sample['meta'])  # 应该有 hour_angle, minute_angle
```

## 下一步

1. ✅ 生成数据
2. ⏭️ 训练 Stage1 模型（对比学习）
3. ⏭️ 训练 Stage2 模型（SFT）
4. ⏭️ 用 Stage2 模型生成真实图的 preference pairs
5. ⏭️ 训练 Stage3 模型（DPO）

祝训练顺利！🚀
