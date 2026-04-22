# Prefetch Shift 决定性实验计划

## 目标

把当前的 2x2 现象从“有趣观察”补强成“可复现、可归因、可解释”的论文级证据链，重点回答四个问题：

1. 现象是否稳定复现，而不是系统噪声？
2. 退化是否能排除长度、顺序、任务集合等混杂因素？
3. 高 hit 低收益的根因是否真的是 `late/useless prefetch`，而不是简单 miss？
4. 该 failure 是否随着 mixing / drift 强度增强而连续恶化？

## 证据闭环

### H1 现象可复现

在 `shift + mixing` 条件下，decoder prefetch 相比 on-demand 持续出现负收益。

### H2 现象可归因

在长度控制与顺序控制后，退化方向仍然成立。

### H3 机理可解释

failure 的主导成分不是 pure miss，而是 `late` 与 `useless` prefetch 上升，使 aggregate reuse 无法转化为 latency 改善。

### H4 趋势可一般化

随着 intra-request mixing 和 cross-request drift 强度增加，utility 退化呈连续或近连续恶化。

## 分阶段实验

### Phase 0: 基线 2x2 + 机制 trace

目的：
先把当前主现象和机制记录链稳定下来。

实验矩阵：

- conditions: `stable_homogeneous`, `shifted_homogeneous`, `stable_mixed`, `shifted_mixed`
- methods: `on_demand`, `prefetch`
- cache ratios: `0.03`, `0.1`, `0.4`

产物：

- `latency_compare.csv`
- `formal_analysis/prefetch_mechanism_metrics.csv`
- `formal_analysis/prefetch_utility_by_layer.csv`
- `formal_analysis/factorial_interaction_effects.csv`

判定：

- `shifted_mixed` 在多个 cache ratio 上持续慢于 on-demand
- `timely_useful_prefetch_ratio` 明显低于 `stable_homogeneous`
- `late_prefetch_ratio` 与 `avg_prefetch_stall_ms` 上升

### Phase 1: 多 seed 重复

目的：
给 2x2 主结果和机制结果补均值、方差和 95% CI。

建议配置：

- seeds: `0 1 2 3 4`

命令入口：

```bash
python scripts/run_prefetch_shift_repeated.py \
  --results_root <results_root> \
  --seeds 0 1 2 3 4 \
  --model_path <model_path> \
  --ckpt_path <ckpt_path> \
  --offload_path <offload_path> \
  --conditions stable_homogeneous shifted_homogeneous stable_mixed shifted_mixed \
  --methods on_demand prefetch \
  --cache_ratios 0.03 0.1 0.4
```

输出：

- `repeat_summary/latency_repeat_summary.csv`
- `repeat_summary/mechanism_repeat_summary.csv`
- `repeat_summary/interaction_repeat_summary.csv`

判定：

- `prefetch_over_on_demand_mean_ci95` 不跨 1 或大多数 seed 方向一致
- `interaction_beta3` 为正且重复出现

### Phase 2: 顺序控制

目的：
排除 “某一种排序方式的人造 artifact”。

建议顺序族：

- `block shift`
- `round-robin`
- `random shuffle`
- `gradual drift`

实现建议：

- 在 `scripts/eval_prefetch_shift.py` 增加 request ordering mode
- 保持 request 集合不变，只改排列顺序

核心指标：

- latency ratio
- expert-set F1
- timely/late/useful/useless breakdown

判定：

- 多种 order mode 下 `shift + mixing` 方向一致地更差

### Phase 3: 长度控制

目的：
排除 token 长度和样本负载强度混杂。

实现建议：

- 为 translation / summarization 各自先建立长度桶
- 构造混合流时，对齐 input token length
- 控制输出长度上限和总 token budget

记录字段：

- `input_token_count`
- `output_token_count`
- `matched_length_bucket`

判定：

- 长度控制后，`shifted_mixed` 退化仍然存在

### Phase 4: mixing / drift 连续化

目的：
证明 failure 不是二元跳变，而是连续趋势。

建议 sweep：

- mixing ratio: `0, 0.25, 0.5, 0.75, 1.0`
- drift major fraction: `0.5, 0.6, 0.7, 0.8, 0.9`

判定：

- `timely_useful_prefetch_ratio` 随 mixing 增强而下降
- `late_prefetch_ratio` 和 latency ratio 随 drift 强度增强而上升

## 关键日志与分析接口

### Runtime trace

`prefetch_trace.tsv` 现在应至少支持这些列：

- `prefetch_issue_id`
- `expert_id`
- `cache_hit`
- `ready_before_consume`
- `stall_time_ms`

说明：

- `CONFUSION` 行给出 predicted vs actual expert-set
- `PREFETCH_EXPERT` 行给出每个 prefetched expert 是否在消费点之前 ready
- 二者通过 `prefetch_issue_id` 关联，进而派生 `timely_useful / late_useful / timely_useless / late_useless`

### 主要分析脚本

- `scripts/analyze_prefetch_confusion.py`
  用于 expert-set confusion / F1 分析
- `scripts/analyze_prefetch_mechanisms.py`
  用于 utility、stall、transition entropy、interaction 汇总
- `scripts/build_prefetch_shift_repeat_summary.py`
  用于多 seed mean/std/CI 汇总

## 图表计划

### 必做图

1. 2x2 latency ratio 主图，带误差条
2. expert-set F1 主图
3. timely / late / useful / useless 堆叠图
4. per-layer utility breakdown
5. latency ratio vs timely useful ratio 散点图

### 强化图

1. mixing ratio sweep 曲线
2. drift intensity sweep 曲线
3. length-controlled 与原始设置并排对照图

## 当前代码对应关系

### 已有

- `scripts/eval_prefetch_shift.py`
  负责生成请求流与单次运行
- `scripts/run_prefetch_shift_formal.py`
  负责 2x2 / ablation 的单 seed 正式跑法

### 本轮新增

- `scripts/run_prefetch_shift_repeated.py`
  负责多 seed 重复实验
- `scripts/build_prefetch_shift_repeat_summary.py`
  负责重复实验的统计汇总
- `scripts/analyze_prefetch_mechanisms.py`
  已扩展到汇总 prefetch utility 指标

## 执行顺序建议

1. 先完成 Phase 0，确认 trace 字段正常落盘
2. 立刻跑 Phase 1，先拿到误差条
3. 再补 Phase 2 和 Phase 3，优先堵 reviewer 最容易打的混杂项
4. 最后做 Phase 4，提升趋势性与一般化表达

## 内部通过标准

满足下面四条时，可以认为“论文地基基本稳住”：

1. `shifted_mixed` 的负收益在多次重复中稳定出现
2. 长度和顺序控制后，退化仍然存在
3. `late/useless` 上升，而不仅是 miss 上升
4. mixing / drift 强度增加时，退化趋势连续存在
