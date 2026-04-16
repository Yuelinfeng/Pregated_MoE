# Cross-Request Shift + Intra-Request Mixing Formal Experiment

## 1. Core Claim

The experiment should not be framed as an artificially difficult prompt set. It should be framed as a reproducible workload class that violates an implicit assumption in MoE expert prefetching:

> Prefetching needs short-horizon expert transition locality, not only aggregate expert reuse.

Existing MoE offloading and prefetch systems exploit sparse activation traces, temporal locality, request-level traces, or next-layer expert predictability. This works when the request stream is stable and each request is drawn from a single task distribution. The target workload breaks that assumption at two levels:

- `cross-request shift`: the stream-level domain distribution changes across adjacent request windows.
- `intra-request mixing`: a single request contains multiple task/domain segments.

The precise claim should be conservative:

> Intra-request mixing already weakens prefetch. Cross-request shift makes the failure more consistent and more pronounced by preventing the prefetcher from adapting to a stable mixed pattern.

Do not claim that prefetch fails only under the combined condition.

## 2. Formal Workload Definition

Let the request stream be:

```text
R = {r_1, r_2, ..., r_T}
```

Each request has two distributions:

```text
q_t(d): request-level domain distribution inside request r_t
P_w(d): stream-level domain distribution inside window w
```

`Intra-request mixing` holds when:

```text
H(q_t) > 0
```

or equivalently, for the controlled two-domain case:

```text
|{d : q_t(d) > 0}| >= 2
```

`Cross-request shift` holds when adjacent windows differ:

```text
D_JS(P_before || P_after) >= delta
```

The main two-domain controlled setting uses `translation` and `summarization`, but the same structure should be repeated for other domain pairs in robustness runs.

## 3. Main 2x2 Factorial Design

The main experiment has four cells:

| Condition | Cross-request shift | Intra-request mixing | Purpose |
|---|---:|---:|---|
| `stable_homogeneous` | No | No | Clean positive control |
| `shifted_homogeneous` | Yes | No | Isolate stream shift |
| `stable_mixed` | No | Yes | Isolate intra-request mixing |
| `shifted_mixed` | Yes | Yes | Combined failure mode |

All four cells should match:

- number of requests
- total token budget as closely as possible
- average prompt length as closely as possible
- global domain marginal distribution
- model, cache ratio, batch size, decoding parameters, cache policy, and hardware

The important control is the global domain marginal. A slowdown should not be explainable by saying one cell simply has more summarization or longer prompts.

Recommended controlled construction:

```text
stable_homogeneous:
  A B A B A B A B ...

shifted_homogeneous:
  80% A + 20% B | 20% A + 80% B

stable_mixed:
  AB(50/50) AB(50/50) AB(50/50) ...

shifted_mixed:
  AB(80/20) AB(80/20) | BA(80/20) BA(80/20)
```

This keeps the overall A/B marginal approximately matched while changing short-term stream distribution and request-internal composition.

## 4. Ratio and Order Controls

The previous `shifted_mixed` construction can be attacked as an order-shift experiment rather than a stream-shift experiment. Therefore, the formal suite includes explicit controls:

| Condition | Meaning |
|---|---|
| `stable_mixed_ab` | Stable mixed requests with order A then B |
| `stable_mixed_ba` | Stable mixed requests with order B then A |
| `stable_mixed_balanced_order` | Stable mixed requests alternating AB and BA |
| `shifted_mixed_ratio_only` | Keep AB order fixed; shift only A/B ratio |
| `shifted_mixed_order_only` | Keep 50/50 ratio fixed; shift only AB vs BA order |
| `shifted_mixed_ratio_and_order` | Shift both ratio and order |

Interpretation rule:

- If `ratio_only` degrades, stream-level domain distribution shift is sufficient.
- If `order_only` degrades, request-internal domain order also breaks the short-horizon transition kernel.
- If `ratio_and_order` is worst, both axes contribute.

## 5. Metrics

Report system outcome and mechanism evidence together.

System metrics:

- latency ratio: `prefetch / on-demand`
- p50/p95 per-request latency ratio
- throughput if available

Cache and prefetch utility metrics:

- average cache hit rate
- useful prefetch precision:

```text
useful_prefetch_precision = TP / (TP + FP)
```

- expert-set recall:

```text
expert_set_recall = TP / (TP + FN)
```

- expert-set F1
- wasted prefetch experts: `FP`
- wasted prefetch bytes:

```text
wasted_prefetch_bytes = FP * bytes_per_expert
```

The current trace can compute all of the above. True `prefetch-induced eviction count` requires an additional cache-eviction event in the C++ cache path; until that is logged, report wasted bytes and missed actual experts as directly measured proxies, not as eviction counts.

Transition-locality metrics:

- transition entropy
- transition-kernel Jensen-Shannon divergence

For each decoder MoE transition:

```text
T_l(i -> j) = P(expert j at layer l+1 | expert i at layer l)
```

Compare:

```text
D_JS(T_stable_homogeneous || T_shifted_mixed)
D_JS(T_stable_mixed || T_shifted_mixed)
```

This separates aggregate expert hotness from short-horizon expert transition locality.

## 6. Interaction Effect

Use the 2x2 contrast per cache ratio:

```text
beta_3 =
  latency(shifted_mixed)
- latency(shifted_homogeneous)
- latency(stable_mixed)
+ latency(stable_homogeneous)
```

If `beta_3 > 0` with confidence intervals across seeds, say:

> shift and mixing interactively amplify prefetch degradation.

If `beta_3` is not significant, use the more conservative interpretation:

> mixing is the dominant stressor, and shift makes the degradation persistent across cache budgets.

## 7. Recommended Runs

For each condition, run both `on_demand` and `prefetch`.

Main cache ratios:

```text
0.03, 0.1, 0.4
```

Main seeds:

```text
0, 1, 2
```

Main workload commands should use:

```text
--num_requests 128
--shift_block_size 32
--domain_order translation,summarization
--match_domain_marginal
--homogeneous_word_budget 256
--mix_word_budget 256
--stable_mix_fraction 0.5
--shift_major_fraction 0.8
--mix_mode interleave
--interleave_chunk_words 16
```

For workload construction sanity checks, use:

```text
--write_trace_only
```

Each run writes:

- `request_trace.jsonl`
- `workload_summary.json`
- `request_metrics.jsonl`
- `prefetch_trace.tsv`

Then aggregate confusion with:

```text
python scripts/analyze_prefetch_confusion.py \
  --trace_path <run_dir>/prefetch_trace.tsv \
  --request_metrics_path <run_dir>/request_metrics.jsonl \
  --boundary_window 8 \
  --output_csv <run_dir>/confusion_boundary8.csv
```

After all runs are arranged under:

```text
<results_root>/prefetch/<condition>/cr<cache_ratio>/
```

run:

```text
python scripts/analyze_prefetch_mechanisms.py \
  --results_root <results_root> \
  --output_dir <results_root>/formal_analysis \
  --expert_bytes 18874368
```

This produces:

- `prefetch_mechanism_metrics.csv`
- `factorial_interaction_effects.csv`
- `transition_entropy.csv`
- `transition_kernel_divergence.csv`

## 8. Main Figures

Recommended main-paper figures:

1. Workload construction diagram:

```text
Stable homogeneous:  A B A B A B A B
Shifted homogeneous: 80%A/20%B | 20%A/80%B
Stable mixed:        AB AB AB AB
Shifted mixed:       AB-heavy | BA-heavy
```

2. 2x2 heatmap:

- left: latency ratio
- right: expert-set F1 or useful prefetch precision

3. Mechanism breakdown:

- cache hit rate
- useful prefetch precision
- wasted prefetch bytes
- transition entropy or transition-kernel divergence

The most important phrase for the paper is:

> high-hit, low-utility prefetch failure

This emphasizes that high cache hit rate alone is not sufficient evidence that prefetching is useful.

## 9. Current Result Interpretation

The current `switch-base-64` results already support the conservative claim:

```text
shifted_mixed:
0.03 -> 1.023x
0.1  -> 1.021x
0.4  -> 1.048x
```

At the same time, cache hit rates remain high:

```text
0.03 -> 87.7%
0.1  -> 94.9%
0.4  -> 96.1%
```

This should be described as:

> Under shifted mixed workloads, prefetching becomes slower than on-demand across all tested cache ratios despite high cache hit rates. The failure is therefore not a simple capacity-miss problem; it is a high-hit, low-utility prefetch failure caused by unreliable short-horizon expert transitions.

## 10. Robustness Priority

Minimum robustness queue:

1. seeds
2. domain pairs
3. shift block size
4. mix ratio
5. second model
6. second hardware

Concrete settings:

```text
shift_block_size: 8, 16, 32, 64
mix ratio: 25/75, 50/50, 75/25
domain pair: translation/summarization, QA/summarization, code/explanation
model: switch-base-64, switch-large-128
```

If resources are limited, prioritize seeds and domain pairs first.
