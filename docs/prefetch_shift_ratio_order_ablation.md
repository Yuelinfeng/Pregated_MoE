# Ratio / Order Ablation for Prefetch-Shift

## 1. Why this ablation is necessary

The main 2x2 experiment establishes that `shifted_mixed` is a robust stress case, but it still leaves an obvious reviewer question:

> Is the slowdown caused by cross-request workload shift, or simply by changing the internal order of the mixed request?

Without an explicit ratio/order ablation, a reviewer can argue that `shifted_mixed` is not isolating the intended mechanism. In particular:

- if `stable_mixed` is always `AB`
- and `shifted_mixed` becomes `BA` in later phases

then the experiment is changing **two things at once**:

1. the stream-level domain ratio across phases
2. the request-internal domain order

That makes it impossible to say whether degradation comes from:

- ratio shift alone,
- order shift alone,
- or their combination.

This ablation removes that ambiguity and closes the causal story.

## 2. Design idea

We keep the same two-domain pair (`translation`, `summarization`) and split the mixed-condition family into six cells:

| Condition | What is controlled | What changes |
|---|---|---|
| `stable_mixed_ab` | Stable stream, fixed `AB` order | Nothing shifts |
| `stable_mixed_ba` | Stable stream, fixed `BA` order | Nothing shifts |
| `stable_mixed_balanced_order` | Stable stream, alternating `AB` and `BA`, global order marginal balanced | Nothing shifts over time |
| `shifted_mixed_ratio_only` | Keep `AB` order fixed | Only the stream-level domain ratio changes across phases |
| `shifted_mixed_order_only` | Keep the request-level ratio fixed at `50/50` | Only `AB` vs `BA` order changes across phases |
| `shifted_mixed_ratio_and_order` | No extra restriction | Ratio and order both shift |

The recommended comparisons are:

- `shifted_mixed_ratio_only` vs `stable_mixed_ab`
- `shifted_mixed_order_only` vs `stable_mixed_balanced_order`
- `shifted_mixed_ratio_and_order` vs `stable_mixed_balanced_order`

These baselines are chosen so that each comparison changes only the intended axis:

- `ratio_only`: same request order, different stream ratio
- `order_only`: same 50/50 request ratio, different order schedule
- `ratio_and_order`: both axes move together

## 3. What this ablation validates

This ablation lets us answer three reviewer-facing questions precisely.

### A. Does stream-level ratio shift alone hurt prefetch?

Compare:

```text
shifted_mixed_ratio_only
vs
stable_mixed_ab
```

If this pair degrades, then **cross-request workload shift by itself** is sufficient to harm prefetch utility, even when request order is fixed.

### B. Does request-internal order shift alone hurt prefetch?

Compare:

```text
shifted_mixed_order_only
vs
stable_mixed_balanced_order
```

If this pair degrades, then **short-horizon expert transition locality depends on domain order inside the mixed request**, not just aggregate domain content.

### C. Is the combined stressor stronger than either single axis?

Compare:

```text
shifted_mixed_ratio_and_order
vs
max(shifted_mixed_ratio_only, shifted_mixed_order_only)
```

If the combined condition is worst, then the strongest statement is:

> ratio shift and order shift both contribute, and their combination creates the most difficult regime for prefetching.

If one single-axis ablation dominates, use the more conservative wording:

> the main failure is primarily explained by ratio shift

or

> the main failure is primarily explained by order-sensitive transition changes.

## 4. What to report

For each ablation cell, report:

- latency ratio: `prefetch / on-demand`
- mean and p95 latency delta
- cache hit rate
- useful prefetch precision
- expert-set F1
- wasted prefetch bytes

Then produce an ablation summary table with three effect rows:

- `ratio_shift_effect`
- `order_shift_effect`
- `combined_ratio_and_order_effect`

and one comparison row:

- `combined_minus_strongest_single`

The sign of `latency_ratio_delta` is the main readout:

- `> 0`: the shifted variant is worse than its matched stable control
- near `0`: the axis alone does not materially hurt
- `< 0`: the axis may not be the true source of degradation

## 5. AutoDL command

Run the ablation with the same formal runner:

```bash
python scripts/run_prefetch_shift_formal.py \
  --results_root /root/autodl-tmp/pregated_formal_runs/ratio_order_ablation_seed0 \
  --model_path "$MODEL_PATH" \
  --ckpt_path "$CKPT_PATH" \
  --offload_path "$OFFLOAD_PATH" \
  --lib_path "$LIB_PATH" \
  --conditions stable_mixed_ab stable_mixed_ba stable_mixed_balanced_order shifted_mixed_ratio_only shifted_mixed_order_only shifted_mixed_ratio_and_order \
  --cache_ratios 0.1 \
  --num_requests 128 \
  --shift_block_size 32 \
  --domain_order translation,summarization \
  --homogeneous_word_budget 256 \
  --mix_word_budget 256 \
  --stable_mix_fraction 0.5 \
  --shift_major_fraction 0.8 \
  --mix_mode interleave \
  --interleave_chunk_words 16 \
  --beam_width 1 \
  --max_seq_len 128 \
  --sampling_topk 1 \
  --sampling_topp 0.0 \
  --moe_topk 1 \
  --data_type fp32 \
  --tensor_para_size 1 \
  --pipeline_para_size 1 \
  --cache_policy LFU \
  --seed 0 \
  --max_retries 3
```

Then summarize it with:

```bash
python scripts/analyze_ratio_order_ablation.py \
  --results_root /root/autodl-tmp/pregated_formal_runs/ratio_order_ablation_seed0
```

Generate the single-summary figure with:

```bash
python scripts/plot_ratio_order_ablation_summary.py \
  --results_root /root/autodl-tmp/pregated_formal_runs/ratio_order_ablation_seed0 \
  --output_dir /root/autodl-tmp/pregated_formal_runs/ratio_order_ablation_seed0/figures
```

This writes:

- `ablation_analysis/ratio_order_ablation_summary.csv`
- `ablation_analysis/ratio_order_ablation_effects.csv`
- `figures/ratio_order_ablation_summary.png`

## 6. Recommended wording for the paper

If both single-axis ablations degrade, and the combined one is worst:

> The ratio/order ablation shows that both stream-level ratio shift and request-internal order shift contribute to prefetch degradation. The combined ratio-and-order condition is the most harmful, indicating that the prefetcher depends on both stable domain proportions across requests and stable transition order within mixed requests.

If only one single-axis ablation is strong:

> The ratio/order ablation indicates that the observed degradation is primarily driven by <ratio shift / order shift>, while the other axis has limited standalone effect.
