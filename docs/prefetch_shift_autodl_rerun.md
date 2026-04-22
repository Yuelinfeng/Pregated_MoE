# AutoDL Rerun Commands for Formal Prefetch-Shift Experiment

The new recommended entry point is:

```text
scripts/run_prefetch_shift_formal.py
```

It does the following for every cell:

1. runs `eval_prefetch_shift.py`
2. verifies the cell is complete
3. runs `analyze_prefetch_confusion.py` for prefetch cells
4. retries failed or incomplete cells
5. validates the full result grid
6. rebuilds `latency_compare.csv`
7. regenerates mechanism analysis tables
8. regenerates figures

This prevents incomplete cells such as a partially written `request_metrics.jsonl` from silently entering the final tables.

## 1. Environment

```bash
cd /root/autodl-tmp/Pregated_MoE

export REPO=/root/autodl-tmp/Pregated_MoE
export BUILD=$REPO/build
export MODEL_PATH=google/switch-base-64
export CKPT_PATH=/root/autodl-tmp/ft/switch-base-64
export OFFLOAD_PATH=/root/autodl-tmp/ft/switch-base-64
export LIB_PATH=$BUILD/lib/libth_transformer.so

export HF_HOME=/root/autodl-tmp/hf_cache
export HF_DATASETS_CACHE=/root/autodl-tmp/hf_cache/datasets
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=$BUILD/lib:$LD_LIBRARY_PATH

pip install -U pandas seaborn matplotlib datasets "pyarrow<21"
```

## 1.5 Rebuild native library after the fetcher fix

The formal runner and validation scripts are pure Python, but the crash observed at
`fetcher.cc:217` comes from the native FasterTransformer library. After updating
`src/fastertransformer/utils/fetcher.h` and `src/fastertransformer/utils/fetcher.cc`,
rebuild `libth_transformer.so` before rerunning the experiment.

```bash
cd /root/autodl-tmp/Pregated_MoE

cmake --build build -j"$(nproc)"

ls -lh /root/autodl-tmp/Pregated_MoE/build/lib/libth_transformer.so
```

If your AutoDL setup uses a separate build directory, rerun the same CMake configure
command you used originally and then rebuild.

## 2. Main 2x2 formal run

```bash
python scripts/run_prefetch_shift_formal.py \
  --results_root /root/autodl-tmp/pregated_formal_runs/main_seed0 \
  --model_path "$MODEL_PATH" \
  --ckpt_path "$CKPT_PATH" \
  --offload_path "$OFFLOAD_PATH" \
  --lib_path "$LIB_PATH" \
  --conditions stable_homogeneous shifted_homogeneous stable_mixed shifted_mixed \
  --cache_ratios 0.03 0.1 0.4 \
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

Background mode:

```bash
nohup python scripts/run_prefetch_shift_formal.py \
  --results_root /root/autodl-tmp/pregated_formal_runs/main_seed0 \
  --model_path "$MODEL_PATH" \
  --ckpt_path "$CKPT_PATH" \
  --offload_path "$OFFLOAD_PATH" \
  --lib_path "$LIB_PATH" \
  --conditions stable_homogeneous shifted_homogeneous stable_mixed shifted_mixed \
  --cache_ratios 0.03 0.1 0.4 \
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
  --max_retries 3 \
  > /root/autodl-tmp/pregated_formal_runs/main_seed0_runner.log 2>&1 &

tail -f /root/autodl-tmp/pregated_formal_runs/main_seed0_runner.log
```

## 3. Ratio / order ablation

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

After the ablation run completes, build the matched-effect summary:

```bash
python scripts/analyze_ratio_order_ablation.py \
  --results_root /root/autodl-tmp/pregated_formal_runs/ratio_order_ablation_seed0
```

This writes:

```text
/root/autodl-tmp/pregated_formal_runs/ratio_order_ablation_seed0/ablation_analysis/ratio_order_ablation_summary.csv
/root/autodl-tmp/pregated_formal_runs/ratio_order_ablation_seed0/ablation_analysis/ratio_order_ablation_effects.csv
```

Use `ratio_order_ablation_effects.csv` to answer:

- does ratio shift alone degrade prefetch?
- does order shift alone degrade prefetch?
- is the combined ratio+order condition stronger than either single axis?

## 4. Multi-seed rerun

```bash
for SEED in 0 1 2; do
  python scripts/run_prefetch_shift_formal.py \
    --results_root /root/autodl-tmp/pregated_formal_runs/main_seed${SEED} \
    --model_path "$MODEL_PATH" \
    --ckpt_path "$CKPT_PATH" \
    --offload_path "$OFFLOAD_PATH" \
    --lib_path "$LIB_PATH" \
    --conditions stable_homogeneous shifted_homogeneous stable_mixed shifted_mixed \
    --cache_ratios 0.03 0.1 0.4 \
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
    --seed ${SEED} \
    --max_retries 3
done
```

## 5. Post-run spot checks

The runner already validates the full result grid, but these are still useful quick checks:

```bash
cat /root/autodl-tmp/pregated_formal_runs/main_seed0/validate.log
cat /root/autodl-tmp/pregated_formal_runs/main_seed0/build_latency_compare.log
ls -lh /root/autodl-tmp/pregated_formal_runs/main_seed0/formal_analysis
ls -lh /root/autodl-tmp/pregated_formal_runs/main_seed0/figures
```

Each completed cell should contain:

```text
request_trace.jsonl
workload_summary.json
request_metrics.jsonl
run_status.json
```

and each prefetch cell should additionally contain:

```text
prefetch_trace.tsv
confusion_boundary8.csv
confusion_boundary8.json
```
