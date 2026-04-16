import configparser
import subprocess
import pandas as pd
import re
import math
from cpuinfo import get_cpu_info
import psutil
import torch
import argparse


FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|nan|inf"
ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


def normalize_output(output: str):
    output = output.replace("\r\n", "\n").replace("\r", "\n")
    return ANSI_ESCAPE_RE.sub("", output)


def parse_float(value: str):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def parse_output(output: str):
    output = normalize_output(output)

    block_lat = math.nan
    for line in reversed(output.splitlines()):
        m = re.search(rf"BLK AVG: ({FLOAT_RE}) ms", line)
        if m:
            block_lat = parse_float(m[1])
            break

    throughput = math.nan
    for line in reversed(output.splitlines()):
        m = re.search(rf", ({FLOAT_RE}) tokens/sec\.", line)
        if m:
            throughput = parse_float(m[1])
            break

    peak_mem_encoder = math.nan
    peak_mem_decoder = math.nan
    for line in output.splitlines():
        if "MEM usage:" not in line:
            continue
        mem_values = [int(v) for v in re.findall(r"\d+", line.split("MEM usage:", 1)[1])]
        if mem_values:
            peak_mem_encoder = mem_values[0]
            peak_mem_decoder = max(mem_values[1:], default=mem_values[0])
            break

    max_active_experts = math.nan
    for line in output.splitlines():
        m = re.search(r"Max active experts: (\d+)", line)
        if m:
            max_active_experts = int(m[1])
            break

    avg_active_experts = math.nan
    for line in output.splitlines():
        m = re.search(rf"Average active experts: ({FLOAT_RE})", line)
        if m:
            avg_active_experts = parse_float(m[1])
            break

    cache_hit_rate = math.nan
    for line in output.splitlines():
        m = re.search(rf"Average cache hit rate: ({FLOAT_RE})", line)
        if m:
            cache_hit_rate = parse_float(m[1])
            break

    return {
        "block_lat": block_lat,
        "throughput": throughput,
        "peak_mem_encoder": peak_mem_encoder,
        "peak_mem_decoder": peak_mem_decoder,
        "max_active_expert": max_active_experts,
        "avg_active_expert": avg_active_experts,
        "cache_hit_rate": cache_hit_rate,
    }


def is_output_valid(output: str):
    output = normalize_output(output)
    has_block_lat = re.search(rf"BLK AVG: ({FLOAT_RE}) ms", output) is not None
    has_throughput = re.search(rf", ({FLOAT_RE}) tokens/sec\.", output) is not None
    return has_block_lat or has_throughput


def compute_arena_size_bytes(method, size_per_expert, total_experts, num_layer, cache_ratio):
    if method == "GPU-only":
        return 0, 0

    if cache_ratio and cache_ratio > 0:
        expert_slots = max(round(num_layer * total_experts * cache_ratio), 1)
        return expert_slots * size_per_expert, 1

    # With cache disabled, the arena is still used as a staging buffer for fetched experts.
    # The original 20 GiB default is too aggressive for 24 GiB consumer GPUs.
    if method == "Pre-gated":
        expert_slots = 16
    elif method == "DeepSpeed":
        expert_slots = 8
    elif method == "SE-MoE":
        expert_slots = total_experts
    else:
        expert_slots = 1

    expert_slots = max(1, min(expert_slots, total_experts))
    return expert_slots * size_per_expert, 0


def profile_config(cpp_config, model, method, batch_size, forced_num_expert=0, cache_ratio=0, disk_offload=0):
    iterations = 4
    exp_name = f"{model}_{method}_{batch_size}_{forced_num_expert}_{cache_ratio}_{disk_offload}"
    print(f"Running {exp_name}")
    if method == "GPU-only":
        encoder_fetcher_mode = "0"
        decoder_fetcher_mode = "0"
    elif method == "Pre-gated":
        encoder_fetcher_mode = "1"
        decoder_fetcher_mode = "2"
    elif method == "DeepSpeed":
        encoder_fetcher_mode = "1"
        decoder_fetcher_mode = "1"
    elif method == "SE-MoE":
        encoder_fetcher_mode = "1"
        decoder_fetcher_mode = "2"
        iterations = 1

    if "base" in model:
        size_per_expert = 18874368
        num_layer = 6
    elif "large" in model:
        size_per_expert = 33554432
        num_layer = 12

    total_experts = int(re.search(r"\d+", model)[0])

    arena_size, use_cache = compute_arena_size_bytes(
        method, size_per_expert, total_experts, num_layer, cache_ratio
    )

    cpp_config["default"] = {
        "arena_size": f"{arena_size}",
        "encoder_fetcher_mode": encoder_fetcher_mode,
        "decoder_fetcher_mode": decoder_fetcher_mode,
        "profiling": "1",
        "detailed_timing": "0",
        "offload_path": f"/data/ft/{model}/",
        "disk_offload": f"{disk_offload}",
        "load_from_cpp": "1",
        "use_cache": f"{use_cache}",
        "quant_mode": "0",
        "vocab_size": "32128",
        "fetch_all": f"{int(method == 'SE-MoE')}",
        "forced_num_experts": f"{forced_num_expert}",
        "cache_policy": "LFU",
    }

    with open("/workspace/FasterTransformer/cpp_config.ini", "w") as fp:
        cpp_config.write(fp)

    command = (
        f"python /workspace/FasterTransformer/examples/pytorch/t5/perf_benchmark.py "
        f"--batch_size {batch_size} "
        f"--beam_width 4 "
        f"--seq_len 256 "
        f"--data_type fp32 "
        f"--test_time 3 "
        f"--sampling_topk 1 "
        f"--model_type Megatron-DeepSpeed "
        f"--ckpt_path /data/ft/{model}/ "
        f"--model t5-base "
        f"--duration 0 "
        f"--iterations {iterations} "
    )

    print(command)

    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        cwd="/workspace/FasterTransformer/build"
    )

    combined_output = result.stdout
    if result.stderr:
        combined_output += ("\n" if combined_output and not combined_output.endswith("\n") else "") + result.stderr

    with open(f"/workspace/FasterTransformer/logs/{exp_name}.log", "w") as fp:
        fp.write("=== STDOUT ===\n")
        fp.write(result.stdout)
        fp.write("\n=== STDERR ===\n")
        fp.write(result.stderr)

    if result.returncode != 0:
        print(f"[WARN] Command failed with return code {result.returncode}. See logs/{exp_name}.log")
        return {
            "block_lat": math.nan,
            "throughput": math.nan,
            "peak_mem_encoder": math.nan,
            "peak_mem_decoder": math.nan,
            "peak_mem": math.nan,
            "max_active_expert": math.nan,
            "avg_active_expert": math.nan,
            "cache_hit_rate": math.nan,
        }

    parsed = parse_output(combined_output)

    if not is_output_valid(combined_output):
        print(f"[WARN] Failed to parse benchmark output. See logs/{exp_name}.log")
        return {
            "block_lat": math.nan,
            "throughput": math.nan,
            "peak_mem_encoder": math.nan,
            "peak_mem_decoder": math.nan,
            "peak_mem": math.nan,
            "max_active_expert": math.nan,
            "avg_active_expert": math.nan,
            "cache_hit_rate": math.nan,
        }

    peak_mem = math.nan
    if not math.isnan(parsed["peak_mem_decoder"]) and parsed["peak_mem_decoder"] > 0:
        max_active_expert = parsed["max_active_expert"]
        if method == "GPU-only":
            peak_mem = parsed["peak_mem_decoder"]
        else:
            if method == "Pre-gated":
                used_buffer = 2 * max_active_expert if not math.isnan(max_active_expert) else math.nan
            elif method == "DeepSpeed":
                used_buffer = max_active_expert
            elif method == "SE-MoE":
                used_buffer = 2 * total_experts
            else:
                used_buffer = math.nan
            if not math.isnan(used_buffer):
                peak_mem = parsed["peak_mem_decoder"] - arena_size - size_per_expert * (2 * total_experts - used_buffer)

    print(
        f"BLK AVG: {parsed['block_lat']} ms, "
        f"throughput: {parsed['throughput']} tokens/sec, "
        f"peak_mem_encoder: {parsed['peak_mem_encoder']}, "
        f"peak_mem_decoder: {parsed['peak_mem_decoder']}, "
        f"max_active_experts: {parsed['max_active_expert']}, "
        f"avg_active_experts: {parsed['avg_active_expert']}, "
        f"peak_mem: {peak_mem}, "
        f"cache_hit_rate: {parsed['cache_hit_rate']}"
    )

    return {
        "block_lat": parsed["block_lat"],
        "throughput": parsed["throughput"],
        "peak_mem_encoder": parsed["peak_mem_encoder"],
        "peak_mem_decoder": parsed["peak_mem_decoder"],
        "peak_mem": peak_mem,
        "max_active_expert": parsed["max_active_expert"],
        "avg_active_expert": parsed["avg_active_expert"],
        "cache_hit_rate": parsed["cache_hit_rate"],
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--re_run", action="store_true")
    parser.add_argument("--name", type=str, default="")
    parser.set_defaults(rerun=False, use_cache=False)
    return parser.parse_args()


def main():
    args = parse_args()

    models = [
        "switch-base-8",
        "switch-base-64",
        "switch-base-128",
        "switch-large-128",
    ]
    batch_sizes = [
        1,
        # 2,
        # 4,
        # 8,
        # 16,
    ]
    methods = [
        "GPU-only",
        "Pre-gated",
        "DeepSpeed",
        "SE-MoE",
    ]
    metrics = [
        "block_lat",
        "throughput",
        "peak_mem_encoder",
        "peak_mem_decoder",
        "peak_mem",
        "max_active_expert",
        "avg_active_expert",
        "cache_hit_rate",
    ]
    forced_num_experts = [
        0,
        # 1,
        # 2,
        # 4,
        # 8,
        # 16,
    ]
    cache_ratios = [
        0,
        # 0.01,
        # 0.03,
        # 0.05,
        # 0.1,
        # 0.2,
        # 0.4,
        # 0.8,
    ]
    disk_offloads = [
        0,
        # 1,
    ]

    cpp_config = configparser.ConfigParser()
    cpp_config.read("/workspace/FasterTransformer/cpp_config.ini")

    hardware_infos = {
        "CPU": [get_cpu_info()["brand_raw"]] * len(batch_sizes),
        "RAM (GB)": [int(psutil.virtual_memory().total / 1024 / 1024 / 1024)] * len(batch_sizes),
        "GPU": [torch.cuda.get_device_name()] * len(batch_sizes),
    }

    results = {}
    for metric in metrics:
        results[metric] = {
            "bs": batch_sizes,
            "active experts": forced_num_experts,
            "cache ratio": cache_ratios,
        }
        results[metric].update(hardware_infos)

        for key, value in results[metric].items():
            if len(value) == 1:
                results[metric][key] = value * max(len(forced_num_experts), len(batch_sizes), len(cache_ratios))

    def gen_csv_name(metric):
        return f"{args.name}{'_' if args.name else ''}{metric}s.csv"

    if not args.re_run:
        for method in methods:
            for model in models:
                for disk_offload in disk_offloads:
                    records = []

                    for batch_size in batch_sizes:
                        for forced_num_expert in forced_num_experts:
                            for cache_ratio in cache_ratios:
                                records.append(
                                    profile_config(
                                        cpp_config,
                                        model,
                                        method,
                                        batch_size,
                                        forced_num_expert,
                                        cache_ratio,
                                        disk_offload,
                                    )
                                )

                    for metric, result in results.items():
                        result[f"{model}/{method}/{'SSD' if disk_offload else 'CPU'}"] = [
                            record[metric] for record in records
                        ]

                    # Generate CSV after each model and method runned
                    for metric, result in results.items():
                        df = pd.DataFrame.from_dict(result)
                        df.to_csv(f"{gen_csv_name(metric)}", index=False)

    else:
        dfs = {metric: pd.read_csv(f"/workspace/FasterTransformer/performance_data/{gen_csv_name(metric)}") for metric in metrics}
        models = [
            "switch-base-8",
            # "switch-base-64",
            # "switch-base-128",
            # "switch-large-128",
        ]
        batch_sizes = [
            1,
            # 2,
            # 4,
            # 8,
            # 16,
        ]
        methods = [
            "GPU-only",
            # "Pre-gated",
            # "DeepSpeed",
            # "SE-MoE",
        ]
        rerun_configs = [
            (model, method, batch_size)
            for model in models
            for method in methods
            for batch_size in batch_sizes
        ]
        for model, method, batch_size in rerun_configs:
            row_idx = batch_sizes.index(batch_size)
            col_idx = "{}/{}".format(model, method)
            record = profile_config(cpp_config, model, method, batch_size)
            for metric, df in dfs.items():
                df.loc[row_idx, col_idx] = record[metric]
                df.to_csv(f"{gen_csv_name(metric)}", index=False)


if __name__ == "__main__":
    main()
