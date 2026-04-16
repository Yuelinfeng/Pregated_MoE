import argparse
import configparser
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, T5Config


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from examples.pytorch.decoding.utils.recover_bpe import recover_bpe
from examples.pytorch.t5.utils.ft_decoding import FTT5, FTT5Decoding, FTT5DecodingWeight
from examples.pytorch.t5.utils.ft_encoder import FTT5Encoder, FTT5EncoderWeight


METHOD_TO_FETCHER_MODE = {
    "gpu_only": ("0", "0"),
    "on_demand": ("1", "1"),
    "prefetch": ("1", "2"),
}

CONDITION_CHOICES = [
    "stable_homogeneous",
    "shifted_homogeneous",
    "stable_mixed",
    "stable_mixed_ab",
    "stable_mixed_ba",
    "stable_mixed_balanced_order",
    "shifted_mixed",
    "shifted_mixed_ratio_only",
    "shifted_mixed_order_only",
    "shifted_mixed_ratio_and_order",
]

SHIFTED_MIXED_VARIANT_BY_CONDITION = {
    "shifted_mixed": "ratio_and_order",
    "shifted_mixed_ratio_only": "ratio_only",
    "shifted_mixed_order_only": "order_only",
    "shifted_mixed_ratio_and_order": "ratio_and_order",
}


@dataclass
class RequestSpec:
    request_id: str
    condition: str
    domain_label: str
    text: str
    metadata: Dict[str, object]


def parse_layer_index_list(raw_value: str) -> List[int]:
    raw_value = raw_value.strip()
    if raw_value in ("", "[]"):
        return []
    return [int(token) for token in raw_value[1:-1].replace(" ", "").split(",") if token]


def ensure_trailing_sep(path_value: str) -> str:
    return path_value if path_value.endswith("/") else path_value + "/"


def read_ft_config(ckpt_path: Path) -> configparser.ConfigParser:
    ckpt_config_path = ckpt_path / "config.ini"
    if not ckpt_config_path.is_file():
        raise FileNotFoundError(f"FT checkpoint config not found: {ckpt_config_path}")

    ckpt_config = configparser.ConfigParser()
    ckpt_config.read(ckpt_config_path)
    return ckpt_config


def build_t5_config(section: configparser.SectionProxy, *, decoder: bool) -> T5Config:
    config = T5Config(
        vocab_size=section.getint("vocab_size"),
        d_model=section.getint("d_model"),
        d_kv=section.getint("d_kv"),
        d_ff=section.getint("d_ff"),
        num_layers=section.getint("num_layers"),
        num_decoder_layers=section.getint("num_decoder_layers", fallback=section.getint("num_layers")),
        num_heads=section.getint("num_heads"),
        relative_attention_num_buckets=section.getint("relative_attention_num_buckets_or_max_pos_seq_len"),
        feed_forward_proj=section.get("feed_forward_proj", fallback="relu"),
        pad_token_id=section.getint("pad_token_id", fallback=0),
        eos_token_id=section.getint("eos_token_id", fallback=1),
        decoder_start_token_id=section.getint("decoder_start_token_id", fallback=0) if decoder else None,
        is_gated_act=section.getboolean("is_gated_act", fallback=False),
    )
    return config


def build_runtime(
    args: argparse.Namespace,
    ckpt_config: configparser.ConfigParser,
) -> Tuple[AutoTokenizer, FTT5]:
    if args.model_type not in ("Megatron", "Megatron-DeepSpeed"):
        raise ValueError("This experiment script currently supports Megatron and Megatron-DeepSpeed checkpoints only.")

    encoder_config = build_t5_config(ckpt_config["encoder"], decoder=False)
    decoder_config = build_t5_config(ckpt_config["decoder"], decoder=True)

    t5_with_bias = ckpt_config.getboolean("structure", "t5_with_bias")
    position_embedding_type = 0 if ckpt_config.get("structure", "position_embedding_type") == "relative" else 1
    activation_type = encoder_config.feed_forward_proj
    tie_word_embeddings = ckpt_config.getboolean("decoder", "tie_word_embeddings", fallback=True)
    use_gated_activation = encoder_config.is_gated_act
    t5_with_moe = ckpt_config.getint("structure", "t5_with_moe", fallback=0) == 1

    encoder_config.update(
        {
            "num_experts": ckpt_config.getint("encoder", "num_experts", fallback=0),
            "moe_layer_index": parse_layer_index_list(ckpt_config.get("structure", "moe_layers_in_encoder", fallback="[]")),
        }
    )
    decoder_config.update(
        {
            "num_experts": ckpt_config.getint("decoder", "num_experts", fallback=0),
            "moe_layer_index": parse_layer_index_list(ckpt_config.get("structure", "moe_layers_in_decoder", fallback="[]")),
        }
    )

    weight_data_type = {"fp16": np.float16, "fp32": np.float32}[ckpt_config.get("encoder", "weight_data_type")]
    q_scaling = 1.0

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    ft_encoder_weight = FTT5EncoderWeight(
        encoder_config,
        args.tensor_para_size,
        args.pipeline_para_size,
        t5_with_bias=t5_with_bias,
        use_gated_activation=use_gated_activation,
        t5_with_moe=t5_with_moe,
        position_embedding_type=position_embedding_type,
        weight_data_type=weight_data_type,
    )
    ft_decoding_weight = FTT5DecodingWeight(
        decoder_config,
        args.tensor_para_size,
        args.pipeline_para_size,
        t5_with_bias=t5_with_bias,
        use_gated_activation=use_gated_activation,
        t5_with_moe=t5_with_moe,
        position_embedding_type=position_embedding_type,
        weight_data_type=weight_data_type,
    )

    # Real expert weights are loaded by the C++ side when load_from_cpp=1, so
    # tiny placeholders are enough here and avoid allocating giant MoE tensors.
    ft_encoder_weight.empty_weights()
    ft_decoding_weight.empty_weights()

    if args.data_type == "fp16":
        ft_encoder_weight.to_half()
        ft_decoding_weight.to_half()

    remove_padding = True
    ft_encoder = FTT5Encoder(
        ft_encoder_weight.w,
        args.lib_path,
        encoder_config.num_heads,
        encoder_config.d_kv,
        encoder_config.d_ff,
        encoder_config.d_model,
        remove_padding,
        encoder_config.num_layers,
        encoder_config.relative_attention_num_buckets,
        encoder_config.num_experts,
        encoder_config.moe_layer_index,
        128,
        False,
        q_scaling,
        args.tensor_para_size,
        args.pipeline_para_size,
        t5_with_bias,
        position_embedding_type,
        moe_k=args.moe_topk,
        activation_type=activation_type,
    )
    ft_decoding = FTT5Decoding(
        ft_decoding_weight.w,
        args.lib_path,
        decoder_config.num_heads,
        decoder_config.d_kv,
        decoder_config.d_ff,
        encoder_config.d_model,
        decoder_config.d_model,
        decoder_config.num_layers,
        decoder_config.decoder_start_token_id,
        decoder_config.eos_token_id,
        decoder_config.vocab_size,
        q_scaling,
        decoder_config.relative_attention_num_buckets,
        decoder_config.num_experts,
        decoder_config.moe_layer_index,
        max_distance=128,
        tensor_para_size=args.tensor_para_size,
        pipeline_para_size=args.pipeline_para_size,
        t5_with_bias=t5_with_bias,
        position_embedding_type=position_embedding_type,
        moe_k=args.moe_topk,
        activation_type=activation_type,
        tie_word_embeddings=tie_word_embeddings,
    )
    return tokenizer, FTT5(ft_encoder, ft_decoding)


def compute_arena_size_bytes(args: argparse.Namespace, ckpt_config: configparser.ConfigParser) -> int:
    if args.arena_size_bytes is not None:
        return args.arena_size_bytes

    if args.cache_ratio <= 0:
        return args.default_arena_size_bytes

    num_experts = ckpt_config.getint("decoder", "num_experts", fallback=0)
    if num_experts == 0:
        return args.default_arena_size_bytes

    d_model = ckpt_config.getint("decoder", "d_model")
    d_ff = ckpt_config.getint("decoder", "d_ff")
    moe_layers_encoder = parse_layer_index_list(ckpt_config.get("structure", "moe_layers_in_encoder", fallback="[]"))
    moe_layers_decoder = parse_layer_index_list(ckpt_config.get("structure", "moe_layers_in_decoder", fallback="[]"))
    total_moe_layers = len(moe_layers_encoder) + len(moe_layers_decoder)
    size_per_expert = 2 * d_model * d_ff * 4
    return max(round(total_moe_layers * num_experts * args.cache_ratio), 1) * size_per_expert


def write_cpp_config(args: argparse.Namespace, ckpt_config: configparser.ConfigParser, config_path: Path) -> None:
    encoder_fetcher_mode, decoder_fetcher_mode = METHOD_TO_FETCHER_MODE[args.method]
    cpp_config = configparser.ConfigParser()
    cpp_config["default"] = {
        "arena_size": str(compute_arena_size_bytes(args, ckpt_config)),
        "encoder_fetcher_mode": encoder_fetcher_mode,
        "decoder_fetcher_mode": decoder_fetcher_mode,
        "profiling": "1",
        "detailed_timing": "0",
        "offload_path": ensure_trailing_sep(Path(args.offload_path).as_posix()),
        "disk_offload": str(int(args.disk_offload)),
        "load_from_cpp": "1",
        "use_cache": str(int(args.use_cache if args.cache_ratio <= 0 else 1)),
        "quant_mode": "0",
        "vocab_size": str(ckpt_config.getint("decoder", "vocab_size")),
        "fetch_all": str(int(args.fetch_all)),
        "forced_num_experts": str(args.forced_num_experts),
        "cache_policy": args.cache_policy,
    }
    with config_path.open("w", encoding="utf-8") as config_file:
        cpp_config.write(config_file)


def load_translation_prompts(path: Path, limit: int) -> List[str]:
    with path.open("r", encoding="utf-8") as source_file:
        source_lines = recover_bpe(source_file.readlines())
    prompts = [f"translate English to German: {line.strip()}" for line in source_lines if line.strip()]
    return prompts[:limit]


def load_summarization_prompts(limit: int, cache_dir: str) -> List[str]:
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The 'datasets' package is required to build summarization prompts. Install it before running mixed-domain traces."
        ) from exc
    except AttributeError as exc:
        raise RuntimeError(
            "Failed to import 'datasets' because the installed 'pyarrow' version is incompatible. "
            "This usually happens when 'datasets' expects 'pyarrow.PyExtensionType' but the current "
            "pyarrow build no longer provides it. Try upgrading 'datasets' or downgrading 'pyarrow' "
            "to a compatible release such as 'pyarrow<21'."
        ) from exc

    dataset = load_dataset("ccdv/cnn_dailymail", "3.0.0", split="test", cache_dir=cache_dir)
    prompts = []
    for row in dataset:
        article = row["article"].strip().replace(" n't", "n't")
        if article:
            prompts.append(f"summarize: {article}")
        if len(prompts) >= limit:
            break
    return prompts


def needs_summarization_prompts(args: argparse.Namespace) -> bool:
    condition_uses_two_domains = args.condition != "stable_homogeneous"
    if args.match_domain_marginal:
        condition_uses_two_domains = True
    if condition_uses_two_domains:
        return True
    return args.stable_domain == "summarization"


def truncate_words(text: str, budget: int) -> str:
    words = text.split()
    if budget <= 0 or len(words) <= budget:
        return text
    return " ".join(words[:budget])


def interleave_segments(primary: str, secondary: str, chunk_words: int) -> str:
    primary_words = primary.split()
    secondary_words = secondary.split()
    mixed: List[str] = []
    max_len = max(len(primary_words), len(secondary_words))
    for start in range(0, max_len, chunk_words):
        mixed.extend(primary_words[start:start + chunk_words])
        mixed.extend(secondary_words[start:start + chunk_words])
    return " ".join(mixed)


def mix_prompt(primary_text: str, secondary_text: str, args: argparse.Namespace, primary_fraction: float) -> str:
    primary_budget = max(int(args.mix_word_budget * primary_fraction), 1)
    secondary_budget = max(args.mix_word_budget - primary_budget, 1)
    primary_text = truncate_words(primary_text, primary_budget)
    secondary_text = truncate_words(secondary_text, secondary_budget)
    if args.mix_mode == "interleave":
        return interleave_segments(primary_text, secondary_text, args.interleave_chunk_words)
    return primary_text + "\n" + secondary_text


def cycle_pick(pool: Sequence[str], index: int) -> str:
    return pool[index % len(pool)]


def parse_domain_order(raw_value: str) -> Tuple[str, str]:
    domains = [token.strip() for token in raw_value.split(",") if token.strip()]
    if len(domains) != 2:
        raise ValueError("--domain_order must contain exactly two comma-separated domains.")
    if sorted(domains) != ["summarization", "translation"]:
        raise ValueError("This script currently supports the translation/summarization domain pair.")
    return domains[0], domains[1]


def prompt_pool_for_domain(
    domain: str,
    translation_prompts: Sequence[str],
    summarization_prompts: Sequence[str],
) -> Sequence[str]:
    if domain == "translation":
        return translation_prompts
    if domain == "summarization":
        return summarization_prompts
    raise ValueError(f"Unsupported domain: {domain}")


def pick_prompt(
    domain: str,
    request_idx: int,
    translation_prompts: Sequence[str],
    summarization_prompts: Sequence[str],
) -> str:
    return cycle_pick(prompt_pool_for_domain(domain, translation_prompts, summarization_prompts), request_idx)


def phase_major_domain(domain_a: str, domain_b: str, block_index: int) -> str:
    return domain_a if block_index % 2 == 0 else domain_b


def choose_domain_by_fraction(domain_a: str, domain_b: str, relative_position: int, block_size: int, domain_a_fraction: float) -> str:
    cutoff = max(0, min(block_size, round(block_size * domain_a_fraction)))
    return domain_a if relative_position < cutoff else domain_b


def homogeneous_domain_for_request(args: argparse.Namespace, domain_a: str, domain_b: str, block_index: int, relative_position: int) -> str:
    if args.condition == "shifted_homogeneous":
        domain_a_fraction = args.shift_major_fraction if block_index % 2 == 0 else 1.0 - args.shift_major_fraction
        return choose_domain_by_fraction(domain_a, domain_b, relative_position, args.shift_block_size, domain_a_fraction)
    if args.match_domain_marginal:
        return domain_a if (relative_position % 2 == 0) else domain_b
    return args.stable_domain


def mixed_plan_for_request(args: argparse.Namespace, domain_a: str, domain_b: str, block_index: int, request_idx: int) -> Tuple[str, str, float, str]:
    if args.condition in {"stable_mixed", "stable_mixed_ab"}:
        return domain_a, domain_b, args.stable_mix_fraction, "none"
    if args.condition == "stable_mixed_ba":
        return domain_b, domain_a, args.stable_mix_fraction, "none"
    if args.condition == "stable_mixed_balanced_order":
        if request_idx % 2 == 0:
            return domain_a, domain_b, args.stable_mix_fraction, "none"
        return domain_b, domain_a, args.stable_mix_fraction, "none"

    variant = SHIFTED_MIXED_VARIANT_BY_CONDITION[args.condition]
    if variant == "ratio_only":
        primary_fraction = args.shift_major_fraction if block_index % 2 == 0 else 1.0 - args.shift_major_fraction
        return domain_a, domain_b, primary_fraction, "domain_ratio"
    if variant == "order_only":
        primary = domain_a if block_index % 2 == 0 else domain_b
        secondary = domain_b if primary == domain_a else domain_a
        return primary, secondary, 0.5, "domain_order"

    primary = phase_major_domain(domain_a, domain_b, block_index)
    secondary = domain_b if primary == domain_a else domain_a
    return primary, secondary, args.shift_major_fraction, "domain_ratio_and_order"


def distribution_for_plan(primary: str, secondary: str = "", primary_fraction: float = 1.0) -> Dict[str, float]:
    if not secondary:
        return {primary: 1.0}
    return {
        primary: primary_fraction,
        secondary: 1.0 - primary_fraction,
    }


def phase_distribution(args: argparse.Namespace, domain_a: str, domain_b: str, block_index: int) -> Dict[str, float]:
    if args.condition == "shifted_homogeneous":
        domain_a_fraction = args.shift_major_fraction if block_index % 2 == 0 else 1.0 - args.shift_major_fraction
        return {domain_a: domain_a_fraction, domain_b: 1.0 - domain_a_fraction}
    if args.condition in SHIFTED_MIXED_VARIANT_BY_CONDITION:
        variant = SHIFTED_MIXED_VARIANT_BY_CONDITION[args.condition]
        if variant == "order_only":
            return {domain_a: 0.5, domain_b: 0.5}
        domain_a_fraction = args.shift_major_fraction if block_index % 2 == 0 else 1.0 - args.shift_major_fraction
        return {domain_a: domain_a_fraction, domain_b: 1.0 - domain_a_fraction}
    if args.match_domain_marginal or args.condition == "stable_mixed_balanced_order":
        return {domain_a: 0.5, domain_b: 0.5}
    if args.condition.startswith("stable_mixed"):
        if args.condition == "stable_mixed_ba":
            return {domain_b: args.stable_mix_fraction, domain_a: 1.0 - args.stable_mix_fraction}
        return {domain_a: args.stable_mix_fraction, domain_b: 1.0 - args.stable_mix_fraction}
    return {args.stable_domain: 1.0}


def shannon_entropy(distribution: Dict[str, float]) -> float:
    entropy = -sum(prob * math.log2(prob) for prob in distribution.values() if prob > 0.0)
    return 0.0 if abs(entropy) < 1e-12 else entropy


def kl_divergence(left: Dict[str, float], right: Dict[str, float]) -> float:
    keys = set(left) | set(right)
    total = 0.0
    for key in keys:
        p = left.get(key, 0.0)
        q = right.get(key, 0.0)
        if p > 0.0 and q > 0.0:
            total += p * math.log2(p / q)
    return total


def js_divergence(left: Dict[str, float], right: Dict[str, float]) -> float:
    keys = set(left) | set(right)
    midpoint = {key: 0.5 * (left.get(key, 0.0) + right.get(key, 0.0)) for key in keys}
    return 0.5 * kl_divergence(left, midpoint) + 0.5 * kl_divergence(right, midpoint)


def build_trace(args: argparse.Namespace, translation_prompts: Sequence[str], summarization_prompts: Sequence[str]) -> List[RequestSpec]:
    if args.condition not in CONDITION_CHOICES:
        raise ValueError(f"Unsupported condition: {args.condition}")

    domain_a, domain_b = parse_domain_order(args.domain_order)

    requests: List[RequestSpec] = []
    for request_idx in range(args.num_requests):
        block_index = request_idx // args.shift_block_size
        relative_position = request_idx % args.shift_block_size
        distance_to_nearest_boundary = min(relative_position, args.shift_block_size - relative_position)
        stream_distribution = phase_distribution(args, domain_a, domain_b, block_index)

        if args.condition in {"stable_homogeneous", "shifted_homogeneous"}:
            domain_label = homogeneous_domain_for_request(args, domain_a, domain_b, block_index, relative_position)
            request_distribution = {domain_label: 1.0}
            domain_sequence = [domain_label]
            text = pick_prompt(domain_label, request_idx, translation_prompts, summarization_prompts)
            if args.homogeneous_word_budget > 0:
                text = truncate_words(text, args.homogeneous_word_budget)
            shift_axis = "domain_ratio" if args.condition == "shifted_homogeneous" else "none"
        else:
            primary_domain, secondary_domain, primary_fraction, shift_axis = mixed_plan_for_request(args, domain_a, domain_b, block_index, request_idx)
            domain_label = f"mixed:{primary_domain}+{secondary_domain}"
            request_distribution = distribution_for_plan(primary_domain, secondary_domain, primary_fraction)
            domain_sequence = [primary_domain, secondary_domain]
            text = mix_prompt(
                pick_prompt(primary_domain, request_idx, translation_prompts, summarization_prompts),
                pick_prompt(secondary_domain, request_idx, translation_prompts, summarization_prompts),
                args,
                primary_fraction,
            )

        requests.append(
            RequestSpec(
                request_id=f"{args.trace_id}-req-{request_idx:05d}",
                condition=args.condition,
                domain_label=domain_label,
                text=text,
                metadata={
                    "request_index": request_idx,
                    "block_index": block_index,
                    "relative_position_in_block": relative_position,
                    "is_shift_boundary": int(request_idx % args.shift_block_size == 0),
                    "distance_to_nearest_boundary": distance_to_nearest_boundary,
                    "domain_sequence": domain_sequence,
                    "request_domain_distribution": request_distribution,
                    "stream_phase_distribution": stream_distribution,
                    "request_entropy_bits": shannon_entropy(request_distribution),
                    "stream_entropy_bits": shannon_entropy(stream_distribution),
                    "intra_request_mixing": int(len(request_distribution) >= 2),
                    "cross_request_shift": int(args.condition.startswith("shifted_")),
                    "shift_axis": shift_axis,
                },
            )
        )
    return requests


def summarize_workload(requests: Sequence[RequestSpec]) -> Dict[str, object]:
    if not requests:
        return {}

    total_distribution: Dict[str, float] = {}
    phase_distributions: Dict[int, Dict[str, float]] = {}
    mixed_requests = 0
    for request in requests:
        metadata = request.metadata
        request_distribution = metadata["request_domain_distribution"]
        for domain, fraction in request_distribution.items():
            total_distribution[domain] = total_distribution.get(domain, 0.0) + float(fraction)
        if metadata["intra_request_mixing"]:
            mixed_requests += 1
        phase_distributions.setdefault(int(metadata["block_index"]), metadata["stream_phase_distribution"])

    total_requests = float(len(requests))
    total_distribution = {domain: value / total_requests for domain, value in sorted(total_distribution.items())}
    sorted_phases = [phase_distributions[key] for key in sorted(phase_distributions)]
    first_phase = sorted_phases[0]
    second_phase = sorted_phases[1] if len(sorted_phases) > 1 else sorted_phases[0]

    return {
        "condition": requests[0].condition,
        "num_requests": len(requests),
        "domain_marginal": total_distribution,
        "mean_request_entropy_bits": sum(float(request.metadata["request_entropy_bits"]) for request in requests) / total_requests,
        "mixed_request_fraction": mixed_requests / total_requests,
        "first_phase_distribution": first_phase,
        "second_phase_distribution": second_phase,
        "phase_js_divergence_bits": js_divergence(first_phase, second_phase),
        "shift_axes": sorted({str(request.metadata["shift_axis"]) for request in requests}),
    }


def write_workload_summary(summary_path: Path, requests: Sequence[RequestSpec]) -> None:
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summarize_workload(requests), summary_file, ensure_ascii=False, indent=2)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_trace_manifest(trace_path: Path, requests: Iterable[RequestSpec]) -> None:
    with trace_path.open("w", encoding="utf-8") as trace_file:
        for request in requests:
            trace_file.write(
                json.dumps(
                    {
                        "request_id": request.request_id,
                        "condition": request.condition,
                        "domain_label": request.domain_label,
                        "text": request.text,
                        "metadata": request.metadata,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def run_trace(
    args: argparse.Namespace,
    tokenizer: AutoTokenizer,
    ft_t5: FTT5,
    requests: Sequence[RequestSpec],
    request_metrics_path: Path,
) -> int:
    os.environ["PREGATED_TRACE_OUT"] = str((args.output_dir / "prefetch_trace.tsv").resolve())
    os.environ["PREGATED_TRACE_ID"] = args.trace_id
    request_metrics_tmp_path = request_metrics_path.with_suffix(request_metrics_path.suffix + ".tmp")
    request_metrics_partial_path = request_metrics_path.with_suffix(request_metrics_path.suffix + ".partial")

    processed_requests = 0
    try:
        with request_metrics_tmp_path.open("w", encoding="utf-8") as metrics_file:
            for request in requests:
                os.environ["PREGATED_REQUEST_ID"] = request.request_id
                os.environ["PREGATED_CONDITION"] = request.condition
                os.environ["PREGATED_REQUEST_DOMAIN"] = request.domain_label

                tokenized = tokenizer([request.text], return_tensors="pt", padding=True)

                torch.cuda.synchronize()
                start_time = time.perf_counter()
                with torch.no_grad():
                    outputs = ft_t5(
                        tokenized,
                        None,
                        args.beam_width,
                        args.max_seq_len,
                        args.sampling_topk,
                        args.sampling_topp,
                        beam_search_diversity_rate=0.0,
                        is_return_output_log_probs=False,
                        is_return_cum_log_probs=False,
                    )
                torch.cuda.synchronize()
                latency_s = time.perf_counter() - start_time

                if len(outputs) < 2:
                    raise RuntimeError("Unexpected FT output format while running trace workload.")
                output_ids, output_seq_lens = outputs[0], outputs[1]

                metrics_file.write(
                    json.dumps(
                        {
                            "request_id": request.request_id,
                            "condition": request.condition,
                            "domain_label": request.domain_label,
                            "latency_s": latency_s,
                            "input_token_count": int(tokenized.input_ids.shape[1]),
                            "output_token_count": int(output_seq_lens[0][0]),
                            "metadata": request.metadata,
                            "decoded_preview": tokenizer.decode(
                                output_ids[0][0][: output_seq_lens[0][0]],
                                skip_special_tokens=True,
                            ),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                metrics_file.flush()
                processed_requests += 1

        if processed_requests != len(requests):
            raise RuntimeError(
                f"Incomplete trace execution: processed {processed_requests} of {len(requests)} requests."
            )
        os.replace(request_metrics_tmp_path, request_metrics_path)
        if request_metrics_partial_path.exists():
            request_metrics_partial_path.unlink()
        return processed_requests
    except Exception:
        if request_metrics_partial_path.exists():
            request_metrics_partial_path.unlink()
        if request_metrics_tmp_path.exists():
            os.replace(request_metrics_tmp_path, request_metrics_partial_path)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--trace_id", type=str, required=True)
    parser.add_argument("--condition", choices=CONDITION_CHOICES, required=True)
    parser.add_argument("--method", choices=sorted(METHOD_TO_FETCHER_MODE.keys()), required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--offload_path", type=Path, required=True)
    parser.add_argument("--lib_path", type=str, default="lib/libth_transformer.so")
    parser.add_argument("--model_type", type=str, default="Megatron-DeepSpeed")
    parser.add_argument("--translation_source", type=Path, default=REPO_ROOT / "examples/pytorch/decoding/utils/translation/test.en")
    parser.add_argument("--summarization_cache_dir", type=str, default=str(REPO_ROOT / ".cache" / "datasets"))
    parser.add_argument("--num_requests", type=int, default=128)
    parser.add_argument("--shift_block_size", type=int, default=32)
    parser.add_argument("--stable_domain", choices=["translation", "summarization"], default="translation")
    parser.add_argument("--domain_order", type=str, default="translation,summarization")
    parser.add_argument("--match_domain_marginal", action="store_true")
    parser.add_argument("--mix_mode", choices=["concat", "interleave"], default="concat")
    parser.add_argument("--homogeneous_word_budget", type=int, default=0)
    parser.add_argument("--mix_word_budget", type=int, default=256)
    parser.add_argument("--mixed_primary_fraction", type=float)
    parser.add_argument("--stable_mix_fraction", type=float)
    parser.add_argument("--shift_major_fraction", type=float, default=0.8)
    parser.add_argument("--interleave_chunk_words", type=int, default=16)
    parser.add_argument("--beam_width", type=int, default=1)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--sampling_topk", type=int, default=1)
    parser.add_argument("--sampling_topp", type=float, default=0.0)
    parser.add_argument("--moe_topk", type=int, default=1)
    parser.add_argument("--data_type", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--tensor_para_size", type=int, default=1)
    parser.add_argument("--pipeline_para_size", type=int, default=1)
    parser.add_argument("--cache_ratio", type=float, default=0.1)
    parser.add_argument("--arena_size_bytes", type=int)
    parser.add_argument("--default_arena_size_bytes", type=int, default=20602421248)
    parser.add_argument("--cache_policy", choices=["LFU", "LRU", "LIFO"], default="LFU")
    parser.add_argument("--disk_offload", action="store_true")
    parser.add_argument("--use_cache", type=int, default=1)
    parser.add_argument("--fetch_all", action="store_true")
    parser.add_argument("--forced_num_experts", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--write_trace_only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.beam_width != 1:
        raise ValueError("This trace-driven experiment script currently expects beam_width=1.")
    if args.shift_block_size <= 0:
        raise ValueError("shift_block_size must be positive.")
    if args.homogeneous_word_budget < 0:
        raise ValueError("homogeneous_word_budget must be non-negative.")
    if args.mix_word_budget <= 0:
        raise ValueError("mix_word_budget must be positive.")
    if args.stable_mix_fraction is None:
        args.stable_mix_fraction = args.mixed_primary_fraction if args.mixed_primary_fraction is not None else 0.5
    if not 0.0 < args.stable_mix_fraction < 1.0:
        raise ValueError("stable_mix_fraction must be between 0 and 1.")
    if not 0.5 <= args.shift_major_fraction < 1.0:
        raise ValueError("shift_major_fraction must be in [0.5, 1.0).")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    request_metrics_path = args.output_dir / "request_metrics.jsonl"
    request_metrics_partial_path = args.output_dir / "request_metrics.jsonl.partial"
    run_status_path = args.output_dir / "run_status.json"
    os.environ["PREGATED_CPP_CONFIG"] = str((args.output_dir / "cpp_config.ini").resolve())
    trace_output_path = (args.output_dir / "prefetch_trace.tsv").resolve()
    if trace_output_path.exists():
        trace_output_path.unlink()
    if request_metrics_path.exists():
        request_metrics_path.unlink()
    if request_metrics_partial_path.exists():
        request_metrics_partial_path.unlink()
    if run_status_path.exists():
        run_status_path.unlink()

    translation_prompts = load_translation_prompts(args.translation_source, args.num_requests * 2)
    summarization_prompts: List[str] = []
    if needs_summarization_prompts(args):
        summarization_prompts = load_summarization_prompts(args.num_requests * 2, args.summarization_cache_dir)

    if not translation_prompts or (needs_summarization_prompts(args) and not summarization_prompts):
        raise RuntimeError("Failed to load enough prompts for the requested trace.")

    requests = build_trace(args, translation_prompts, summarization_prompts)
    write_trace_manifest(args.output_dir / "request_trace.jsonl", requests)
    write_workload_summary(args.output_dir / "workload_summary.json", requests)

    run_status = {
        "trace_id": args.trace_id,
        "condition": args.condition,
        "method": args.method,
        "cache_ratio": args.cache_ratio,
        "seed": args.seed,
        "expected_num_requests": len(requests),
        "completed_num_requests": 0,
        "status": "trace_ready",
        "started_at_utc": utc_timestamp(),
    }
    write_json(run_status_path, run_status)

    if args.write_trace_only:
        run_status["completed_num_requests"] = len(requests)
        run_status["status"] = "trace_only_complete"
        run_status["completed_at_utc"] = utc_timestamp()
        write_json(run_status_path, run_status)
        return

    ckpt_config = read_ft_config(args.ckpt_path)
    write_cpp_config(args, ckpt_config, Path(os.environ["PREGATED_CPP_CONFIG"]))

    try:
        tokenizer, ft_t5 = build_runtime(args, ckpt_config)
        processed_requests = run_trace(args, tokenizer, ft_t5, requests, request_metrics_path)
        run_status["completed_num_requests"] = processed_requests
        run_status["status"] = "complete"
        run_status["completed_at_utc"] = utc_timestamp()
        write_json(run_status_path, run_status)
    except Exception as exc:
        run_status["status"] = "failed"
        run_status["completed_num_requests"] = sum(
            1 for line in request_metrics_partial_path.open("r", encoding="utf-8")
            if line.strip()
        ) if request_metrics_partial_path.exists() else 0
        run_status["completed_at_utc"] = utc_timestamp()
        run_status["error"] = repr(exc)
        write_json(run_status_path, run_status)
        raise


if __name__ == "__main__":
    main()
