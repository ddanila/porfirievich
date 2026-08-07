#!/usr/bin/env python3
"""Repeatable end-to-end benchmark for the recovered Porfirievich sampler."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import resource
import statistics
import time

import torch
import transformers
from importlib.metadata import version

from run_cpu import (
    DEFAULT_MODEL,
    OriginalYTEncoder,
    build_filters,
    generate_replies,
    resolve_device,
    seed_runtime,
)


DEFAULT_PROMPT = "Однажды вечером"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark repeatable Porfirievich completion rounds.",
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--runs", type=int, default=5, help="measured rounds")
    parser.add_argument("--warmup", type=int, default=1, help="unmeasured warmup rounds")
    parser.add_argument("--threads", type=int, default=6, help="PyTorch CPU threads")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--length", type=int, default=15)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--profile",
        choices=("checkpoint-era", "latest-site"),
        default="latest-site",
    )
    parser.add_argument("--allow-linebreak", action="store_true")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, help="also save the JSON report")
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def cpu_name() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def output_hash(replies: list[str]) -> str:
    encoded = json.dumps(replies, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_args(args: argparse.Namespace) -> None:
    if args.runs < 1:
        raise SystemExit("--runs must be at least 1")
    if args.warmup < 0:
        raise SystemExit("--warmup cannot be negative")
    if args.threads < 1:
        raise SystemExit("--threads must be at least 1")
    if args.length < 1:
        raise SystemExit("--length must be at least 1")
    max_length = 500 if args.profile == "checkpoint-era" else 60
    if args.length > max_length:
        raise SystemExit(f"--length must be at most {max_length} for {args.profile}")
    max_samples = 10 if args.profile == "checkpoint-era" else 5
    if not 1 <= args.num_samples <= max_samples:
        raise SystemExit("--num-samples is outside the selected profile's service limit")


def main() -> None:
    args = parse_args()
    validate_args(args)
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)

    tokenizer = OriginalYTEncoder(args.model / "encoder.model")
    prompt_tokens = tokenizer.encode(args.prompt)
    filter_single, newline_token = build_filters(
        tokenizer,
        args.profile,
        args.allow_linebreak,
    )

    synchronize(device)
    load_started = time.perf_counter()
    model = transformers.GPT2LMHeadModel.from_pretrained(
        args.model,
        local_files_only=True,
        use_safetensors=False,
    )
    model.to(device)
    model.eval()
    synchronize(device)
    model_load_seconds = time.perf_counter() - load_started

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def run_once() -> tuple[float, list[str]]:
        seed_runtime(args.seed, device)
        synchronize(device)
        started = time.perf_counter()
        replies = generate_replies(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            length=args.length,
            num_samples=args.num_samples,
            profile=args.profile,
            filter_single=filter_single,
            newline_token=newline_token,
            device=device,
        )
        synchronize(device)
        return time.perf_counter() - started, replies

    warmup_seconds = [run_once()[0] for _ in range(args.warmup)]
    measured = [run_once() for _ in range(args.runs)]
    elapsed_seconds = [item[0] for item in measured]
    replies = [item[1] for item in measured]
    hashes = [output_hash(item) for item in replies]
    generated_tokens = args.length * args.num_samples

    device_name = cpu_name()
    gpu_memory = None
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
        gpu_memory = {
            "allocated_bytes": torch.cuda.memory_allocated(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        }

    report = {
        "benchmark_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "youtokentome": version("youtokentome"),
            "torch_cuda": torch.version.cuda,
            "torch_hip": torch.version.hip,
        },
        "device": {
            "requested": args.device,
            "resolved": str(device),
            "name": device_name,
            "cpu_threads": torch.get_num_threads(),
            "gpu_memory": gpu_memory,
        },
        "workload": {
            "prompt": args.prompt,
            "prompt_tokens": len(prompt_tokens),
            "length": args.length,
            "num_samples": args.num_samples,
            "generated_tokens_per_run": generated_tokens,
            "profile": args.profile,
            "allow_linebreak": args.allow_linebreak,
            "seed_reset_each_run": args.seed,
            "sampler": "original-uncached",
        },
        "model_load_seconds": round(model_load_seconds, 6),
        "warmup_seconds": [round(item, 6) for item in warmup_seconds],
        "run_seconds": [round(item, 6) for item in elapsed_seconds],
        "summary": {
            "runs": args.runs,
            "mean_seconds": round(statistics.mean(elapsed_seconds), 6),
            "median_seconds": round(statistics.median(elapsed_seconds), 6),
            "min_seconds": round(min(elapsed_seconds), 6),
            "max_seconds": round(max(elapsed_seconds), 6),
            "population_stdev_seconds": round(statistics.pstdev(elapsed_seconds), 6),
            "mean_generated_tokens_per_second": round(
                generated_tokens / statistics.mean(elapsed_seconds),
                6,
            ),
            "outputs_identical": len(set(hashes)) == 1,
        },
        "output_sha256": hashes,
        "sample_replies": replies[0],
        "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }

    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
