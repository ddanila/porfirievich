#!/usr/bin/env python3
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted in 2026 by ddanila from mgrankin/ru_transformers.
# Changes: modern CPU compatibility, historical service profiles, and iterative selection.
"""Faithful CPU port of the original Porfirievich completion service."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import regex as re
import torch
import torch.nn.functional as F
import youtokentome as yttm
from transformers import GPT2LMHeadModel


FILTER_VALUE = -float("inf")
NEW_LINE = "<|n|>"
ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "model" / "m_checkpoint-3365357"


class OriginalYTEncoder:
    """The transformations from the repository's original yt_encoder.py."""

    def __init__(self, filename: Path) -> None:
        self.bpe = yttm.BPE(str(filename))

    def encode(self, text: str) -> list[int]:
        if text and text[0] != " ":
            text = " " + text
        text = re.sub(r"(?=[^ ])([\W])([\w])", r"\g<1> \g<2>", text)
        text = text.replace("\n", f" {NEW_LINE} ")
        return self.bpe.encode([text], output_type=yttm.OutputType.ID)[0]

    def decode(self, tokens: list[int] | torch.Tensor) -> str:
        if not isinstance(tokens, list):
            tokens = tokens.tolist()
        result = self.bpe.decode(tokens)[0]
        result = re.sub(r"( )?(<\|n\|>)( )?", "\n", result)
        result = re.sub(r"([\n(]) (\w)", r"\g<1>\g<2>", result)
        result = re.sub(r'(\W)([«"\n(]|^) (\w)', r"\g<1>\g<2>\g<3>", result)
        result = re.sub(r"(\w)- (\w)", r"\g<1>-\g<2>", result)
        return result


def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 0.0,
) -> torch.Tensor:
    """Original 2019 top-k/nucleus filtering algorithm."""
    assert logits.dim() == 1
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = FILTER_VALUE

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = FILTER_VALUE
    return logits


def sample_sequence(
    model: GPT2LMHeadModel,
    length: int,
    context: list[int],
    num_samples: int,
    filter_single: list[int],
    filter_double: list[int],
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Original sampling loop, intentionally without a modern KV-cache rewrite."""
    context_tensor = torch.tensor(context, dtype=torch.long, device=device)
    generated = context_tensor.unsqueeze(0).repeat(num_samples, 1)

    with torch.inference_mode():
        for _ in range(length):
            outputs = model(input_ids=generated[:, -1023:])
            next_tokens = torch.zeros(num_samples, dtype=torch.long, device=device)
            for sample_index in range(num_samples):
                next_token_logits = outputs[0][sample_index, -1, :] / 1.0
                next_token_logits[filter_single] = FILTER_VALUE
                if generated[sample_index, -1] in filter_double:
                    next_token_logits[generated[sample_index, -1]] = FILTER_VALUE
                filtered_logits = top_k_top_p_filtering(
                    next_token_logits,
                    top_k=0,
                    top_p=0.9,
                )
                next_tokens[sample_index] = torch.multinomial(
                    F.softmax(filtered_logits, dim=-1),
                    num_samples=1,
                )
            generated = torch.cat((generated, next_tokens.unsqueeze(-1)), dim=1)
    return generated


class CachedNextLogits:
    """Full-precision GPT-2 forward preserving the original sliding window."""

    def __init__(self, model: GPT2LMHeadModel) -> None:
        self.model = model
        self.past = None
        self.previous_length = 0

    def __call__(self, generated: torch.Tensor) -> torch.Tensor:
        length = generated.shape[1]
        # Once the window slides, GPT-2's absolute positions reset. Dropping
        # only the oldest KV entry would retain incorrect positions/context.
        incremental = (
            self.past is not None
            and length == self.previous_length + 1
            and length <= 1023
        )
        outputs = self.model.transformer(
            input_ids=generated[:, -1:] if incremental else generated[:, -1023:],
            past_key_values=self.past if incremental else None,
            use_cache=length < 1023,
            return_dict=True,
        )
        self.past = outputs.past_key_values
        self.previous_length = length
        # The vocabulary projection is needed only for the final position.
        return self.model.lm_head(outputs.last_hidden_state[:, -1, :])


def filter_logits_batched(
    logits: torch.Tensor,
    previous_tokens: torch.Tensor,
    filter_single: list[int],
    filter_double: list[int],
) -> torch.Tensor:
    """Original masks and nucleus threshold without GPU-to-Python branches."""
    logits[:, filter_single] = FILTER_VALUE
    for token in filter_double:
        logits[:, token].masked_fill_(previous_tokens == token, FILTER_VALUE)
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
    remove = cumulative_probs > 0.9
    remove[:, 1:] = remove[:, :-1].clone()
    remove[:, 0] = False
    # A fixed-size scatter avoids the dynamic boolean-index result that forces
    # GPU synchronization in the reference implementation.
    mask = torch.zeros_like(remove).scatter_(1, sorted_indices, remove)
    return logits.masked_fill_(mask, FILTER_VALUE)


def sample_sequence_optimized(
    model: GPT2LMHeadModel,
    length: int,
    context: list[int],
    num_samples: int,
    filter_single: list[int],
    filter_double: list[int],
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    generated = torch.empty((num_samples, len(context) + length), dtype=torch.long, device=device)
    generated[:, :len(context)] = torch.tensor(context, dtype=torch.long, device=device)
    forward = CachedNextLogits(model)
    with torch.inference_mode():
        for position in range(len(context), len(context) + length):
            logits = forward(generated[:, :position])
            logits = filter_logits_batched(logits, generated[:, position - 1], filter_single, filter_double)
            # Keep one draw per candidate, in the original order, so this
            # optimization does not intentionally alter RNG consumption.
            for sample_index in range(num_samples):
                generated[sample_index, position] = torch.multinomial(
                    F.softmax(logits[sample_index], dim=-1), num_samples=1,
                )[0]
    return generated


def trim_like_site(text: str, profile: str) -> str:
    punctuation_and_linebreak = re.match(r"[\w\W]*[\.!?]\n", text)
    if punctuation_and_linebreak:
        return punctuation_and_linebreak[0]
    if profile == "latest-site":
        punctuation = re.match(r"[\w\W]*[\.!?]", text)
        if punctuation:
            return punctuation[0]
    return text


def generate_replies(
    model: GPT2LMHeadModel,
    tokenizer: OriginalYTEncoder,
    prompt: str,
    length: int,
    num_samples: int,
    profile: str,
    filter_single: list[int],
    newline_token: list[int],
    device: torch.device | str = "cpu",
    sampler: str = "optimized",
) -> list[str]:
    context_tokens = tokenizer.encode(prompt)
    sample = sample_sequence_optimized if sampler == "optimized" else sample_sequence
    generated = sample(
        model=model,
        context=context_tokens,
        length=length,
        num_samples=num_samples,
        filter_single=filter_single,
        filter_double=newline_token,
        device=device,
    ).to("cpu")

    decoded_prompt_length = len(tokenizer.decode(context_tokens))
    replies = []
    for item in generated:
        continuation = tokenizer.decode(item)[decoded_prompt_length:]
        replies.append(trim_like_site(continuation, profile))
    return replies


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    if requested == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("Apple GPU (MPS) is not available; use --device cpu.")
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA/ROCm is not available in this PyTorch environment; use --device cpu."
        )
    return torch.device(requested)


def seed_runtime(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    elif device.type == "mps":
        torch.mps.manual_seed(seed)


def build_filters(
    tokenizer: OriginalYTEncoder,
    profile: str,
    allow_linebreak: bool,
) -> tuple[list[int], list[int]]:
    newline_token = tokenizer.encode("\n")[-1:]
    filter_single = [1]
    if profile == "latest-site":
        filter_single += tokenizer.encode("[")[-1:] + tokenizer.encode("(")[-1:]
    if not allow_linebreak:
        filter_single += newline_token
    return filter_single, newline_token


def select_reply_index(replies: list[str], strategy: str, rng: random.Random) -> int:
    if strategy == "first":
        return 0
    if strategy == "random":
        return rng.randrange(len(replies))
    if strategy == "longest":
        return max(range(len(replies)), key=lambda index: len(replies[index]))
    raise ValueError(f"Unknown selection strategy: {strategy}")


def pick_reply(
    replies: list[str],
    iteration: int,
    iterations: int,
    auto_select: str | None,
    selection_rng: random.Random,
) -> str:
    print(f"\nIteration {iteration}/{iterations}")
    for index, reply in enumerate(replies, start=1):
        print(f"  {index}. {reply}")

    if auto_select is not None:
        selected_index = select_reply_index(replies, auto_select, selection_rng)
        print(f"Auto-selected {selected_index + 1} ({auto_select}).")
        return replies[selected_index]

    while True:
        try:
            choice = input(f"Pick a continuation [1-{len(replies)}] (q to quit): ").strip()
        except EOFError as exc:
            raise SystemExit("\nNo selection received; stopping.") from exc
        if choice.lower() in {"q", "quit"}:
            raise SystemExit("Stopped without completing all iterations.")
        try:
            selected_index = int(choice) - 1
        except ValueError:
            selected_index = -1
        if 0 <= selected_index < len(replies):
            return replies[selected_index]
        print(f"Please enter a number from 1 to {len(replies)}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Complete Russian prose with the original Porfirievich checkpoint.",
    )
    parser.add_argument("prompt", help="Russian text to continue")
    parser.add_argument("--length", type=int, default=15, help="tokens per completion")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument(
        "--iterations",
        type=int,
        metavar="N",
        help="select and append a continuation N times",
    )
    parser.add_argument(
        "--auto-select",
        choices=("first", "random", "longest"),
        help="choose each iteration automatically instead of prompting",
    )
    parser.add_argument(
        "--allow-linebreak",
        action="store_true",
        help="allow a generated newline (double newlines remain suppressed)",
    )
    parser.add_argument(
        "--profile",
        choices=("checkpoint-era", "latest-site"),
        default="latest-site",
        help="select the historically documented service behavior",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="set a reproducible seed; omitted matches the service's evolving RNG state",
    )
    parser.add_argument("--threads", type=int, help="PyTorch CPU worker threads")
    parser.add_argument("--sampler", choices=("original", "optimized"), default="optimized",
                        help="optimized uses full-precision KV caching; original preserves the reference loop")
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "mps", "auto"),
        default="cpu",
        help="execution device; Apple GPU uses 'mps', ROCm uses 'cuda'",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.length < 1:
        raise SystemExit("--length must be at least 1")
    if args.iterations is not None and args.iterations < 1:
        raise SystemExit("--iterations must be at least 1")
    if args.auto_select is not None and args.iterations is None:
        raise SystemExit("--auto-select requires --iterations")
    max_length = 500 if args.profile == "checkpoint-era" else 60
    if args.length > max_length:
        raise SystemExit(f"--length must be at most {max_length} for {args.profile}")
    if not 1 <= args.num_samples <= (10 if args.profile == "checkpoint-era" else 5):
        raise SystemExit("--num-samples is outside the selected profile's service limit")
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    if args.seed is not None:
        seed_runtime(args.seed, device)

    tokenizer = OriginalYTEncoder(args.model / "encoder.model")
    model = GPT2LMHeadModel.from_pretrained(
        args.model,
        local_files_only=True,
        use_safetensors=False,
    )
    model.to(device)
    model.eval()

    filter_single, newline_token = build_filters(
        tokenizer,
        args.profile,
        args.allow_linebreak,
    )

    if args.iterations is None:
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
            sampler=args.sampler,
        )
        print(json.dumps({"replies": replies}, ensure_ascii=False, indent=2))
        return

    accumulated_text = args.prompt
    selection_rng = random.Random(args.seed)
    for iteration in range(1, args.iterations + 1):
        replies = generate_replies(
            model=model,
            tokenizer=tokenizer,
            prompt=accumulated_text,
            length=args.length,
            num_samples=args.num_samples,
            profile=args.profile,
            filter_single=filter_single,
            newline_token=newline_token,
            device=device,
            sampler=args.sampler,
        )
        accumulated_text += pick_reply(
            replies,
            iteration,
            args.iterations,
            args.auto_select,
            selection_rng,
        )

    print("\nFinal text:\n")
    print(accumulated_text)


if __name__ == "__main__":
    main()
