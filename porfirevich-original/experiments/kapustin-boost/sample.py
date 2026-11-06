#!/usr/bin/env python3
"""Generate with the experiment's prefix-aware Капустин bias (original weights)."""
import argparse
import json
import math

from experiment import (
    ROOT, DEFAULT_MODEL, GPT2LMHeadModel, OriginalYTEncoder, NameBias,
    biased_sequence, build_filters, resolve_device, seed_runtime, trim_like_site, torch,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prompt')
    parser.add_argument('--strength', type=float, default=3.0,
                        help='Name logit bias; 3 matches the experiment, 0 disables it.')
    parser.add_argument('--length', type=int, default=60, help='New tokens per sample.')
    parser.add_argument('--count', type=int, default=1)
    parser.add_argument('--seed', type=int, help='First seed; subsequent samples increment it.')
    parser.add_argument('--device', choices=('auto', 'cpu', 'mps', 'cuda'), default='auto')
    args = parser.parse_args()
    if not args.prompt.strip():
        parser.error('prompt must not be empty')
    if args.length < 1 or args.count < 1:
        parser.error('length and count must be positive')
    if not math.isfinite(args.strength) or args.strength < 0:
        parser.error('strength must be finite and nonnegative')

    device = resolve_device(args.device)
    torch.set_num_threads(6)
    tokenizer = OriginalYTEncoder(DEFAULT_MODEL / 'encoder.model')
    context = tokenizer.encode(args.prompt)
    if not context:
        parser.error('prompt must produce at least one token')
    forms = json.loads((ROOT / 'design.json').read_text())['forms']
    bias = NameBias([tokenizer.encode(form) for form in forms], args.strength) if args.strength else None
    one, two = build_filters(tokenizer, 'latest-site', False)
    model = GPT2LMHeadModel.from_pretrained(
        DEFAULT_MODEL, local_files_only=True, use_safetensors=False,
    ).to(device).eval()
    decoded_prompt = tokenizer.decode(context)
    with torch.inference_mode():
        for index in range(args.count):
            if args.seed is not None:
                seed_runtime(args.seed + index, device)
            ids = biased_sequence(model, context, args.length, one, two, device, bias)
            continuation = trim_like_site(tokenizer.decode(ids)[len(decoded_prompt):], 'latest-site')
            if args.count > 1:
                print(f'\n--- Sample {index + 1} ---')
            print(args.prompt + continuation, flush=True)


if __name__ == '__main__':
    main()
