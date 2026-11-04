#!/usr/bin/env python3
"""Compare full-precision predictions, masks, and sliding-window behavior."""

import argparse
import json
from pathlib import Path
import sys

import torch
from transformers import GPT2LMHeadModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_cpu import (  # noqa: E402
    CachedNextLogits, DEFAULT_MODEL, OriginalYTEncoder, build_filters,
    filter_logits_batched, resolve_device, sample_sequence,
    sample_sequence_optimized, seed_runtime, top_k_top_p_filtering,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu', choices=('cpu', 'mps', 'cuda'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(6)
    device = resolve_device(args.device)
    model = GPT2LMHeadModel.from_pretrained(DEFAULT_MODEL, local_files_only=True,
                                          use_safetensors=False).to(device).eval()
    assert next(model.parameters()).dtype == torch.float32
    tokenizer = OriginalYTEncoder(DEFAULT_MODEL / 'encoder.model')
    single, double = build_filters(tokenizer, 'latest-site', False)
    report = {'device': str(device), 'dtype': 'float32', 'prediction_checks': [],
              'seeded_token_checks': []}

    with torch.inference_mode():
        # Check linebreak suppression both enabled and disabled, and the old
        # profile's smaller single-token mask, including repeated newlines.
        for profile in ('latest-site', 'checkpoint-era'):
            for allow_linebreak in (False, True):
                one, two = build_filters(tokenizer, profile, allow_linebreak)
                seed_runtime(123, device)
                logits = torch.randn(3, model.config.vocab_size, device=device)
                previous = torch.tensor([two[0], 42, two[0]], device=device)
                expected = logits.clone()
                for i, token in enumerate(previous.cpu().tolist()):
                    expected[i, one] = -float('inf')
                    if token in two:
                        expected[i, token] = -float('inf')
                    top_k_top_p_filtering(expected[i], top_p=0.9)
                actual = filter_logits_batched(logits.clone(), previous, one, two)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        report['filter_masks_exact'] = True

        text = 'Однажды вечером я вышел из дома. За окном шумел дождь. '
        tokens = tokenizer.encode(text * 100)
        # Teacher forcing compares both paths on identical input, including
        # cache growth and two successive window shifts.
        for start in (3, 48, 1022):
            forward = CachedNextLogits(model)
            for size in range(start, start + 4):
                ids = torch.tensor([tokens[:size]], device=device)
                expected = model(ids[:, -1023:], use_cache=False).logits[:, -1, :]
                actual = forward(ids)
                p = expected.softmax(-1)
                q = actual.softmax(-1)
                max_error = (p - q).abs().max().item()
                tv = ((p - q).abs().sum(-1) / 2).max().item()
                assert max_error < 2e-4 and tv < 2e-4, (size, max_error, tv)
                report['prediction_checks'].append({
                    'context_tokens': size, 'max_probability_error': max_error,
                    'total_variation': tv,
                    'argmax_equal': bool(torch.equal(p.argmax(-1), q.argmax(-1))),
                })

        for prompt in ('Однажды вечером', 'Когда я проснулся, за окном', 'Человек состоит из'):
            for seed in (42, 7):
                kwargs = dict(model=model, length=15, context=tokenizer.encode(prompt),
                              num_samples=3, filter_single=single, filter_double=double,
                              device=device)
                seed_runtime(seed, device)
                expected = sample_sequence(**kwargs)
                seed_runtime(seed, device)
                actual = sample_sequence_optimized(**kwargs)
                report['seeded_token_checks'].append({
                    'prompt': prompt, 'seed': seed,
                    'identical': bool(torch.equal(expected, actual)),
                })

    report['passed'] = True
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + '\n')


if __name__ == '__main__':
    main()
