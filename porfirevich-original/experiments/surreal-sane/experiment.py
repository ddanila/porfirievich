#!/usr/bin/env python3
"""Reversible, gradient-free style steering pilot. Never writes model weights."""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
from transformers import GPT2LMHeadModel

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from run_cpu import DEFAULT_MODEL, OriginalYTEncoder, build_filters, generate_replies, resolve_device, seed_runtime


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


@contextmanager
def edit(model, layer, delta):
    if layer is None:
        yield
        return
    bias = model.transformer.h[layer].mlp.c_proj.bias
    original = bias.detach().clone()
    try:
        with torch.no_grad():
            bias.add_(torch.tensor(delta, device=bias.device, dtype=bias.dtype))
        yield
    finally:
        with torch.no_grad():
            bias.copy_(original)
        assert torch.equal(bias, original)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('discovery', 'test'), default='discovery')
    parser.add_argument('--patch', type=Path, help='selected patch for held-out evaluation')
    args = parser.parse_args()
    if args.phase == 'test' and not args.patch:
        parser.error('--patch is required for test')
    data = json.loads((ROOT / 'data.json').read_text())
    device = resolve_device('auto')
    torch.set_num_threads(6)
    model = GPT2LMHeadModel.from_pretrained(DEFAULT_MODEL, local_files_only=True, use_safetensors=False).to(device).eval()
    tokenizer = OriginalYTEncoder(DEFAULT_MODEL / 'encoder.model')
    one, two = build_filters(tokenizer, 'latest-site', False)
    layers = [7, 11, 15]

    def ids(text):
        return torch.tensor([tokenizer.encode(text)], device=device)

    def nll(text):
        tokens = ids(text)
        return model(tokens, labels=tokens, use_cache=False).loss.item()

    with torch.inference_mode():
        if args.phase == 'discovery':
            features = {layer: [[], []] for layer in layers}
            for pair in data['train']:
                for label, text in enumerate(pair):
                    hidden = model.transformer(ids(text), output_hidden_states=True, use_cache=False).hidden_states
                    for layer in layers:
                        features[layer][label].append(hidden[layer + 1][0, -8:].mean(0).cpu())
            variants = [{'name': 'baseline', 'layer': None, 'delta': [], 'temperature': 1.0},
                        {'name': 'temperature-0.85', 'layer': None, 'delta': [], 'temperature': 0.85},
                        {'name': 'temperature-1.15', 'layer': None, 'delta': [], 'temperature': 1.15}]
            for layer in layers:
                direction = torch.stack(features[layer][1]).mean(0) - torch.stack(features[layer][0]).mean(0)
                for strength in (0.5, 1.0, 2.0):
                    variants.append({'name': f'layer{layer}-strength{strength}', 'layer': layer,
                                     'strength': strength, 'delta': (strength * direction).tolist(), 'temperature': 1.0})
            save(ROOT / 'patches.json', {'method': 'mean last-eight-token residual difference; add to MLP output bias',
                                         'data_sha256': hashlib.sha256((ROOT / 'data.json').read_bytes()).hexdigest(),
                                         'variants': variants})
            split = 'dev'
        else:
            patch = json.loads(args.patch.read_text())
            variants = [{'name': 'baseline', 'layer': None, 'delta': [], 'temperature': 1.0},
                        {'name': 'temperature-0.85', 'layer': None, 'delta': [], 'temperature': 0.85},
                        {'name': 'temperature-1.15', 'layer': None, 'delta': [], 'temperature': 1.15}, patch]
            split = 'test'
        results = []
        for variant in variants:
            started = time.perf_counter()
            with edit(model, variant['layer'], variant['delta']):
                pairs = [{'ordinary_nll': nll(pair[0]), 'surreal_nll': nll(pair[1])} for pair in data[split + '_pairs']]
                prose = [nll(text) for text in data[split + '_prose']]
                generations = []
                for prompt in data[split + '_prompts']:
                    for seed in (42, 137):
                        seed_runtime(seed, device)
                        replies = generate_replies(model, tokenizer, prompt, 60, 1, 'latest-site', one, two,
                                                   device=device, temperature=variant['temperature'])
                        generations.append({'prompt': prompt, 'seed': seed, 'text': prompt + replies[0]})
            result = {k: v for k, v in variant.items() if k != 'delta'}
            result.update(style_margin=sum(p['ordinary_nll'] - p['surreal_nll'] for p in pairs) / len(pairs),
                          prose_nll=sum(prose) / len(prose), pairs=pairs, generations=generations,
                          seconds=time.perf_counter() - started)
            results.append(result)
            save(ROOT / (args.phase + '.json'), {'device': str(device), 'phase': args.phase, 'results': results})
            print(result['name'], 'margin', round(result['style_margin'], 4), 'prose NLL', round(result['prose_nll'], 4), flush=True)


if __name__ == '__main__':
    main()
