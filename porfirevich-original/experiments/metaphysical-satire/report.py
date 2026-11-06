"""Render fixed sample selections without choosing attractive outputs."""
import json
import math
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parent
data = json.loads((ROOT / 'data.json').read_text())
test = json.loads((ROOT / 'test.json').read_text())['results']
assert len(test) == 6, 'Wait for the entire evaluation'
baseline = test[0]
lines = ['# Held-out results', '',
         '| Variant | Target margin | Change from baseline | Prose NLL | Perplexity change |',
         '|---|---:|---:|---:|---:|']
for r in test:
    lines.append(f"| {r['name']} | {r['style_margin']:.4f} | {r['style_margin'] - baseline['style_margin']:+.4f} | "
                 f"{r['prose_nll']:.4f} | {100 * math.expm1(r['prose_nll'] - baseline['prose_nll']):+.1f}% |")
lines += ['', 'Margin is ordinary-minus-target mean token NLL on four authored pairs. '
          'It is a relative likelihood diagnostic, not a style-authenticity or quality score. '
          'Temperature controls use identical raw logits for likelihood measurement.', '',
          '# Matched samples', '', 'First seed for each of the four evaluation prompts, all settings. '
          'Every other generation is retained in test.json.', '']
for prompt in data['test_prompts']:
    lines += [f'## {prompt}', '']
    for r in test:
        g = next(g for g in r['generations'] if g['prompt'] == prompt and g['seed'] == data['test_seeds'][0])
        lines += [f"**{r['name']}**", '', g['text'], '']
(ROOT / 'RESULTS.md').write_text('\n'.join(lines).rstrip() + '\n')

candidate = json.loads((ROOT / 'candidate.json').read_text())['name']
comparison = [r for r in test if r['name'] in ('baseline', candidate, 'previous-surreal-sane')]
rng = random.Random(80421)
blind = ['# Blind comparison', '', 'Which continuation best balances deadpan metaphysical satire, '
         'coherence, and enjoyment? “None” is a valid choice. These are the first two test prompts '
         'at the second test seed, selected by position, not quality. Labels are shuffled independently '
         'for each set. Avoid RESULTS.md and blind-key.json until after judging.', '']
key = {}
for number, prompt in enumerate(data['test_prompts'][:2], 1):
    variants = list(comparison)
    rng.shuffle(variants)
    blind += [f'## Set {number}: {prompt}', '']
    for label, r in zip('ABC', variants):
        g = next(g for g in r['generations'] if g['prompt'] == prompt and g['seed'] == data['test_seeds'][1])
        blind += [f'**{number}{label}**', '', g['text'], '']
        key[f'{number}{label}'] = r['name']
(ROOT / 'BLIND.md').write_text('\n'.join(blind).rstrip() + '\n')
(ROOT / 'blind-key.json').write_text(json.dumps(key, indent=2) + '\n')
print('\n'.join(lines[:10]))
