"""Render aggregate measurements and a deterministic, matched sample selection."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
data = json.loads((ROOT / 'test.json').read_text())
results = data['results']
assert len(results) == 4, 'Wait for all variants to finish'
baseline = results[0]
lines = [
    '# Held-out measurements', '',
    '| Variant | Visible name hits | Raw name hits | Other Ка… words | Name-prefix probability ratio | Prose NLL change |',
    '|---|---:|---:|---:|---:|---:|',
]
for result in results:
    summary = result['summary']
    ratio = math.exp(sum(s['family_logp'] - b['family_logp']
                         for s, b in zip(result['scores'], baseline['scores'])) / len(result['scores']))
    lines.append(f"| {result['variant']['name']} | {summary['with_name']}/48 | "
                 f"{summary['raw_with_name']}/48 | {summary['with_other_ka_word']}/48 | "
                 f"{ratio:.2f}× | {result['prose_nll'] - baseline['prose_nll']:+.4f} |")
lines += ['', 'Hits count continuations, not mentions. Probability ratios are geometric means across six prompts, '
          'using the sum of eight surname token-prefix probabilities before filtering/top-p; these are not whole-word '
          'or generation-frequency ratios. NLL is the mean of sentence means on the shared screening prose, '
          'in nats/token, before filtering/top-p. This prose was also used for development selection.', '',
          '# Matched samples', '',
          'Selection: first seed for every test prompt, plus every prompt/seed with a visible surname hit '
          'in any variant. Each selected pair includes all four variants. All other outputs remain in test.json.', '']
chosen = [i for i, g in enumerate(baseline['generations'])
          if g['seed'] == baseline['generations'][0]['seed']
          or any(r['generations'][i]['name_mentions'] for r in results)]
for i in chosen:
    original = baseline['generations'][i]
    lines += [f"## Prompt: {original['prompt']} — seed {original['seed']}", '']
    for result in results:
        g = result['generations'][i]
        assert (g['prompt'], g['seed']) == (original['prompt'], original['seed'])
        lines += [f"**{result['variant']['name']}**", '', g['continuation'] or '(empty after site trimming)', '']
(ROOT / 'RESULTS.md').write_text('\n'.join(lines).rstrip() + '\n')
print('\n'.join(lines[:8]))
