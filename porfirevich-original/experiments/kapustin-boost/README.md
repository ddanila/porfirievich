# Increasing Капустин appearances

This experiment compares prefix-aware sampling bias with temporary activation
amplification in the original checkpoint. It does not load the surreal-sane
patch, edit weights, or change runner defaults.

## Outcome

On the 48 held-out prompt/seed pairs per setting:

| Setting | Visible surname matches | Person appearances, manually read | Other Ка… words |
|---|---:|---:|---:|
| Original model | 0/48 | 0/48 | 6/48 |
| Prefix-aware bias +3 | 6/48 | 5/48 | 9/48 |
| MLP 9 ×1.5 | 0/48 | 0/48 | 4/48 |
| Only boost initial Ка by +3 | 2/48 | 1/48 | 12/48 |

Prefix-aware bias is the useful intervention in this pilot. Five hits refer
to people; the sixth is the place Капустин Яр. Four person hits came from the
envelope/recipient prompt, and one from the police-station prompt. Both hits
from the initial-token control are also present in the prefix-aware setting,
with identical text. Thus continuation bias adds four hits in this sample,
all in the envelope prompt, while producing fewer continuations with other
Ка… words than the initial-token control. This is encouraging but strongly
prompt-dependent evidence, not a universal success rate.

Internal amplification raised the geometric mean surname-prefix probability
only 1.04× on evaluation prompts and generated no actual mentions. The
prefix-aware and initial-token controls raised this diagnostic 258.30× and
19.37× respectively, from low initial probabilities. These numbers are not
appearance-frequency multipliers. The shared prose NLL increased by 0.0252
nat/token for internal amplification and 0.0035 for either sampling bias.

Manual reading of every positive sample and the first seed for each prompt
shows a real tradeoff: the bias introduces people in locally recognizable
roles, but often overuses the surname. For example, the envelope/seed-19
continuation repeats it four times in a short paragraph; seed 37 contains
«этого Капустина — Капустина так звали». Grammar and narrative continuity
remain uneven, including «экстрасенсу и протезе Капустину» and changing names
on an envelope. These are informal observations, not blind quality scores.
The tiny repeated-four-token metric misses much of this surname repetition.

My assessment: use prefix-aware bias for deliberately inviting Капустин into
a scene, while expecting occasional repetition and Ка-word spillover. This
pilot provides no reason to prefer broad MLP amplification for that purpose,
and no evidence that either intervention improves overall writing quality.
The original model remains the default. Across development and evaluation,
320 continuations were generated; both checkpoint and parameter integrity
checks passed.

## Method

`design.json` was saved before running the experiment. Development used four
prompts × four seeds per setting; evaluation used six different prompts × eight
different seeds per setting. Every continuation has 60 sampled tokens,
temperature 1, top-p 0.9, and the production runner's latest-site filters and
trimming. Each setting uses the same prompt/seed pairs. The baseline loop is
checked against the production runner.

The sampling intervention adds +1.5 or +3 to next-token logits along the
longest matching prefix of eight grammatical forms of Капустин. This covers
`▁Ка → пусти → н/на/ну/ным/не/ны/ных/ными`. Starting a name necessarily boosts the
shared `▁Ка` token: this is prefix-aware bias, not exact whole-sequence
reweighting. A first-token-only control measures whether the extra continuation
bias does anything beyond encouraging words beginning with Ка.

The internal interventions multiply MLP layer 9 or 16 output by 1.25 or 1.5,
or neuron 3384 in layer 9 by 2 before its output projection. Layer numbering
is zero-based. These components came from the previous tracing experiment;
attenuation evidence does not guarantee that amplification has the opposite
effect. Hooks are removed after each setting.

Selection uses development results only: maximize visible surname hits for
sampling bias, breaking ties toward the weaker setting. For internal steering,
maximize the mean log probability of the surname family relative to five
control names, among settings increasing ordinary prose NLL by at most 0.1
nat/token. This selected **prefix bias +3** and **MLP 9 ×1.5**. The selection
criterion and thresholds were not retuned on evaluation results.

## Measurements and limitations

See [RESULTS.md](RESULTS.md) for measurements and matched examples. Full
development and evaluation outputs, including untrimmed continuations, are in
`discovery.json` and `test.json`.

The primary measure counts continuations containing a whole surname form after
site trimming, excluding the prompt. It can count **Капустин Яр**, a place,
as well as people; semantic interpretation requires reading the samples.
Probability diagnostics sum the eight token-prefix probabilities before
filtering/top-p and do not enforce a following word boundary. They are not
the probability of a character appearing anywhere in a story.

Other capitalized Ка… words, selected other surnames, and repeated four-token
sequences are recorded as side-effect proxies. Ordinary prose NLL is measured
before filtering/top-p on six authored sentences, averaged by sentence. The
same prose is reused from development screening; it is not an independent
quality benchmark. These proxies do not establish coherence or writing quality.
The dataset is small and authored; counts should not be read as reliable rates
for every prompt or long story. No setting teaches the model a stable character.

## Try it interactively

From the repository root:

```bash
.venv/bin/python porfirevich-original/experiments/kapustin-boost/sample.py \
  'На конверте было написано имя получателя:' --strength 3 --seed 19
```

This reproduces a positive example on the experiment's Mac/software setup.
Omit `--seed` for fresh samples, or add `--count 5` for several alternatives.
`--length 60` is the default number of new tokens; longer lengths are allowed.
`--strength 0` disables the bias for comparison. Strength 1.5 is gentler, but
yielded no visible mentions in the small development sweep. Increased strength
does not guarantee a mention and can encourage repetition and other Ка… words.

The sampler retains the experiment's temperature 1, top-p 0.9, and site trimming.
It uses the original weights, automatically selects the device, and does not
apply the surreal-sane patch. Use `--device cpu` or `--device mps` to override.

## Reproduce

From the repository root, with the original model installed:

```bash
.venv/bin/python porfirevich-original/experiments/kapustin-boost/test_bias.py
.venv/bin/python porfirevich-original/experiments/kapustin-boost/experiment.py
.venv/bin/python porfirevich-original/experiments/kapustin-boost/experiment.py --phase test
.venv/bin/python porfirevich-original/experiments/kapustin-boost/report.py
```

The experiment uses the automatically selected device (MPS on this Mac).
Each phase verifies the original checkpoint SHA-256 and compares all model
parameter bytes before/after; see the two `*-integrity.json` files. Results can
differ across backends or software versions despite matched seeds. Recorded
seconds include probability scoring and generation, not just decoding speed.
