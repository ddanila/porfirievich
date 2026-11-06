# Metaphysical satire: separate “Pelevinization” pilot

The name is shorthand for the user's requested experiment. The target here is
deadpan metaphysical satire: ordinary institutions treating identity, reality,
freedom, and desire as administrative or commercial products. This is not an
authenticity test for any author's voice. The data consists entirely of new,
assistant-authored examples, with no excerpts from novels or author-name prompts.

This is separate from the surreal-sane and Капустин experiments. The original
checkpoint is preserved; the resulting JSON patch is opt-in and changes only an
in-memory MLP output bias. No defaults or existing candidate files are changed.

## Try it

From the repository root:

```bash
./porfirevich-original/run.sh 'Курьер доставил коробку без обратного адреса.' \
  --device mps --seed 101 --length 60 --num-samples 1 \
  --style-patch porfirevich-original/experiments/metaphysical-satire/candidate.json
```

Omit `--style-patch` for the original model. This patch does not include
Капустин boosting or the previous surreal-sane patch. Normal runner sampling
and iterative continuation options remain available.

## Method

The numerical procedure is adapted from the previous surreal-sane experiment.
Sixteen pairs contrast a mundane scene with a satirical/metaphysical revision.
Average the final eight residual states after blocks 7, 11, and 15 (zero-based),
subtract the ordinary mean from the target mean, and add 0.25, 0.5, or 1 times
that vector to the corresponding MLP output bias. Each edit is restored exactly
after evaluation. No gradients or fine-tuning are used.

Before viewing results, selection was fixed: maximize development target margin
among edits increasing unrelated prose NLL by at most 0.1 nat/token. Stop if no
edit passes. Development has four contrast pairs, four prose passages, and four
generation prompts with two seeds. Evaluation has separate pairs, prose, and
prompts, with three different seeds. Generation uses 60 new tokens, top-p 0.9,
and the existing latest-site filters/trimming; all patch variants use temperature
1. The selected candidate is not retuned on evaluation results.

Controls are the original model and temperatures 0.85/1.15. Evaluation additionally
tests the negative of the selected vector and the recorded original surreal-sane
layer-7/strength-0.5 vector from that experiment's `patches.json`. It does not
load the user's separately edited surreal-sane `candidate.json`. The generic
surrealism control was added before viewing the sweep results or running test.

“Target margin” is mean ordinary NLL minus mean target NLL on paired passages.
It can improve by making ordinary prose worse. The prose NLL guard uses separate
ordinary passages, but neither metric measures coherence, enjoyment, or author
similarity. Temperature controls have identical raw-logit likelihoods, although
their generations differ. The same prompts and seeds are used across settings.

## Findings

Selection chose **layer 7, strength 0.5**. On development, its target margin
rose from 0.1903 to 0.3155 and prose NLL rose by 0.0545 nat/token, passing
the prespecified screening gate. Across 168 continuations (96 development,
72 evaluation), all outputs were kept and no test-driven retuning was done.

On the held-out data:

| Setting | Target margin | Ordinary prose perplexity change |
|---|---:|---:|
| Original | 0.0616 | — |
| New candidate | 0.0995 | +10.9% |
| Opposite direction | 0.0359 | −7.4% |
| Previous surreal-sane | 0.1194 | +8.9% |

The new vector has a small directional effect, but **does not outperform the
previous surreal-sane vector on this target diagnostic** and damages ordinary
prose likelihood slightly more. Its evaluation prose NLL increase is 0.1038
nat/token, just beyond the development screening threshold. This is evidence
against treating the candidate as a better or well-isolated “Pelevin” control.
The vectors have cosine similarity about 0.59 at layer 7, consistent with
overlapping directions, though geometric overlap alone does not establish
behavioral equivalence.

Informal reading of the first seed on every test prompt suggests more abstract
explanation in some candidate outputs, but no reliable improvement in satire
or coherence. For instance, the box prompt becomes an explanation about gifts,
danger, and Hell, but contradicts itself about whether failing to find the
boxes is bad or good. The postcard scene revolves around a “golden nut” with
grammatical errors. The original model also produces metaphysical and satirical
material without steering. These observations are not blind human ratings.

Keep this as a separate exploratory option, not a replacement for the earlier
patch. Both before/after parameter hashes and checkpoint hashes match. Patch
loader validation tests pass; the CLI is also exercised on a recorded prompt.

## Inspect and reproduce

[BLIND.md](BLIND.md) compares original, previous surreal-sane, and this candidate
on two fixed prompt/seed pairs with shuffled labels. Judge it before opening
`blind-key.json` or the identified [RESULTS.md](RESULTS.md). No human ratings
have been collected yet. All generated text, including poor samples, is retained
in `discovery.json` and `test.json`.

```bash
.venv/bin/python porfirevich-original/experiments/metaphysical-satire/experiment.py
.venv/bin/python porfirevich-original/experiments/metaphysical-satire/experiment.py \
  --phase test --patch porfirevich-original/experiments/metaphysical-satire/candidate.json
.venv/bin/python porfirevich-original/experiments/metaphysical-satire/report.py
```

These commands regenerate this experiment's artifacts. `data.json` records the
split and selection rule; `patches.json` records every direction/strength and a
data hash. Each phase verifies the original checkpoint hash and compares all
parameter bytes before/after. Results can differ with hardware/software.

This is a small authored pilot with topic, vocabulary, and length confounds.
It may learn conceptual vocabulary rather than a narrative technique. The
comparison with generic surrealism and opposite steering helps diagnose this,
but does not isolate a unique style feature. Generated text still needs human
judgment; no writing-quality improvement should be inferred from likelihood alone.
