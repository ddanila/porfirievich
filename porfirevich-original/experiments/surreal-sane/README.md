# Surreal but coherent: exploratory bias-edit pilot

## Outcome

**A measurable preference shift, but no established improvement in writing.**
The small selected edit raises a paired surreal-prose likelihood score on both
 development and held-out examples. Generated continuations still contain
ungrammatical phrases, abrupt changes of subject, and disconnected images.
The selected edit worsens ordinary-prose perplexity by approximately 5.6% on
four held-out passages. It is experimental and is **not enabled by default**.
In a subsequent blind comparison, the user preferred the candidate on both
shown prompts, describing it as “surreal but rather consistent.” This is
encouraging personal-preference evidence, but the pilot does not establish a
reliable general-purpose “surreal but sane” control.

The original checkpoint and its config/tokenizer/metadata were verified against
`SHA256SUMS` after the experiment. No original model file was edited. Each
variant changed only an in-memory MLP output bias, which was restored exactly
after evaluation. The selected patch is a separate JSON file containing 1,024
float32 bias increments and the original checkpoint SHA-256. The CLI verifies
that hash before applying a patch in memory.

## Hypothesis and method

Inspired by [weight-space programming](https://rikiyat.github.io/blog/weight-space-programming/)
and [contrastive activation steering](https://arxiv.org/abs/2308.10248).
This pilot uses forward-pass measurements rather than claiming to interpret
raw weights directly; no gradients, optimizer, or fine-tuning are used.

Sixteen author-written Russian pairs contrast ordinary coherent scenes with
surreal scenes that retain a causal thread. For each passage, average the final
eight residual states after transformer blocks 7, 11, and 15 (zero-based).
At each block, subtract the mean ordinary vector from the mean surreal vector.
Test adding 0.5, 1.0, or 2.0 times that difference to the block's MLP output bias.
The model is in evaluation mode, so this adds the direction to every position
at that block, without dropout. It is compatible with the KV cache and leaves
all other parameters and the on-disk checkpoint unchanged.

The discovery sweep has nine edits, an untouched baseline, and temperature-only
controls at 0.85 and 1.15. All weight edits use temperature 1.0 and top-p 0.9.
The data is a tiny, hand-authored pilot: lexical/topic/length confounds remain,
and no direction should be interpreted as an isolated “surrealism neuron.”

## Evaluation and selection

Each split has four contrast pairs, four unrelated ordinary prose passages,
and four generation prompts with seeds 42 and 137. Each generation requests
60 tokens and one candidate with the normal website trimming. All outputs,
including bad ones, are retained in JSON. Generation uses the same prompts and
seeds for every variant, and no best-of selection is performed.

“Style margin” is mean ordinary NLL minus surreal NLL across paired passages;
higher means a relative shift toward the surreal examples. It can improve by
making ordinary language worse and is **not** a coherence or quality score.
“Prose NLL” is the mean of per-passage token-average negative log likelihoods
on unrelated ordinary prose; lower is better. Likelihoods are measured from
unscaled model logits, so temperature-only controls have identical likelihood
scores but different generated text. Scores are in nats/token, using this
checkpoint's tokenizer, and are not comparable to other models' tokenizers.

The provisional candidate was chosen after the exploratory discovery sweep:
maximize style margin among edits increasing development prose NLL by no more
than 0.1 nat/token. This chose layer 7, strength 0.5. Its discovery prose NLL
increase was 0.02194 nat/token. Selection happened before evaluating the held-out
split, and the held-out results were not used to tune the edit further.
This was not a preregistered or statistically powered study.

### Development

| Variant | Style margin (higher) | Ordinary prose NLL (lower) |
| --- | ---: | ---: |
| baseline | -0.1738 | 4.1879 |
| temperature-0.85 | -0.1738 | 4.1879 |
| temperature-1.15 | -0.1738 | 4.1879 |
| layer7-strength0.5 | -0.0839 | 4.2099 |
| layer7-strength1.0 | -0.0115 | 4.3073 |
| layer7-strength2.0 | 0.0452 | 4.7866 |
| layer11-strength0.5 | -0.0955 | 4.2479 |
| layer11-strength1.0 | -0.0272 | 4.3874 |
| layer11-strength2.0 | 0.0727 | 4.9487 |
| layer15-strength0.5 | -0.1126 | 4.2190 |
| layer15-strength1.0 | -0.0201 | 4.3201 |
| layer15-strength2.0 | 0.2173 | 4.8694 |

### Held-out

| Variant | Style margin (higher) | Ordinary prose NLL (lower) |
| --- | ---: | ---: |
| baseline | -0.3334 | 4.5503 |
| temperature-0.85 | -0.3334 | 4.5503 |
| temperature-1.15 | -0.3334 | 4.5503 |
| layer7-strength0.5 | -0.2121 | 4.6048 |

The held-out margin rises from -0.3334 to -0.2121, while prose NLL rises from
4.5503 to 4.6048. The latter corresponds to a 5.6% perplexity increase.
The strongest edits in discovery were substantially more damaging. The gentle
candidate is a useful reversible test artifact, not a recommended replacement.

One selected development prompt/seed was shown to the user with variant names
hidden. This is a small preference elicitation, not a blinded randomized study.
That development question was not answered. A subsequent comparison used two
held-out prompts (library and market), seed 137, and three variants: baseline,
temperature 0.85, and the candidate. Labels were shuffled independently per set
using Random(60193), and identities were withheld until after the response.
The user chose 1B and 2B, both the candidate: “surreal but rather consistent.”
This is two preferences from one person at one seed, not statistical evidence
of general superiority. The full samples, label key, and verbatim response are
in `human-review.json`. All eight generations for each variant are in
`discovery.json` and `test.json`. Future tuning must use new evaluation prompts
if the current feedback informs the next edit.

## Try or reproduce

From the repository root, try the opt-in candidate:

```bash
./porfirevich-original/run.sh 'В комнате стоял старый шкаф.' \
  --device mps --seed 137 --length 60 --num-samples 1 \
  --style-patch porfirevich-original/experiments/surreal-sane/candidate.json
```

Omit `--style-patch` for the untouched model. Sampling controls and automatic
continuation selection remain available. No command here saves edited full
weights or changes the default model.

Reproduce the discovery sweep and the fixed candidate's held-out evaluation:

```bash
.venv/bin/python porfirevich-original/experiments/surreal-sane/experiment.py
.venv/bin/python porfirevich-original/experiments/surreal-sane/experiment.py \
  --phase test --patch porfirevich-original/experiments/surreal-sane/candidate.json
```

The first command regenerates `patches.json` and `discovery.json`; the second
regenerates `test.json`. `candidate.json` preserves the exact evaluated edit.
`data.json` contains all split assignments. `patches.json` includes the data
hash and every tested edit. Reproduction uses the installed runtime and
prefers an available GPU; sampling can differ on another device/runtime.

Validation: patch CLI output matches the experiment, a 60-token untouched
baseline matches the archival original sampler, patch tests reject invalid
layers/shapes/nonfinite values/checkpoint hashes before changing parameters,
and checkpoint bytes remain unchanged. The ordinary sampling tests also pass.

Next useful step: extend blind human ratings to more prompts and seeds, and construct
more varied, tightly matched contrast pairs if this style axis is worth
pursuing. A second coherence direction or a small curated fine-tune would be
separate experiments, not established fixes from this pilot.
