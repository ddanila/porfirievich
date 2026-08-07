# Current recovery findings

Status date: 2026-08-07

This document distinguishes direct evidence from inference and open questions.
Detailed object metadata and hashes are in
`porfirevich-original/provenance.md`; generation history is in
`porfirevich-original/generation-settings.md`.

## Recovered and directly confirmed

### Original checkpoint

The anonymous S3 bucket `models.dobro.ai` remains live through its regional AWS
endpoint even though the custom HTTPS frontend returns `502`. The exact target
directory is publicly listable and downloadable:

```text
s3://models.dobro.ai/gpt2/ru/pelevin/m_checkpoint-3365357/
```

It contains five objects:

| Object | Bytes | SHA-256 |
| --- | ---: | --- |
| `config.json` | 674 | `f52f31f5693b586cd6046e5e037c50177ddcc3981f7595b408bcc16a205a2a38` |
| `encoder.model` | 724643 | `54e9770a67a24eeb072c02b0ed05c96d385af811208e9b3b5b7958069136508f` |
| `pytorch_model.bin` | 1725858844 | `c508fdf6db765a4b1cf11380038233d7bd969ce17a8055368a9095c6dc10089d` |
| `step.txt` | 7 | `71a55cefe12467366ae1b99eafb59d7c21392203fe17acd7daafc3758f65eeb3` |
| `training_args.bin` | 1420 | `ca373c704488156ce106c8b02ca97efdefc1c47ce9ce30101875f34770b35ebb` |

All local files matched their S3 ETags. The weight file's multipart ETag was
reproduced as `b107f76dd11a38332b69f6daa1450795-206` using 206 parts of 8 MiB.

### Model architecture

The untouched checkpoint loads as `GPT2LMHeadModel` with:

- 354,823,168 parameters
- 24 transformer layers
- 16 attention heads
- embedding width 1024
- vocabulary size 50,257
- context size 1024
- FP32 weights

This is the GPT-2 Medium architecture described by the original project.

### Production identity

The original repository's deployment instructions copy:

```bash
cp -R gpt2/pelevin/m_checkpoint-3365357 gpt2/medium
```

The service then loads `gpt2/medium`. Together with the README statement that
the website model is in the Pelevin folder, this directly identifies the
recovered checkpoint as the deployed Pelevin medium model.

### Exact tokenizer

The checkpoint's `encoder.model` is byte-for-byte identical to the original
repository's `bpe/yt.model`. Both have SHA-256:

```text
54e9770a67a24eeb072c02b0ed05c96d385af811208e9b3b5b7958069136508f
```

The source confirms YouTokenToMe BPE with a 50,257-token vocabulary and records
the exact input preprocessing and output cleanup transformations.

### Original source and history

The complete `mgrankin/ru_transformers` Git history is preserved as the
`porfirevich-original/original-inference-code` submodule, pinned at:

```text
e698092ed31b340dc39ecea26f1b0be87cdbd022
```

The clone contains 536 commits. Repository history was used to date changes in
the public service rather than treating its final source as a single timeless
configuration.

### Public-service generation behavior

Directly recovered core settings:

| Setting | Value |
| --- | --- |
| temperature | 1.0 |
| top-k | 0 (disabled) |
| top-p | 0.9 |
| default generated tokens | 15 |
| default candidates | 3 |
| repetition penalty | none |
| stop token | none |
| model input window | most recent 1023 tokens |

The service generated the full requested token count and then trimmed displayed
text with greedy punctuation regular expressions. It suppressed token ID `1`,
suppressed newlines by default, and always prevented two consecutive newline
tokens. The REST service did not reset a seed per request; its process RNG state
advanced naturally.

Two source-backed profiles are exposed:

- `checkpoint-era`: source contemporaneous with the 2019-12-16 S3 upload.
- `latest-site`: final recovered source, including bracket-token suppression,
  punctuation fallback trimming, and tighter request limits added later in
  December 2019.

### Final Pelevin training invocation

`training_args.bin` was decoded with PyTorch's restricted `weights_only` loader.
Confirmed notable values include:

- training data: `./data/pelevin_all.txt`
- validation data: `./data/pelevin_valid.txt`
- model/output: `output/pelevin_m`
- tokenizer: `bpe/yt.model`, class `YTEncoder`
- block size: 1024
- epochs: 3
- learning rate: `4e-5`
- per-device batch size: 2 on 8 XLA devices
- warmup samples: 6400
- unfreeze level: `-1` (all layers)
- training seed: 17013
- checkpoint step: 3,365,357

### Local CPU reproduction

The checkpoint loads unchanged with the pinned modern CPU environment. On the
investigation machine (Ryzen 5 5600G):

- 15 tokens × 3 candidates completed in approximately 7.3 seconds
- peak resident memory was approximately 2.1 GiB

For an identical prompt, filters, seed, and model, the recovered 2019
`sample_sequence` and the compatibility port generated identical token IDs.
Known-good outputs and the executable comparison are committed under
`porfirevich-original/tests/`.

## Local extensions, not historical behavior

The iterative picker and `--auto-select` modes are local conveniences. They do
not alter generation within a round:

- interactive selection
- `first`: always follow candidate one
- `random`: uniform selection using an RNG separate from model sampling
- `longest`: select the longest visible post-trimmed candidate

The selected visible continuation is appended verbatim to the accumulated text,
which becomes the next prompt.

## Supported inferences

- The remembered literary behavior likely comes from the combination of the
  Russian/literary/Pelevin training stages, exact tokenizer transformations,
  nucleus sampling, token suppression, and visible-text trimming—not solely
  from the neural weights.
- Seeded CPU output is a reliable local regression fixture, but it is not
  guaranteed to be bit-identical to the 2019 CUDA/PyTorch 1.3 deployment because
  device kernels and multinomial implementations can differ.

## Remaining investigation

- Recover the original website frontend and determine what controls it exposed
  at different dates.
- Search archived deployments for any service wrapper newer than the public
  repository HEAD.
- Preserve the rest of the still-live S3 model tree and its intermediate
  checkpoints if desired.
- Determine exact versions of PyTorch, Transformers, YouTokenToMe, CUDA, and
  Apex used for each production deployment.
- Recover exact large-corpus and literary-corpus snapshots and preprocessing
  provenance if faithful retraining is ever needed.
- Compare outputs against surviving screenshots, logs, or user-recorded
  Porfirievich completions from the original website.
