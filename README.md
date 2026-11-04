# Porfirievich recovery

Archival recovery and CPU inference for the original Russian text-completion
model «Порфирьевич» by Mikhail Grankin.

The original Pelevin-tuned GPT-2 Medium checkpoint
`m_checkpoint-3365357`, its exact YouTokenToMe tokenizer, and the public-service
sampling behavior have been recovered. The checkpoint runs locally as a pure
autoregressive completer; no chat template or instruction tuning is added.

## Quick start

```bash
git clone --recurse-submodules https://github.com/ddanila/porfirievich.git
cd porfirievich
git submodule update --init --recursive
./porfirevich-original/download_original.sh
./porfirevich-original/setup_cpu.sh
./porfirevich-original/run.sh 'Однажды вечером'
```

On macOS, install a current Python and the Xcode command-line tools first
(`brew install python` and `xcode-select --install`, if needed). The same commands
above work on Apple Silicon: setup detects Homebrew Python when Apple's system
Python is too old and installs the macOS PyTorch wheel. You can choose an
interpreter explicitly with `PYTHON=/path/to/python3 ./porfirevich-original/setup_cpu.sh`.
Generation defaults to CPU. Add `--device mps` for the Apple GPU, or `--device auto`
to prefer an available CUDA/ROCm GPU, then MPS, then CPU.

Quick benchmark of three rounds of automatic random selection:

```bash
./porfirevich-original/benchmark.sh --device mps --iterations 3 --auto-select random --runs 2
```

Use `--device cpu` with the same settings to compare CPU speed.

Sampling controls are optional: `--temperature 1.0 --top-p 0.9` preserves the
original defaults. Temperature below 1 favors more likely tokens; above 1 makes
choices more varied. Top-p keeps the smallest set of likely tokens whose total
probability reaches the chosen cutoff, after temperature is applied. See the
[runner documentation](porfirevich-original/README.md#sampling-controls) for examples.

The default `--sampler optimized` uses full-precision KV caching and GPU-friendly
filtering. Use `--sampler original` in either command to reproduce the archival
uncached loop. Both use the same weights, tokenizer, nucleus threshold, and
selection rules. Exact seeded text can differ slightly across execution paths
or devices because of floating-point rounding.

Interactive continuation building:

```bash
./porfirevich-original/run.sh \
  'Однажды вечером' \
  --iterations 5
```

Automatic branching:

```bash
./porfirevich-original/run.sh \
  'Однажды вечером' \
  --iterations 5 \
  --auto-select random \
  --seed 42
```

See [FINDINGS.md](FINDINGS.md) for the evidence and current recovery status,
and [porfirevich-original/README.md](porfirevich-original/README.md) for all
runner options.

For repeatable CPU/GPU performance comparisons, see
[porfirevich-original/BENCHMARKING.md](porfirevich-original/BENCHMARKING.md).

## Repository layout

An exploratory [surreal-but-coherent style experiment](porfirevich-original/experiments/surreal-sane/README.md)
tests reversible, gradient-free bias edits. It found a numerical style shift,
but no established writing-quality improvement. The original weights are
unchanged; the candidate patch is opt-in via `--style-patch` and is not a new
default model.

A separate [read-only Капустин tracing experiment](porfirevich-original/experiments/kapustin-trace/README.md)
maps which activations influence the name in the untouched model. It finds
context-dependent contributors, not a unique character neuron, and includes
layer plots, control comparisons, and before/after integrity hashes.

- `IDEA.md`: original investigation brief
- `FINDINGS.md`: confirmed results, inferences, and remaining questions
- `porfirevich-original/model/`: original model artifacts (large weights ignored by Git)
- `porfirevich-original/original-inference-code/`: original repository as a submodule
- `porfirevich-original/run_cpu.py`: faithful modern CPU compatibility runner
- `porfirevich-original/SHA256SUMS`: hashes of all original checkpoint objects

The 1.7 GB `pytorch_model.bin` remains in the local working tree but is ignored
by ordinary Git. Its exact SHA-256 and S3 multipart ETag are committed, and
`download_original.sh` restores and verifies it on a fresh checkout.

## Credits and licensing

The original **Porfirievich / «Порфирьевич» model and Russian GPT-2 project
were created by Mikhail Grankin**. The original source is
[mgrankin/ru_transformers](https://github.com/mgrankin/ru_transformers), preserved
here as an unchanged, pinned submodule with its license and copyright notices.

This repository contains archival recovery work, instructions, a modern
compatibility runner, and CPU/Vega benchmarks maintained by
[ddanila](https://github.com/ddanila). It is an independent preservation project.
See [CREDITS.md](CREDITS.md) for upstream attribution and the scope of the
[Apache License 2.0](LICENSE). The model and tokenizer retain their original
provenance; this repository does not assign them a new license.
