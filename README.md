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
Generation defaults to CPU; Apple GPU acceleration is not enabled.

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
