# Porfirievich recovery

Archival recovery and CPU inference for the original Russian text-completion
model «Порфирьевич» by Mikhail Grankin.

The original Pelevin-tuned GPT-2 Medium checkpoint
`m_checkpoint-3365357`, its exact YouTokenToMe tokenizer, and the public-service
sampling behavior have been recovered. The checkpoint runs locally as a pure
autoregressive completer; no chat template or instruction tuning is added.

## Quick start

```bash
git submodule update --init --recursive
./porfirevich-original/download_original.sh
./porfirevich-original/setup_cpu.sh
./porfirevich-original/run.sh 'Однажды вечером'
```

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
