# Porfirievich archival CPU runner

This directory contains the untouched original Porfirievich checkpoint and a
faithful CPU compatibility runner.

## Run

Download the checkpoint, create the local environment, and run a completion:

```bash
./porfirevich-original/download_original.sh
./porfirevich-original/setup_cpu.sh
./porfirevich-original/run.sh 'Однажды вечером' --length 15
```

Add `--seed 42` for repeatable local tests. Omitting it more closely matches
the public service, which did not reset its random seed per request.

For interactive writing, specify how many selection rounds to run:

```bash
./porfirevich-original/run.sh \
  'Однажды вечером' \
  --iterations 5
```

Each round displays the generated candidates. The selected continuation is
appended to all previously accumulated text, and that complete text becomes the
prompt for the next round. `--num-samples` controls the number of choices shown.
The final round's selection is included in the final text.

Selections can also be automated:

```bash
# Always follow the first candidate (deterministic with --seed).
./porfirevich-original/run.sh \
  'Однажды вечером' \
  --iterations 5 \
  --auto-select first \
  --seed 42

# Choose uniformly among the candidates.
./porfirevich-original/run.sh \
  'Однажды вечером' \
  --iterations 5 \
  --auto-select random

# Prefer the longest visible candidate after website-style trimming.
./porfirevich-original/run.sh \
  'Однажды вечером' \
  --iterations 5 \
  --auto-select longest
```

`random` uses a separate selection RNG, so choosing a branch does not consume
the model sampler's random state. Supplying `--seed` makes both generation and
automatic selection repeatable.

The default `latest-site` profile uses the final recovered website behavior.
To reproduce the source as it stood when this checkpoint was uploaded:

```bash
./porfirevich-original/run.sh \
  'Однажды вечером' \
  --profile checkpoint-era
```

Use `./porfirevich-original/run.sh --help` for all options. CPU generation uses
the original uncached sampling loop and can be slow.

## Benchmark

Run a repeatable five-round CPU baseline:

```bash
./porfirevich-original/benchmark.sh \
  --device cpu \
  --threads 6 \
  --warmup 1 \
  --runs 5
```

See `BENCHMARKING.md` for saved JSON reports and an identical ROCm workload.
The experimental Ryzen 5 5600G Vega setup is reproducible with
`setup_rocm_experimental.sh`, but requires the documented 16 GiB BIOS UMA
allocation. Read `ROCM-EXPERIMENT.md` first; this GPU is unsupported by AMD's
production ROCm matrix.

## Contents

- `model/m_checkpoint-3365357/`: untouched original S3 objects
- `download_original.sh`: resumable download with SHA-256 verification
- `benchmark.py` / `benchmark.sh`: repeatable CPU or ROCm performance harness
- `BENCHMARKING.md`: benchmark methodology and comparison instructions
- `ROCM-EXPERIMENT.md`: Vega setup, failures, UMA requirements, and results
- `setup_rocm_experimental.sh`: pinned native `gfx90c` environment setup
- `original-inference-code/`: complete unchanged upstream Git clone
- `run_cpu.py`: modern CPU port preserving the original algorithm
- `environment-lock.txt`: exact package versions used for verified outputs
- `generation-settings.md`: settings, behavior, and historical evidence
- `provenance.md`: artifact origins and S3 metadata
- `SHA256SUMS`: hashes for the original model objects
- `tests/expected-seed42.json`: known-good CPU output fixtures
- `tests/compare_original_sampler.py`: token-level comparison with the recovered sampler
