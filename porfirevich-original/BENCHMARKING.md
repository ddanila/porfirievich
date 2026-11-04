# Repeatable performance measurements

The benchmark loads the untouched checkpoint once, performs unmeasured warmup
rounds, then runs the same generation workload repeatedly. It resets the model
sampling seed before every round and records hashes of all outputs.

## Standard CPU baseline

```bash
./porfirevich-original/benchmark.sh \
  --device cpu \
  --threads 6 \
  --warmup 1 \
  --runs 5 \
  --seed 42 \
  --output porfirevich-original/benchmarks/cpu.json
```

The standard workload is deliberately the original website default:

- prompt: `Однажды вечером`
- 15 generated tokens per candidate
- 3 candidates
- temperature 1.0, top-k 0, top-p 0.9
- final recovered website filtering profile
- original uncached sampling loop

The report separates model-load and warmup time from measured rounds. The main
comparison values are `median_seconds` and
`mean_generated_tokens_per_second`. Generated-token throughput is based on 45
requested tokens per round. Because this faithfully recomputes the complete
context at every token, it should not be compared directly with cached modern
LLM benchmarks.

For lower-noise comparisons, keep the prompt, seed, thread count, background
load, power mode, and model runtime unchanged. Run the benchmark more than once
and compare medians. `outputs_identical` should be true within one device and
runtime.

The standard CPU run and successful native Vega run on this Ryzen 5 5600G
produced:

| Metric | CPU | Vega `gfx90c` |
| --- | ---: | ---: |
| measured rounds | 5 | 5 |
| mean | 3.192 s | 1.655 s |
| median | 3.196 s | 1.655 s |
| population standard deviation | 0.009 s | 0.001 s |
| requested-token throughput | 14.10/s | 27.19/s |
| identical output hashes | 5/5 | 5/5 |

The complete machine-readable reports are
[`benchmarks/cpu-ryzen-5600g.json`](benchmarks/cpu-ryzen-5600g.json) and
[`benchmarks/vega-ryzen-5600g-uma16g.json`](benchmarks/vega-ryzen-5600g-uma16g.json).

## Apple GPU and automatic continuation

Use `--device mps` for Apple Silicon GPU execution. `--device auto` chooses
CUDA/ROCm first, then MPS, then CPU. GPU timing includes synchronization.

```bash
./porfirevich-original/benchmark.sh \
  --device mps --iterations 3 --auto-select random \
  --warmup 1 --runs 2 --seed 42 \
  --output porfirevich-original/benchmarks/mps-random.json
```

Repeat with `--device cpu` and a different output filename to compare.
Each run starts from the original prompt and resets both the model seed and the
independent selection RNG. Within a run, the selected continuation is appended
to the prompt for the next iteration, just like `run.sh --iterations`.
`first` and `longest` selection are also supported.

Reports contain full-run timings, average seconds per iteration, and each
iteration's time, context length, candidates, and selected branch. Throughput
counts all generated candidate tokens before website trimming, not just the
selected continuation: the example generates 3 × 3 × 15 = 135 tokens per run.
Model load, warmup, and terminal output are excluded from measured run time.
`outputs_identical` compares text and selections, excluding timing fields.

On this M3 Pro (12 CPU cores, 18 GPU cores, 36 GB unified memory), the small
three-iteration random-selection benchmark produced:

| Metric | CPU (6 threads) | Apple GPU (MPS) |
| --- | ---: | ---: |
| Measured runs after one warmup | 2 | 2 |
| Mean full run | 5.475 s | 1.149 s |
| Mean per iteration | 1.825 s | 0.383 s |
| Generated candidate tokens/s | 24.66 | 117.44 |
| Repeated outputs identical | yes | yes |

This is about 4.8× faster on MPS for this short workload, not a general speed
guarantee. CPU and MPS generate different text, so later context lengths can
differ even with the same seed. Longer stories cost more with the original
uncached sampler. Reports:
[`cpu-m3-pro-random.json`](benchmarks/cpu-m3-pro-random.json) and
[`mps-m3-pro-random.json`](benchmarks/mps-m3-pro-random.json).

## AMD/ROCm comparison

ROCm-backed PyTorch exposes AMD GPUs through PyTorch's `cuda` API. With a
working ROCm environment, run the identical workload as:

```bash
PORFIRIEVICH_VENV=.venv-rocm \
  ./porfirevich-original/benchmark.sh \
  --device cuda \
  --threads 6 \
  --warmup 1 \
  --runs 5 \
  --seed 42 \
  --output porfirevich-original/benchmarks/vega.json
```

GPU execution is synchronized immediately around every timed region. Reports
include `torch_hip`, the detected GPU name, and allocated/peak device memory.
CPU and GPU seeded text can differ because multinomial and numerical kernels are
device-specific; compare workload settings and speed, not output hashes across
different devices.

The Ryzen 5 5600G's Cezanne/Vega `gfx90c` GPU is not in AMD's current official
ROCm support matrix. Trying it therefore requires an isolated experimental
runtime and may fail or be slower than CPU due to shared DDR4 bandwidth, kernel
support, and launch overhead. Do not replace the verified `.venv`; use a
separate environment for the experiment.

The successful local configuration requires a 16 GiB BIOS UMA frame buffer and
the native nightly device package. Recreate the pinned environment with:

```bash
./porfirevich-original/setup_rocm_experimental.sh
```

A local HSA runtime probe successfully enumerated this APU as `gfx90c`, with
seven compute units and kernel dispatch available. This confirms that the Linux
driver can expose the device, but not that PyTorch kernels will run correctly.
AMD's current production matrix omits `gfx90c`; TheRock's development matrix
lists it as build-passing but not sanity-tested or release-ready. Its documented
`torch[device-*]` extras also omit `gfx90c`, but the nightly index contains
direct `amd-torch-device-gfx90c` and `rocm-sdk-device-gfx90c` packages. The
successful setup pins those native packages instead of using the less reliable
`gfx900` compatibility override.

References:

- [AMD Ryzen ROCm compatibility matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/compatibility/compatibilityryz/native_linux/native_linux_compatibility.html)
- [ROCm system requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.2.3/reference/system-requirements.html)
- [TheRock development GPU matrix](https://github.com/ROCm/TheRock/blob/main/SUPPORTED_GPUS.md)
- [TheRock release and package instructions](https://github.com/ROCm/TheRock/blob/main/RELEASES.md)
- [ROCm issue discussing the 5600G/5700G workaround](https://github.com/ROCm/ROCm/issues/2774)

The local attempts, driver failures, native `gfx90c` package discovery, and UMA
memory change are recorded in [`ROCM-EXPERIMENT.md`](ROCM-EXPERIMENT.md).
