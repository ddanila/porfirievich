# Ryzen 5 5600G Vega/ROCm experiment

Status date: 2026-08-07

This log records the unsupported attempt to run the recovered model on the
Ryzen 5 5600G's integrated Cezanne/Vega GPU. It is deliberately separate from
the verified CPU environment and historical model findings.

## Hardware and environment

- GPU: AMD Cezanne integrated graphics, native target `gfx90c`, 7 compute units
- motherboard: ASRock B450M Pro4, BIOS P10.60 (2026-04-17)
- kernel: Ubuntu `7.0.0-29-generic`, inbox `amdgpu` driver
- experimental environment: `.venv-rocm` (ignored by Git), Python 3.12.13
- PyTorch: `2.12.0+rocm7.15.0a20260728`
- ROCm: `7.15.0a20260728`
- device packs: `amd-torch-device-gfx90c` and
  `rocm-sdk-device-gfx90c`, both dated 2026-07-28

The verified `.venv` remains the independent CPU-only environment.

## Results before increasing UMA memory

The initial BIOS allocation was 512 MiB VRAM, with a 1 GiB GART and roughly
30 GiB of GTT/shared memory exposed by the driver.

1. The prebuilt `gfx900` device pack with
   `HSA_OVERRIDE_GFX_VERSION=9.0.0` successfully enumerated the GPU and ran a
   small 1024-by-1024 matrix multiplication.
2. Moving the 1.4 GiB FP32 model to the device caused an `sdma0` timeout,
   GPU page faults, repeated GPU resets, and temporary display loss.
3. Retrying with `HSA_ENABLE_SDMA=0` still wedged and reset the GPU while
   transferring the model. This ruled out SDMA as the only cause.
4. A native, not-yet-documented `gfx90c` PyTorch device pack was found in
   TheRock's nightly package index and replaced the `gfx900` workaround.
5. Testing that pack after several resets faulted in `libamdhip64.so.7`; a
   reboot was required to restore trustworthy driver state.
6. On a clean boot, the native pack successfully ran elementwise kernels and a
   1024-by-1024 matrix multiplication in 0.043 seconds.
7. The first isolated 128 MiB host-to-device transfer then caused a complete
   machine freeze. The journal ended with repeated
   `amdgpu_irq_handle_ih_soft` CPU-hog warnings and contained no successful GPU
   reset, so the machine was manually restarted.

## UMA changes and transfer boundary

The BIOS UMA frame-buffer allocation was first changed from 512 MiB to 3 GiB.
With 3 GiB exposed as native VRAM, isolated 128, 256, and 512 MiB transfers all
completed and verified successfully. Their respective transfer times were
1.95, 2.87, and 4.43 seconds.

The subsequent 1 GiB test completely froze the machine; the planned 1.5 GiB
test never started. The final journal messages show SVM range mapping and retry
fault work inside `amdgpu`, an escalating interrupt-workqueue storm, and soft
lockups in the KMS display thread and worker CPUs. The driver never reached its
normal GPU-reset path.

The UMA allocation was then increased to 8 GiB. The formerly fatal 1 GiB
transfer completed and verified in 11.61 seconds, but a 1.5 GiB transfer caused
the same interrupt-workqueue storm and hard freeze. The journal recorded a
soft-locked worker CPU but no successful GPU reset.

The UMA allocation was then increased again to 16 GiB. The current kernel
reports:

```text
VRAM: 16384M
BAR: 16384M
GART: 1024M
GTT: 23508M
```

The system continues to boot in legacy mode with the 16 GiB UMA setting; no CSM
change was required.

At 16 GiB UMA, the formerly fatal 1.5 GiB transfer completed and verified in
13.88 seconds. The complete FP32 checkpoint then transferred in 12.98 seconds,
used 1.44 GiB for weights, and completed a two-token forward pass in 0.52
seconds. End-to-end tokenization, filtering, multinomial sampling, and decoding
also passed.

## Successful benchmark

The standard 15-token by 3-candidate workload completed for one warmup and five
measured rounds:

| Metric | CPU | Vega `gfx90c` |
| --- | ---: | ---: |
| mean seconds | 3.192 | 1.655 |
| median seconds | 3.196 | 1.655 |
| requested-token throughput | 14.10/s | 27.19/s |
| population standard deviation | 0.009 s | 0.001 s |
| identical hashes within runtime | 5/5 | 5/5 |

Steady-state Vega generation was 1.93 times as fast, reducing round latency by
48.2%. Loading is substantially slower because the FP32 weights must first be
transferred: 14.39 seconds in the saved benchmark versus 1.27 seconds for CPU.
Peak GPU allocation was 1.52 GiB.

The kernel logged an intensive `amdgpu_irq_handle_ih_soft` workqueue during
large transfers, but it stopped after the process exited. There were no GPU
resets, page faults, or soft lockups in the successful 16 GiB run. This remains
an unsupported experimental configuration.

Machine-readable reports are saved as
`benchmarks/cpu-ryzen-5600g.json` and
`benchmarks/vega-ryzen-5600g-uma16g.json`.
