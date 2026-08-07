# Recovered generation settings

## Directly confirmed in the original source

The public service used `rest.py`, `yt_encoder.py`, and the custom sampling
loop in `run_generation.py`. The original repository is preserved unchanged in
`original-inference-code/` at commit
`e698092ed31b340dc39ecea26f1b0be87cdbd022`.

Core sampling settings in `rest.py`:

| Setting | Value |
| --- | --- |
| temperature | `1.0` |
| top-k | `0` (disabled) |
| top-p | `0.9` |
| default generated tokens | `15` |
| default samples | `3` |
| model context | last `1023` tokens |
| repetition penalty | none |
| explicit stop token | none |

The service always generated the requested number of tokens. It then decoded
the full prompt plus completion, removed the decoded prompt by character count,
and trimmed the visible result with a greedy punctuation regular expression.

Input preprocessing and output cleanup are implemented by the original
YouTokenToMe wrapper. It adds a leading space when absent, spaces certain
punctuation/word boundaries, and maps newlines to the literal `<|n|>` token.

Token ID `1` was suppressed. Newlines were suppressed by default, and two
consecutive newline tokens were always prohibited. No seed was set by the REST
service, so its process RNG state evolved across requests. The generic command
line generator separately defaulted to seed `42`; that was not the web-service
behavior.

## Confirmed historical profiles

The source changed shortly after the checkpoint was uploaded on 2019-12-16:

- `checkpoint-era`: the 2019-12-16 service suppressed token `1`; it returned a
  punctuation-trimmed result only when punctuation was followed by a newline.
  API limits were 500 tokens and 10 samples.
- `latest-site`: on 2019-12-23 the service also suppressed tokens encoding `[` and
  `(` and added trimming at the last punctuation even without a newline. On
  2019-12-27 limits became 60 tokens and 5 samples. Defaults remained 15 and 3.

`run_cpu.py` exposes both behaviors. Its default is `latest-site`, matching the
final recovered public-service source. Use `--profile checkpoint-era` for the
implementation contemporaneous with the checkpoint upload.

## Modern compatibility boundary

The checkpoint and tokenizer files remain untouched. `run_cpu.py` is a CPU port
of the original tokenizer transformations, sampling loop, filtering, and output
postprocessing. It loads the legacy PyTorch state dictionary with a current CPU
runtime; therefore seeded output need not be bit-identical to CUDA/PyTorch 1.3
sampling from 2019 even though the algorithm and probability filters are the
same.

The recovered 2019 sampler and the CPU port were also run side-by-side with the
same checkpoint, tokenizer, prompt, filters, and seed. Their generated token IDs
were identical. The executable comparison is preserved in
`tests/compare_original_sampler.py`; known-good text outputs are in
`tests/expected-seed42.json`.

## Local iterative picker

`--iterations N` is a local convenience interface, not a recovered feature of
the historical web service. Every round still uses the recovered tokenizer,
sampler, filters, and postprocessing unchanged. The selected visible reply is
appended verbatim to the accumulated prompt before the next round.
Automatic selection supports `first`, uniform `random`, and `longest` visible
reply modes. Its random generator is separate from PyTorch's sampling generator,
so selection itself does not perturb subsequent model sampling state.
