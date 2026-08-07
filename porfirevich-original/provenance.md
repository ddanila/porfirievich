# Porfirievich original checkpoint provenance

Downloaded byte-for-byte from the public anonymous S3 bucket on 2026-08-07:

```text
s3://models.dobro.ai/gpt2/ru/pelevin/m_checkpoint-3365357/
```

Regional HTTPS endpoint used:

```text
https://s3.us-west-2.amazonaws.com/models.dobro.ai/gpt2/ru/pelevin/m_checkpoint-3365357/
```

The original `mgrankin/ru_transformers` deployment instructions identify this
checkpoint as the medium Pelevin model copied to `gpt2/medium` for the service.

## Original S3 object metadata

| File | Bytes | Last modified (UTC) | S3 ETag |
| --- | ---: | --- | --- |
| `config.json` | 674 | 2019-12-16 12:09:27 | `4bb10b5fb97574bd48ce6c03c3f85cb8` |
| `encoder.model` | 724643 | 2019-12-16 12:09:27 | `d812855d9cf8ffb9884d9d420dc538c6` |
| `pytorch_model.bin` | 1725858844 | 2019-12-16 12:09:27 | `b107f76dd11a38332b69f6daa1450795-206` |
| `step.txt` | 7 | 2019-12-16 12:09:27 | `1b8b4000777593036fc779ab2b12cccd` |
| `training_args.bin` | 1420 | 2019-12-16 12:09:28 | `1cc0fbdf9a760beeba11aa31fdbd0dfb` |

Local SHA-256 hashes are recorded in `SHA256SUMS`.

All five downloads were verified against their S3 ETags. The large weight
file's multipart ETag was reproduced using 206 parts of 8 MiB.

## Original source

The complete Git history of `https://github.com/mgrankin/ru_transformers` was
cloned to `original-inference-code/`. The recovered HEAD is:

```text
e698092ed31b340dc39ecea26f1b0be87cdbd022
```

The source clone is preserved unchanged. Local CPU compatibility code lives at
the archive root rather than inside that clone.

## Tokenizer identity

The checkpoint's `model/m_checkpoint-3365357/encoder.model` and the original
repository's `bpe/yt.model` are byte-for-byte identical. Both have SHA-256:

```text
54e9770a67a24eeb072c02b0ed05c96d385af811208e9b3b5b7958069136508f
```
