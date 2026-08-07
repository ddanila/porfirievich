# Porfirievich: investigation and recovery task

I want to investigate and, ideally, locally run the original Russian text-generation model **«Порфирьевич» (Porfirievich)** created by Mikhail Grankin around 2019.

The goal is **not** merely to find a modern Russian GPT model or something vaguely similar. I specifically want to recover the original Porfirievich model if possible, together with its tokenizer and generation settings, and reproduce its characteristic text-completion behaviour.

Porfirievich produced remarkably distinctive Russian literary prose. In particular, its behaviour was quite different from modern instruction/chat LLMs: given the beginning of a text, it simply continued it. Preserving that behaviour is important. Do not attempt to turn it into a chatbot or instruction-following model.

The first priority is therefore **digital archaeology and recovery**, not retraining. Retraining/reconstruction should only be considered if the original checkpoint can no longer be found.

## What is already known

The original project is:

```text
mgrankin/ru_transformers
```

Original GitHub repository:

```text
https://github.com/mgrankin/ru_transformers
```

Porfirievich appears to have used **GPT-2 Medium**, approximately **355M parameters**.

The strongest identifier found for the actual production model is:

```text
pelevin/m_checkpoint-3365357
```

The original deployment instructions apparently copied this checkpoint into the normal GPT-2 `medium` directory:

```bash
cp -R gpt2/pelevin/m_checkpoint-3365357 gpt2/medium
```

This is strong evidence that `m_checkpoint-3365357` was the Pelevin-tuned GPT-2 Medium checkpoint actually used by the public Porfirievich service.

## Apparent training history

The model was **not simply stock English GPT-2 fine-tuned on Pelevin**.

The approximate training pipeline appears to have been:

```text
OpenAI GPT-2
    ↓
large Russian corpus (~230 GB)
    ↓
selected Russian literary corpus (~500 MB)
    ↓
Pelevin-specific fine-tuning
    ↓
Porfirievich production checkpoint
```

The large ~230 GB Russian corpus appears to have been generated from a **large FB2 ebook library**, apparently obtained from Booktracker or a related book-tracker source.

The repository contains corpus-processing tools including:

```text
corpus/corpus.ipynb
FB2_2_txt.xsl
```

These should be inspected carefully because they may document the exact preprocessing performed on the books.

A subsequent literary corpus was assembled from Russian literature. Authors explicitly associated with the project include:

```text
Dostoevsky
Tolstoy
Pushkin
Bulgakov
Gogol
Pelevin
```

Finally, a Pelevin-specific fine-tuning stage produced the model used by Porfirievich.

There appears to have been a progressive-unfreezing/training sequence approximately:

```text
0 → 1 → 2 → 7 → -1
```

The repository should be examined to determine exactly what these values mean and how they were used.

## Tokenizer

The project apparently did **not** simply use the stock English GPT-2 tokenizer for Russian.

It appears to have used a Russian BPE tokenizer based on **YouTokenToMe (YTTM)** with a vocabulary of approximately:

```text
50,257 tokens
```

Recovering the **exact tokenizer files** is extremely important.

Even if the neural-network weights can be reconstructed, using a different tokenizer could substantially change the model's behaviour. Therefore the tokenizer should be considered part of the model artifact, not an implementation detail.

## Original model hosting

The original README instructed users to retrieve models anonymously from an S3 bucket:

```bash
aws s3 sync --no-sign-request s3://models.dobro.ai/gpt2/ru all
```

There is/was also an HTTPS endpoint:

```text
https://models.dobro.ai/gpt2/ru/
```

As of August 2026, an HTTP request to that endpoint returns:

```text
502
```

However, failure of the HTTP frontend does **not necessarily prove that the underlying S3 bucket or objects are gone**.

The S3 API should therefore be tested directly:

```bash
aws s3 ls --no-sign-request s3://models.dobro.ai/gpt2/ru/
```

Then specifically:

```bash
aws s3 ls --no-sign-request \
  s3://models.dobro.ai/gpt2/ru/pelevin/m_checkpoint-3365357/
```

If it still exists, retrieve it immediately:

```bash
aws s3 sync --no-sign-request \
  s3://models.dobro.ai/gpt2/ru/pelevin/m_checkpoint-3365357/ \
  ./porfirevich
```

If any part of the original S3 tree remains accessible, enumerate and preserve it before doing anything else. Tokenizer files, configuration files, intermediate checkpoints and generation code may be as valuable as the final weights.

## Recovery status

So far, no confirmed live mirror of the exact:

```text
pelevin/m_checkpoint-3365357
```

checkpoint has been identified.

This needs substantially more investigation.

Useful exact search strings include:

```text
"m_checkpoint-3365357"
"pelevin/m_checkpoint-3365357"
"models.dobro.ai/gpt2/ru"
"ru_transformers" "pelevin"
"Porfirevich" checkpoint
"Porfirievich" checkpoint
"Порфирьевич" checkpoint
```

Search broadly across:

- GitHub repositories and forks
- GitHub issues, discussions and commit histories
- Hugging Face
- Internet Archive / archive.org
- Wayback Machine
- old Google Colab notebooks
- old Russian ML forums and blogs
- Reddit
- Telegram/VK references where searchable
- cached S3 URLs
- personal mirrors
- old Docker images
- package archives
- torrent/index archives
- forks and derivatives of `mgrankin/ru_transformers`

In particular, **enumerate every useful fork of `mgrankin/ru_transformers`**.

Look not only for committed model binaries but also for:

- changed download URLs
- mirrors
- shell scripts
- notebook output
- comments
- issue discussions
- Dockerfiles
- cached filenames
- hashes/checksums
- alternative S3 endpoints
- people saying they downloaded the models
- exact filenames inside the checkpoint directory

Finding exact filenames and hashes would make further archival searches considerably easier.

## Recover the original inference implementation too

Recovering the weights alone is not sufficient.

The original **generation/inference code and sampling parameters** should be identified.

Determine, as precisely as possible:

- temperature
- top-k
- top-p / nucleus sampling, if used
- repetition penalty or equivalent
- context length
- number of generated tokens
- stop conditions
- preprocessing of input
- postprocessing of output
- random-seed behaviour
- whether any tokens were explicitly suppressed
- whether the public website used different parameters from repository examples

Sampling is particularly important here. A substantial part of the remembered "Porfirievich style" may come from the combination of the model and its sampling configuration.

Look for the source of the original web service/API as well as the training repository.

There are historical references to infrastructure under domains such as:

```text
dobro.ai
gpt.dobro.ai
pelevin.gpt.dobro.ai
models.dobro.ai
```

Investigate archived versions of those services and their client-side code. Old JavaScript may reveal API parameters even if the backend is gone.

## Why preserving the original model behaviour matters

Porfirievich was essentially a **pure autoregressive text-completion model**.

Its conceptual interface was:

```text
Russian prose
    ↓
predict next token
    ↓
predict next token
    ↓
...
```

It was not:

```text
user message
    ↓
determine user intention
    ↓
construct a helpful answer
```

This distinction is central to the project.

Modern instruction-tuned/RLHF/chat models are much more capable generally, but their learned behaviour strongly biases them toward answering, explaining, structuring and helping.

Porfirievich instead operated directly inside a distribution dominated by Russian prose and literature and was finally tuned toward Pelevin.

That combination may explain why its creative Russian continuations could have a peculiar literary quality that is difficult to reproduce simply by prompting a modern general-purpose LLM.

Therefore **do not introduce instruction tuning, chat templates, system prompts or other modern LLM conventions when reproducing it**.

The first target should be faithful next-token completion.

## If the original checkpoint is found

Do **not** immediately convert or modify it.

First preserve the original artifacts byte-for-byte.

Record:

```text
original filenames
file sizes
SHA-256 hashes
directory structure
format/framework
model configuration
tokenizer files
associated metadata
```

Make an untouched archival copy.

Then determine the minimum environment required to load it.

The initial goal is **CPU-only inference using ordinary system RAM**. GPU acceleration and optimization can be investigated later.

Prefer reproducing output with the original implementation first. Only after that works should the model be converted to modern formats/runtimes.

Create a small reproducibility test set containing several Russian prompts and save generated outputs from the original implementation. Those outputs can later be used to validate conversions.

## If the original checkpoint cannot be recovered

Only after exhausting recovery possibilities should reconstruction/retraining be considered.

A surprisingly large amount of the original recipe appears to survive:

```text
architecture:
GPT-2 Medium, ~355M parameters

tokenizer:
Russian YTTM BPE, ~50,257 vocabulary

base Russian corpus:
~230 GB, apparently derived from an FB2 ebook library

literary corpus:
~500 MB

authors:
Dostoevsky
Tolstoy
Pushkin
Bulgakov
Gogol
Pelevin

final specialization:
Pelevin

training code:
mgrankin/ru_transformers

FB2 conversion tooling:
present in repository

progressive training/unfreezing:
apparently documented
```

Before attempting retraining, determine exactly what is still missing.

Likely missing/research-required pieces include:

```text
exact 230 GB corpus snapshot
exact list/version of books
exact corpus filtering
exact preprocessing
exact tokenizer model
exact tokenizer training corpus
exact initial GPT-2 checkpoint
exact training hyperparameters
exact intermediate checkpoints
exact Pelevin corpus
exact final checkpoint selection
exact generation parameters
```

The original Booktracker torrent identifiers/links should be recovered from the repository history and archived README versions where possible.

If those exact torrents no longer exist, torrent metadata, magnet hashes, file lists or archived descriptions could still allow identification of the original corpus.

## Priority order

Please approach this as an archival/recovery investigation in roughly this order:

1. Inspect the complete original `mgrankin/ru_transformers` repository and its history.
2. Test the original anonymous S3 bucket directly.
3. Determine the exact contents/filenames expected inside `m_checkpoint-3365357`.
4. Search the exact checkpoint identifier globally.
5. Enumerate and inspect GitHub forks.
6. Search archives and old notebooks for mirrors or downloaded copies.
7. Recover the exact tokenizer.
8. Recover the original inference/web-service code.
9. Determine the production sampling parameters.
10. Recover and run the original checkpoint unchanged on CPU.
11. Preserve known-good sample outputs.
12. Only then consider conversion to a modern runtime.
13. Only if recovery fails completely, investigate faithful retraining.

## Desired outcome

The ideal result is a locally runnable archival copy containing something like:

```text
porfirevich-original/
├── model/
│   └── m_checkpoint-3365357/
├── tokenizer/
├── config/
├── original-inference-code/
├── generation-settings.md
├── provenance.md
├── SHA256SUMS
└── run.sh
```

The success criterion is **not**:

> "We found a Russian GPT-2 that writes similarly."

The success criterion is:

> **Recover the actual Porfirievich production checkpoint (`m_checkpoint-3365357`), its exact tokenizer and its original generation configuration, and reproduce the original autoregressive Russian text-completion behaviour locally.**

If the exact checkpoint is genuinely lost, document the evidence for that conclusion and then determine how faithfully it can be reconstructed from the surviving code, tokenizer, datasets and training recipe.

Please distinguish throughout the investigation between:

- things directly confirmed by original sources,
- things supported by secondary/archived sources,
- reasonable inferences,
- and things that remain unknown.

Do not silently treat assumptions as facts.
