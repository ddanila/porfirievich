# Credits and licensing

## Original project

Mikhail Grankin created Porfirievich («Порфирьевич») and the Russian GPT-2
training and inference project: https://github.com/mgrankin/ru_transformers.
The original project distributed the recovered checkpoint through models.dobro.ai.
Its complete source is referenced as the unchanged submodule
`porfirevich-original/original-inference-code`, pinned to
`e698092ed31b340dc39ecea26f1b0be87cdbd022`.

The upstream source is distributed under Apache License 2.0. Its license and
individual source notices remain intact. A copy of that license is also
provided in this repository's root LICENSE file.

## Adapted inference code

`porfirevich-original/run_cpu.py` adapts the upstream sampling and tokenizer
behavior for a modern runtime and adds historical service profiles and
iterative candidate selection. The upstream generation code carries these
notices, also retained in the adapted file:

> Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
>
> Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

Credit also belongs to OpenAI for GPT-2, Hugging Face for Transformers,
and the PyTorch and YouTokenToMe projects used by the runner. Dependencies
retain their respective licenses.

## Recovery work and artifacts

The recovery documentation, compatibility changes, scripts, and benchmarks in
this repository were prepared by ddanila in 2026 and are provided under
Apache License 2.0. Original authors retain copyright in their contributions.

The recovered model objects and tokenizer are original third-party artifacts,
not newly trained models. The source-code license is not asserted here as a
separate license grant for model weights, tokenizer artifacts, or training
texts. Their recorded provenance and checksums are in
[provenance.md](porfirevich-original/provenance.md) and
[SHA256SUMS](porfirevich-original/SHA256SUMS). The large weight file is excluded
from Git; the download script retrieves it from its original distribution endpoint.
