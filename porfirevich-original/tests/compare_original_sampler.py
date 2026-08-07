#!/usr/bin/env python3
"""Verify that the CPU port returns the recovered sampler's exact token IDs."""

from pathlib import Path
import sys

import torch
import transformers


ARCHIVE = Path(__file__).resolve().parents[1]
UPSTREAM = ARCHIVE / "original-inference-code"
MODEL_PATH = ARCHIVE / "model" / "m_checkpoint-3365357"

# The old module builds a CLI help list from attributes removed in Transformers
# 4.x. Supplying empty maps lets us import its unmodified sample_sequence.
for class_name in (
    "GPT2Config",
    "OpenAIGPTConfig",
    "XLNetConfig",
    "TransfoXLConfig",
):
    setattr(
        getattr(transformers, class_name),
        "pretrained_config_archive_map",
        {},
    )

sys.path.insert(0, str(UPSTREAM))
from run_generation import sample_sequence as original_sample_sequence  # noqa: E402

sys.path.insert(0, str(ARCHIVE))
from run_cpu import OriginalYTEncoder, sample_sequence as ported_sample_sequence  # noqa: E402


def main() -> None:
    tokenizer = OriginalYTEncoder(MODEL_PATH / "encoder.model")
    model = transformers.GPT2LMHeadModel.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        use_safetensors=False,
    )
    model.eval()

    context = tokenizer.encode("Однажды вечером")
    newline = tokenizer.encode("\n")[-1:]
    filtered = (
        [1]
        + tokenizer.encode("[")[-1:]
        + tokenizer.encode("(")[-1:]
        + newline
    )

    torch.manual_seed(42)
    expected = original_sample_sequence(
        model=model,
        context=context,
        length=5,
        temperature=1,
        top_k=0,
        top_p=0.9,
        device="cpu",
        filter_single=filtered,
        filter_double=newline,
        num_samples=3,
    )
    torch.manual_seed(42)
    actual = ported_sample_sequence(
        model=model,
        context=context,
        length=5,
        num_samples=3,
        filter_single=filtered,
        filter_double=newline,
    )

    if not torch.equal(expected, actual):
        raise SystemExit("FAIL: sampled token IDs differ")
    print("PASS: sampled token IDs are identical")


if __name__ == "__main__":
    main()
