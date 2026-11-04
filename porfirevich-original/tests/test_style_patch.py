"""Ensure style patches are explicit, validated, and never alter checkpoint files."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style_patch import apply_style_patch


class StylePatchTest(unittest.TestCase):
    def test_in_memory_edit_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'pytorch_model.bin'
            checkpoint.write_bytes(b'original checkpoint sentinel')
            original_file = checkpoint.read_bytes()
            patch = {'format': 'porfirievich-mlp-bias-v1', 'layer': 0,
                     'base_checkpoint_sha256': hashlib.sha256(original_file).hexdigest(),
                     'delta': [0.1, -0.2, 0.3]}
            bias = torch.nn.Parameter(torch.zeros(3))
            model = SimpleNamespace(transformer=SimpleNamespace(h=[
                SimpleNamespace(mlp=SimpleNamespace(c_proj=SimpleNamespace(bias=bias)))
            ]))
            path = root / 'patch.json'
            path.write_text(json.dumps(patch))
            apply_style_patch(model, root, path)
            torch.testing.assert_close(bias, torch.tensor(patch['delta']))
            self.assertEqual(checkpoint.read_bytes(), original_file)
            for field, bad in [('layer', -1), ('delta', [1]),
                               ('delta', [float('nan'), 0, 0]),
                               ('base_checkpoint_sha256', 'wrong'), ('format', 'unknown')]:
                before = bias.detach().clone()
                path.write_text(json.dumps({**patch, field: bad}))
                with self.subTest(field=field), self.assertRaises(ValueError):
                    apply_style_patch(model, root, path)
                torch.testing.assert_close(bias, before, rtol=0, atol=0)
                self.assertEqual(checkpoint.read_bytes(), original_file)


if __name__ == '__main__':
    unittest.main()
