"""Check the actual sampling probabilities and CLI parameter boundaries."""

import argparse
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_cpu as runner


class SamplingControlsTest(unittest.TestCase):
    def test_temperature_then_nucleus(self):
        # Temperature 2 maps logits [6, 4, 0] to [3, 2, 0]. At top-p
        # 0.9, the first two survive; renormalization gives softmax([3, 2]).
        logits = torch.tensor([6.0, 4.0, 0.0])
        expected = torch.tensor([0.7310586, 0.2689414, 0.0])

        class Model:
            def __call__(self, input_ids):
                return (logits.expand(input_ids.shape[0], input_ids.shape[1], -1).clone(),)

        def check_draw(probabilities, num_samples):
            torch.testing.assert_close(probabilities, expected)
            return torch.tensor([0])

        for sample in (runner.sample_sequence, runner.sample_sequence_optimized):
            with self.subTest(sampler=sample.__name__), \
                 patch.object(runner, 'CachedNextLogits') as cached, \
                 patch.object(torch, 'multinomial', side_effect=check_draw) as draw:
                cached.return_value.side_effect = lambda ids: logits.expand(ids.shape[0], -1).clone()
                sample(Model(), length=1, context=[1], num_samples=2,
                       filter_single=[], filter_double=[], temperature=2.0, top_p=0.9)
                self.assertEqual(draw.call_count, 2)

    def test_top_p_endpoints_and_hard_masks(self):
        for top_p, expected in ((1.0, [6.0, 4.0, 0.0]), (0.01, [6.0, -float('inf'), -float('inf')])):
            actual = runner.filter_logits_batched(
                torch.tensor([[6.0, 4.0, 0.0, 99.0]]), torch.tensor([0]), [3], [], top_p=top_p,
            )
            torch.testing.assert_close(actual, torch.tensor([expected + [-float('inf')]]))

    def test_defaults_and_invalid_values(self):
        parser = argparse.ArgumentParser()
        runner.add_sampling_args(parser)
        args = parser.parse_args([])
        self.assertEqual((args.temperature, args.top_p), (1.0, 0.9))
        for field, values in (('temperature', [0, -1, float('nan'), float('inf')]),
                              ('top_p', [0, -1, 1.1, float('nan'), float('inf')])):
            for value in values:
                args = argparse.Namespace(temperature=1.0, top_p=0.9)
                setattr(args, field, value)
                with self.subTest(field=field, value=value), self.assertRaises(SystemExit):
                    runner.validate_sampling_args(args)


if __name__ == '__main__':
    unittest.main()
