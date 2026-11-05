"""Check prefix matching and hook cleanup without loading checkpoint weights."""
import unittest
import torch
from types import SimpleNamespace
from experiment import NameBias, amplify


class BiasTests(unittest.TestCase):
    def test_longest_prefix_and_alternative_endings(self):
        bias = NameBias([[2, 3, 4], [2, 3, 5]], 3)
        history = torch.tensor([[0, 1], [0, 2], [2, 3], [3, 4]])
        actual = bias(torch.zeros(4, 8), history)
        expected = torch.zeros(4, 8)
        expected[0, 2] = expected[1, 3] = 3
        expected[2, 4:6] = 3
        expected[3, 2] = 3
        torch.testing.assert_close(actual, expected)

    def test_first_token_control_ignores_prefix(self):
        bias = NameBias([[2, 3, 4]], 3, first_only=True)
        actual = bias(torch.zeros(1, 8), torch.tensor([[2, 3]]))
        expected = torch.zeros(1, 8)
        expected[0, 2] = 3
        torch.testing.assert_close(actual, expected)

    def test_activation_hook_removed_even_on_error(self):
        mlp = torch.nn.Linear(2, 2)
        model = SimpleNamespace(transformer=SimpleNamespace(h=[SimpleNamespace(mlp=mlp)]))
        x = torch.ones(1, 2)
        original = mlp(x).detach().clone()
        with self.assertRaises(RuntimeError):
            with amplify(model, {'kind': 'mlp', 'layer': 0, 'factor': 1.5}):
                torch.testing.assert_close(mlp(x), original * 1.5)
                raise RuntimeError('cleanup check')
        torch.testing.assert_close(mlp(x), original)


if __name__ == '__main__':
    unittest.main()
