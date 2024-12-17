import unittest
import torch
import torchmetrics

from utils import Metrics

class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.metrics = Metrics()

    def test_update_and_compute(self):
        logits = torch.tensor([[[2.0, 0.5, 0.3], [0.1, 0.2, 0.7]]])
        labels = torch.tensor([[0, 2]])

        self.metrics.update(logits, labels)

        computed_metrics = self.metrics.compute()

        self.assertIn("perplexity", computed_metrics)
        perplexity = computed_metrics["perplexity"]
        self.assertIsInstance(perplexity, torch.Tensor)

    def test_reset(self):
        logits = torch.tensor([[[2.0, 0.5, 0.3], [0.1, 0.2, 0.7]]])
        labels = torch.tensor([[0, 2]])

        self.metrics.update(logits, labels)

        self.metrics.reset()

        computed_metrics = self.metrics.compute()

        self.assertEqual(computed_metrics["perplexity"].item(), float("inf"))


if __name__ == "__main__":
    unittest.main()