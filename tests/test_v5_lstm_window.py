"""Pure local tests: no Qlib provider, financial data or secrets required."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from lstm_research.lstm_window_v5 import (
    WINDOWS, SEEDS, window_batch, materialize_cohort, new_lstm_model,
    train_one, check_paired_daily, across_seeds_daily, summarize_window,
)
from lstm_research.pilot import evaluate_validation


class V5Tests(unittest.TestCase):
    def test_sequence_time_order_and_common_anchor(self):
        x = np.arange(90 * 16, dtype=np.float32).reshape(90, 16)
        anchors = np.array([33, 64], dtype=np.int64)
        for w in WINDOWS:
            window = window_batch(x, anchors, w)
            self.assertEqual(window.shape, (2, w, 16))
            np.testing.assert_array_equal(window[:, -1, :], x[anchors])
            np.testing.assert_array_equal(window[:, 0, :], x[anchors - w + 1])
            np.testing.assert_array_equal(materialize_cohort(x, anchors, w, chunk=1), window)

    def test_window_rejects_invalid_input(self):
        x = np.ones((50, 16), dtype=np.float32)
        with self.assertRaises(ValueError):
            window_batch(x, np.array([5]), 30)
        with self.assertRaises(ValueError):
            window_batch(x, np.array([100]), 1)
        x[29, 3] = np.nan
        with self.assertRaises(ValueError):
            window_batch(x, np.array([31]), 5)
        with self.assertRaises(ValueError):
            window_batch(x, np.array([32]), 20)

    def test_lstm_parameter_count_independent_of_window(self):
        import torch
        model = new_lstm_model().eval()
        with torch.no_grad():
            for w in WINDOWS:
                self.assertEqual(tuple(model(torch.randn(4, w, 16)).shape), (4,))
        self.assertGreater(sum(p.numel() for p in model.parameters()), 0)

    def test_paired_requires_equal_days_and_stock_counts(self):
        d = pd.DataFrame({'date':pd.to_datetime(['2021-01-04','2021-01-05','2021-01-06']),
                          'n_stocks':[100, 101, 102], 'rankic':[.1, .2, .3], 'ic':[.11, .22, .33]})
        self.assertAlmostEqual(check_paired_daily(d, d)['mean'], 0)
        wrong = d.copy()
        wrong.loc[0, 'n_stocks'] += 1
        with self.assertRaises(ValueError): check_paired_daily(d, wrong)
        wrong = d.copy()
        wrong.loc[1, 'date'] = pd.Timestamp('2021-01-14')
        with self.assertRaises(ValueError): check_paired_daily(d, wrong)

    def test_three_seed_daily_mean_uses_dates_not_pseudoreplication(self):
        days = pd.to_datetime(['2021-01-04','2021-01-05','2021-01-06'])
        data = [pd.DataFrame({'date':days, 'n_stocks':[80]*3, 'rankic':[.1+i*.02]*3,
                              'ic':[.01+i*.01]*3}) for i in range(3)]
        result = across_seeds_daily(data)
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(float(result.rankic.iloc[0]), .12)
        with self.assertRaises(ValueError): across_seeds_daily(data[:2])

    def test_train_one_cpu_synthetic_smoke(self):
        # Small end-to-end test; production settings remain frozen at 12 epochs.
        rng = np.random.default_rng(17)
        train = rng.normal(size=(48, 5, 16)).astype(np.float32)
        valid = rng.normal(size=(24, 5, 16)).astype(np.float32)
        ytrain = (.1*train[:, -1, 0]).astype(np.float32)
        yvalid = (.1*valid[:, -1, 0]).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'test.pt'
            pred, detail = train_one(train, ytrain, valid, yvalid, window=5,
                                     seed=SEEDS[0], device_request='cpu', checkpoint=path,
                                     max_epochs=2, patience=2, batch_size=16)
            self.assertEqual(pred.shape, (24,))
            self.assertTrue(np.isfinite(pred).all())
            self.assertTrue(path.is_file())
            self.assertEqual(detail['epochs_run'], 2)
            self.assertTrue(1 <= detail['best_epoch_by_valid_mse'] <= 2)

if __name__ == '__main__':
    unittest.main()
