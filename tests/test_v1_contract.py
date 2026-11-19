"""Data contract tests. Tests do NOT need Qlib data or train models."""
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lstm_research.feature_spec import (
    FACTORS, FEATURE_EXPRESSIONS, FEATURE_NAMES, LABEL_EXPR,
    SPLITS, assert_contract,
)
from lstm_research.quality import assign_splits_and_label_end, with_history_streak


class FrozenFactorTests(unittest.TestCase):
    def test_exact_16_frozen_and_label(self):
        assert_contract()
        self.assertEqual(len(FEATURE_NAMES), 16)
        self.assertEqual(len(set(FEATURE_NAMES)), 16)
        self.assertNotIn('CORR20', FEATURE_NAMES)
        self.assertIn('VWAP0', FEATURE_NAMES)
        self.assertEqual(LABEL_EXPR, 'Ref($open,-6)/Ref($open,-1)-1')
        for f in FACTORS:
            self.assertFalse(',-' in f.expr or ', -' in f.expr)

    def test_formula_fingerprint_stable(self):
        self.assertEqual(FEATURE_EXPRESSIONS[0], '$close/Ref($close,1)-1')
        self.assertEqual(FEATURE_EXPRESSIONS[5], 'Std($close/Ref($close,1)-1, 20)')
        self.assertEqual(FEATURE_EXPRESSIONS[-1], '$vwap/$close')

    def test_purge_t_plus_6_at_boundary(self):
        # Synthetic calendar ending in a few trading days of 2021.
        cal = pd.bdate_range('2020-12-15', '2021-01-20')
        frame = pd.DataFrame({
            'instrument': ['SH000001'] * len(cal), 'datetime': cal,
        })
        x = assign_splits_and_label_end(frame, cal)
        dec = x[x.datetime.dt.year == 2020]
        self.assertTrue((dec.iloc[-6:]['within_split_label'] == False).all())
        self.assertEqual(dec.iloc[-7]['label_end_date'], cal[cal.get_loc(dec.iloc[-7]['datetime']) + 6])
        self.assertTrue(bool(dec.iloc[-7]['within_split_label']))
        self.assertTrue((x[x.datetime.dt.year == 2021]['split'] == 'valid').all())

    def test_contiguous_history_and_nonfinite(self):
        cal = pd.bdate_range('2020-06-01', periods=12)
        dt = cal.delete(5)  # gap must reset streak
        frame = pd.DataFrame({'instrument': ['SH000001'] * len(dt),
                              'datetime': dt, 'calendar_pos': cal.get_indexer(dt),
                              'split': ['train'] * len(dt),
                              'within_split_label': [True] * len(dt),
                              'label': [0.01] * len(dt)})
        for f in FEATURE_NAMES:
            frame[f] = 1.0
        frame.loc[2, 'RET1'] = np.inf
        got = with_history_streak(frame, windows=(3, 5))
        self.assertEqual(got.loc[2, 'strict_streak'], 0)
        self.assertEqual(got.loc[5, 'strict_streak'], 1)
        self.assertFalse(bool(got.loc[5, 'strict_window_3']))
        self.assertTrue(bool(got.loc[8, 'strict_window_3']))


if __name__ == '__main__':
    unittest.main()
