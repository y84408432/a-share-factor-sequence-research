"""V4 contract tests. No market or private Qlib data needed."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from lstm_research.pilot import anchor_sha256
from lstm_research.hypothesis_v4 import (
    flat_window, load_v3_baseline, compare_daily, verify_label_scaler,
    MLP_VARIANTS, make_mlp,
)


class V4Tests(unittest.TestCase):
    def test_same_anchor_different_window_content_and_order(self):
        data=np.arange(120*16,dtype=np.float32).reshape(120,16)
        anchor=np.array([32,71],dtype=np.int64)
        latest=flat_window(data,anchor,1)
        five=flat_window(data,anchor,5)
        thirty=flat_window(data,anchor,30)
        self.assertEqual(latest.shape,(2,16))
        self.assertEqual(five.shape,(2,5*16))
        self.assertEqual(thirty.shape,(2,30*16))
        np.testing.assert_array_equal(latest[0],data[32])
        np.testing.assert_array_equal(five[0].reshape(5,16)[0],data[28])
        np.testing.assert_array_equal(five[0,-16:],latest[0])
        np.testing.assert_array_equal(thirty[1].reshape(30,16)[0],data[42])
        np.testing.assert_array_equal(thirty[1,-16:],data[71])

    def test_window_edge_and_finite_contract(self):
        data=np.ones((40,16),dtype=np.float32)
        with self.assertRaises(ValueError):
            flat_window(data,np.array([3]),5)
        data[27,2]=np.nan
        with self.assertRaises(ValueError):
            flat_window(data,np.array([30]),5)

    def test_paired_daily_rejects_mismatched_time_or_constituents(self):
        dates=pd.to_datetime(['2021-01-04','2021-01-05','2021-01-06'])
        a=pd.DataFrame({'date':dates,'rankic':[.1,.2,.3],'ic':[.1,.2,.3],'n_stocks':[100,100,100]})
        b=a.copy()
        b['rankic']=[.05,.15,.2]
        c=compare_daily(a,b)
        self.assertAlmostEqual(c['mean'],(.05+.05+.1)/3)
        b.loc[1,'n_stocks']=99
        with self.assertRaises(ValueError):
            compare_daily(a,b)
        b=a.copy()
        b.loc[0,'date']=pd.Timestamp('2021-01-07')
        with self.assertRaises(ValueError):
            compare_daily(a,b)

    def test_reused_v3_baseline_checks_anchor(self):
        idx=pd.DataFrame({'panel_row':[30,31,32]})
        digest=anchor_sha256(idx)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            m={'valid_anchor_sha256':digest,'n_samples':3,'seed':20261115,'window':30}
            (path/'ridge_valid_metrics.json').write_text(json.dumps(m))
            pd.DataFrame({'date':['2021-01-04'],'n_stocks':[100], 'rankic':[.02], 'ic':[.03]}).to_csv(path/'ridge_daily_rankic.csv',index=False)
            got, daily=load_v3_baseline(path,'ridge',digest,3)
            self.assertEqual(got['n_samples'],3)
            self.assertEqual(len(daily),1)
            with self.assertRaises(ValueError):
                load_v3_baseline(path,'ridge','different',3)

    def test_label_scaler_comparison_and_model_sizes(self):
        a={'fit_split':'train','low':-.1,'high':.1,'mean':0.,'std':.04}
        verify_label_scaler(a,a)
        b=dict(a); b['mean']=1.0
        with self.assertRaises(ValueError):
            verify_label_scaler(a,b)
        self.assertNotEqual(MLP_VARIANTS['mlp_flat_small']['hidden'],
                            MLP_VARIANTS['mlp_flat_baseline']['hidden'])
        self.assertEqual(MLP_VARIANTS['mlp_flat_strong_wd']['hidden'],
                         MLP_VARIANTS['mlp_flat_baseline']['hidden'])
        self.assertGreater(MLP_VARIANTS['mlp_flat_strong_wd']['weight_decay'],
                           MLP_VARIANTS['mlp_flat_baseline']['weight_decay'])
        small=make_mlp('mlp_flat_small')
        large=make_mlp('mlp_flat_baseline')
        self.assertLess(sum(p.numel() for p in small.parameters()),
                        sum(p.numel() for p in large.parameters()))

if __name__=='__main__': unittest.main()
