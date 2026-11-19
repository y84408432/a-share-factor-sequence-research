"""Small synthetic contract tests. They do not use Qlib, dates or held-out labels."""
import unittest
import numpy as np
import pandas as pd
from lstm_research.pilot import (
    make_batch,stack_sequences,fit_train_label_scaler,target_for_train,
    target_for_valid,pred_to_raw,daily_cs_metrics,paired_rankic,anchor_sha256
)

class PilotContracts(unittest.TestCase):
    def test_temporal_order_old_to_new_and_flat(self):
        raw=np.arange(240,dtype=np.float32).reshape(80,3)
        rows=np.array([30,70])
        x=make_batch(raw,rows,window=30)
        self.assertEqual(x.shape,(2,30,3))
        np.testing.assert_array_equal(x[0,0,:],raw[1])
        np.testing.assert_array_equal(x[0,-1,:],raw[30])
        np.testing.assert_array_equal(stack_sequences(raw,rows,True)[0].reshape(30,3),x[0])
    def test_label_fit_is_train_only(self):
        tr=np.linspace(-.12,.15,1000)
        va=np.array([1000.,-1000.,.01])
        cfg=fit_train_label_scaler(tr)
        self.assertLess(cfg['high'],.16)
        self.assertGreater(abs(target_for_valid(va,cfg)[0]),1000)
        np.testing.assert_allclose(pred_to_raw(target_for_train(tr,cfg),cfg),
                                   np.clip(tr,cfg['low'],cfg['high']),atol=1e-7)
    def test_daily_cs_equal_weight_and_paired(self):
        n=40
        dates=np.repeat([np.datetime64('2021-01-04'),np.datetime64('2021-01-05')],n)
        y=np.tile(np.arange(n,dtype=float),2)
        pred=np.hstack((np.arange(n,dtype=float), -np.arange(n,dtype=float)))
        df=daily_cs_metrics(dates,y,pred)
        self.assertEqual(len(df),2)
        self.assertAlmostEqual(df.rankic.mean(),0.,places=12)
        diff=paired_rankic(df,df)
        self.assertAlmostEqual(diff['mean'],0.,places=12)
    def test_anchor_identity_changes_when_indices_change(self):
        a=pd.DataFrame({'panel_row':[1,2,3]})
        b=pd.DataFrame({'panel_row':[1,3,2]})
        self.assertNotEqual(anchor_sha256(a),anchor_sha256(b))

if __name__=='__main__': unittest.main()
