"""V7 protocol tests: no market downloads, no later-period labels, small CPU smoke."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from lstm_research.lstm_window_v5 import SEEDS
from lstm_research.representation_v7 import (
    A_FEATURES,A_NAMES,check_a_spec,consecutive_window_mask,
    source_consistency,materialize,model_for,fit_and_predict,
    average_daily,mean_rankic_contrast,difference_in_differences,
)


class TestV7(unittest.TestCase):
    def test_exact_8_daily_features_no_long_window_lookahead(self):
        check_a_spec()
        self.assertEqual(len(A_FEATURES),8)
        self.assertEqual(len(set(A_NAMES)),8)
        for f in A_FEATURES:
            self.assertTrue(f.name.startswith('A_'))
            self.assertNotIn('Ref($close,-',f.expression)
            self.assertNotIn('Ref($volume,-',f.expression)

    def test_strict_30_a_feature_window(self):
        good=np.ones(80,dtype=bool)
        good[11]=False
        a=np.array([29,40,41,42,70],dtype=np.int64)
        mask=consecutive_window_mask(good,a,30)
        np.testing.assert_array_equal(mask,[False,False,True,True,True])
        with self.assertRaises(ValueError):
            consecutive_window_mask(good,np.array([20]),30)

    def test_inputs_order_dimensions_and_recent_day(self):
        data=np.arange(45*8,dtype=np.float32).reshape(45,8)
        xs=materialize(data,np.array([30,35]),30)
        self.assertEqual(xs.shape,(2,30,8))
        np.testing.assert_array_equal(xs[0,0],data[1])
        np.testing.assert_array_equal(xs[0,-1],data[30])
        one=materialize(data,np.array([30,35]),1)
        np.testing.assert_array_equal(one[:,0,:],data[[30,35]])
        with self.assertRaises(ValueError):materialize(data,np.array([30]),5)

    def test_a_b_overlap_audit_catches_misalignment(self):
        n=120
        rng=np.random.default_rng(15)
        arr=rng.normal(size=(n,8)).astype(np.float32)
        b=pd.DataFrame({'RET1':arr[:,0],'KMID':arr[:,2],
                        'KLEN':arr[:,3],'VWAP0':arr[:,6]+1})
        audit=source_consistency(arr,b,np.arange(n),tuple(b.columns))
        self.assertEqual(audit['A_RET1']['compared_rows'],n)
        b.loc[10,'KMID']+=0.1
        with self.assertRaises(ValueError):source_consistency(arr,b,np.arange(n),tuple(b.columns))

    def test_paired_means_and_interaction(self):
        base=pd.DataFrame({'date':pd.date_range('2021-03-01',periods=40,freq='B'),
                           'n_stocks':[280]*40,'rankic':np.arange(40)*.001,'ic':.01})
        a1=base.copy();a30=base.copy();b1=base.copy();b30=base.copy()
        a30['rankic']+=.015
        b30['rankic']+=.005
        avg=average_daily([a1,a30,a30])
        self.assertAlmostEqual(avg.rankic.mean(),base.rankic.mean()+.01)
        self.assertAlmostEqual(mean_rankic_contrast(a30,a1)['mean'],.015,places=8)
        self.assertAlmostEqual(difference_in_differences(a1,a30,b1,b30)['mean'],.01,places=8)
        b30.loc[0,'n_stocks']=150
        with self.assertRaises(ValueError):difference_in_differences(a1,a30,b1,b30)

    def test_two_widths_same_hidden_head_and_cpu_checkpoint(self):
        a=model_for(8);b=model_for(16)
        self.assertEqual(a.rnn.hidden_size,b.rnn.hidden_size)
        self.assertEqual(a.head[0].out_features,b.head[0].out_features)
        self.assertNotEqual(sum(p.numel() for p in a.parameters()),sum(p.numel() for p in b.parameters()))
        rng=np.random.default_rng(17)
        for nfeatures,window in ((8,1),(8,30),(16,30)):
            xtr=rng.normal(size=(48,window,nfeatures)).astype(np.float32)
            xva=rng.normal(size=(35,window,nfeatures)).astype(np.float32)
            yt=(xtr[:,-1,0]*.03).astype(np.float32)
            yv=(xva[:,-1,0]*.03).astype(np.float32)
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'model.pt'
                pred,info=fit_and_predict(xtr,yt,xva,yv,seed=SEEDS[0],device='cpu',
                                          checkpoint=path,max_epochs=1,patience=1,batch_size=16)
                self.assertTrue(path.exists())
                self.assertTrue(np.isfinite(pred).all())
                self.assertEqual(len(pred),len(xva))
                self.assertEqual(info['best_epoch'],1)


if __name__=='__main__':unittest.main()
