"""V6 no-data unit tests and small CPU checkpoint smoke test."""
from __future__ import annotations
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from lstm_research.order_v6 import (shuffle_past, shuffle_seed, predict_checkpoint,
                                    average_daily, compare_daily, SEEDS)
from lstm_research.lstm_window_v5 import train_one


class V6Tests(unittest.TestCase):
    def test_shuffle_preserves_latest_day_and_full_day_vectors(self):
        x=np.arange(12*10*16,dtype=np.float32).reshape(12,10,16)
        sh=shuffle_past(x,77,chunk=4)
        np.testing.assert_array_equal(sh[:,-1,:],x[:,-1,:])
        for i in range(len(x)):
            a=sorted(tuple(r) for r in x[i,:-1,:])
            b=sorted(tuple(r) for r in sh[i,:-1,:])
            self.assertEqual(a,b)
            self.assertNotIn(tuple(x[i,-1,:]), b)
        self.assertFalse(np.array_equal(x[:,:-1,:],sh[:,:-1,:]))

    def test_shuffle_reproducible_and_seed_independent(self):
        x=np.arange(40*30*16,dtype=np.float32).reshape(40,30,16)
        np.testing.assert_array_equal(shuffle_past(x,33,chunk=13),shuffle_past(x,33,chunk=13))
        np.testing.assert_array_equal(shuffle_past(x,33,chunk=13),shuffle_past(x,33,chunk=13))
        self.assertFalse(np.array_equal(shuffle_past(x,33),shuffle_past(x,34)))
        self.assertNotEqual(shuffle_seed(SEEDS[0],window=10,experiment='C'),
                            shuffle_seed(SEEDS[0],window=10,experiment='D'))
        self.assertNotEqual(shuffle_seed(SEEDS[0],window=10,experiment='D',split='train'),
                            shuffle_seed(SEEDS[0],window=10,experiment='D',split='valid'))

    def test_shuffle_rejects_bad_input(self):
        with self.assertRaises(ValueError): shuffle_past(np.ones((4,1,16)),1)
        with self.assertRaises(ValueError): shuffle_past(np.ones((4,10,12)),1)
        y=np.ones((4,10,16));y[3,0,0]=np.nan
        with self.assertRaises(ValueError): shuffle_past(y,1)

    def test_average_daily_paired_not_pseudoreplicated(self):
        dates=pd.to_datetime(['2021-01-04','2021-01-05','2021-01-06'])
        a=pd.DataFrame({'date':dates,'n_stocks':[101]*3,'rankic':[.2,.3,.4], 'ic':[.1,.1,.1]})
        b=a.copy();b['rankic']+=.02
        c=average_daily([a,b,b])
        self.assertAlmostEqual(c.rankic.mean(),a.rankic.mean()+.04/3)
        self.assertAlmostEqual(compare_daily(a,a)['mean'],0)
        bad=a.copy();bad.loc[0,'n_stocks']=50
        with self.assertRaises(ValueError): average_daily([a,bad])

    def test_v5_checkpoint_reload_cpu_smoke(self):
        rng=np.random.default_rng(25)
        xtr=rng.normal(size=(32,10,16)).astype('float32')
        xva=rng.normal(size=(16,10,16)).astype('float32')
        ytr=xtr[:,-1,0]*.15
        yva=xva[:,-1,0]*.15
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'v5_checkpoint.pt'
            first,_=train_one(xtr,ytr,xva,yva,window=10,seed=SEEDS[0],
                              device_request='cpu',checkpoint=p,max_epochs=1,
                              batch_size=16,patience=1)
            restored=predict_checkpoint(xva,p,window=10,seed=SEEDS[0],device_request='cpu')
            np.testing.assert_allclose(first,restored,atol=1e-7)
            perturbed=shuffle_past(xva,1024)
            np.testing.assert_array_equal(xva[:,-1,:],perturbed[:,-1,:])
            _=predict_checkpoint(perturbed,p,window=10,seed=SEEDS[0],device_request='cpu')
            with self.assertRaises(ValueError):
                predict_checkpoint(xva,p,window=10,seed=SEEDS[1],device_request='cpu')

if __name__=='__main__':
    unittest.main()
