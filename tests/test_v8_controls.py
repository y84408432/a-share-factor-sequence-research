"""V8 contract tests; do not open production data or late-period labels."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from lstm_research.lstm_window_v5 import SEEDS
from lstm_research.v8_controls import (
    WINDOW,FEATURES,REPEATS,perm_seed,permute_past,flatten_a30,
    fit_ridge,flat_mlp,fit_flat_mlp,same_day_metrics,mean_daily_diff,
)


def fake_x(n):
    # Each day is unique, channels move TOGETHER.
    i=np.arange(n,dtype=np.float32)[:,None,None]*10000
    t=np.arange(30,dtype=np.float32)[None,:,None]*100
    f=np.arange(8,dtype=np.float32)[None,None,:]
    return np.ascontiguousarray(i+t+f,dtype=np.float32)


class TestV8Controls(unittest.TestCase):
    def test_8_channels_30_days_240_flat_oldest_to_latest(self):
        x=fake_x(3)
        z=flatten_a30(x)
        self.assertEqual(z.shape,(3,240))
        np.testing.assert_array_equal(z[0,:8],x[0,0])
        np.testing.assert_array_equal(z[2,-8:],x[2,-1])
        with self.assertRaises(ValueError):flatten_a30(np.ones((2,30,16),np.float32))

    def test_permutation_changes_past_but_never_last_or_channels(self):
        x=fake_x(11);copy=x.copy()
        y=permute_past(x,123)
        np.testing.assert_array_equal(x,copy)
        np.testing.assert_array_equal(y[:,-1],x[:,-1])
        for row in range(len(x)):
            # No inter-row mixing: all input daily first-channel values preserved.
            np.testing.assert_array_equal(np.sort(y[row,:-1,0]),np.sort(x[row,:-1,0]))
            np.testing.assert_array_equal(y[row,:,7]-y[row,:,0],np.full(30,7))
        self.assertFalse(np.array_equal(y[:,:-1],x[:,:-1]))

    def test_shuffle_is_chunk_invariant_and_rng_independent(self):
        x=fake_x(31)
        a=permute_past(x,432,chunk=4)
        b=permute_past(x,432,chunk=16)
        np.testing.assert_array_equal(a,b)
        self.assertFalse(np.array_equal(a,permute_past(x,433)))
        tr=perm_seed(SEEDS[0],0,'train')
        va=perm_seed(SEEDS[0],0,'valid')
        self.assertNotEqual(tr,va)
        self.assertNotEqual(tr,perm_seed(SEEDS[0],1,'train'))
        with self.assertRaises(ValueError):perm_seed(SEEDS[0],3,'train')

    def test_same_sample_daily_contrasts_and_no_pseudoreplication(self):
        dates=pd.date_range('2021-01-04',periods=40,freq='B')
        base=pd.DataFrame({'date':dates,'n_stocks':np.repeat(300,40),
            'rankic':np.linspace(.01,.03,40),'ic':np.linspace(.03,.04,40)})
        a=base.copy();a['rankic']+=.015
        b=same_day_metrics([base,a,a])
        self.assertEqual(len(b),40)
        self.assertAlmostEqual(b.rankic.mean()-base.rankic.mean(),.01)
        paired=mean_daily_diff(a,base)
        self.assertEqual(paired['n_days'],40)
        self.assertAlmostEqual(paired['mean'],.015)
        a.loc[0,'n_stocks']=120
        with self.assertRaises(ValueError):mean_daily_diff(a,base)

    def test_ridge_240_input_and_mlp_240_input(self):
        rng=np.random.default_rng(18)
        x=rng.normal(size=(150,240)).astype(np.float32)
        y=(0.04*x[:,0]+0.1*x[:,239]+rng.normal(size=150)*.05).astype(np.float32)
        p,reg=fit_ridge(x[:100],y[:100],x[100:])
        self.assertEqual(p.shape,(50,))
        self.assertEqual(reg.coef_.shape,(240,))
        self.assertTrue(np.isfinite(p).all())
        self.assertEqual(flat_mlp()[0].in_features,240)

    def test_flat_mlp_cpu_checkpoint_smoke(self):
        rng=np.random.default_rng(99)
        xtr=rng.normal(size=(48,240)).astype(np.float32)
        xva=rng.normal(size=(39,240)).astype(np.float32)
        yy=(xtr[:,0]+xtr[:,239]).astype(np.float32)*.02
        yv=(xva[:,0]+xva[:,239]).astype(np.float32)*.02
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'checkpoint.pt'
            pred,fit=fit_flat_mlp(xtr,yy,xva,yv,seed=SEEDS[0],device='cpu',
                                   checkpoint=path,max_epochs=1,patience=1,batch_size=16)
            self.assertEqual(len(pred),len(xva))
            self.assertTrue(np.isfinite(pred).all())
            self.assertTrue(path.exists())
            self.assertEqual(fit['best_epoch'],1)

    def test_cache_protocol_mismatch_is_rejected(self):
        import importlib.util
        loc=Path(__file__).resolve().parents[1]/'08_test_a30_temporal_value.py'
        spec=importlib.util.spec_from_file_location('v8_script',loc)
        mod=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as td:
            with patch.object(mod,'OUT',Path(td)):
                name='ridge_A30_flat';m,d,c=mod.model_paths(name)
                self.assertFalse(mod.check_cache(name,{'protocol_sha256':'hash'}))
                m.write_text(json.dumps({'name':name,'protocol_sha256':'different'}))
                d.write_text('date,n_stocks,rankic,ic\n')
                c.parent.mkdir(exist_ok=True);c.write_text('dummy')
                with self.assertRaises(ValueError):
                    mod.check_cache(name,{'protocol_sha256':'hash'})
                m.unlink()
                with self.assertRaises(RuntimeError):
                    mod.check_cache(name,{'protocol_sha256':'hash'})


if __name__=='__main__':unittest.main()
