import importlib.util
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
from lstm_research import v9_development as v9
from lstm_research.representation_v7 import A_NAMES

class TestFrozenV9(unittest.TestCase):
    def test_v9_runner_imports_real_v7_dependencies(self):
        """Catches V8's previous missing-attribute mistake."""
        spec=importlib.util.spec_from_file_location('v9_runner_test',BASE/'09_development_evaluation.py')
        mod=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for attr in ('audit','frozen_inputs','eval_once','preconditions','valid_reference'):
            self.assertTrue(callable(getattr(mod,attr)))
        self.assertEqual(v9.MODEL_IDS,('B1_Ridge','A30_Ridge','A30_LSTM','B30_LSTM'))

    def test_a_train_only_scaler_reuse(self):
        a=np.asarray([[-2,0,1,2,3,4,5,6],[2,0,1,2,3,4,5,6]],np.float32)
        scaler={'features':list(A_NAMES),'fit_split':'train',
                'parameters':{name:{'clip_low':-1,'clip_high':1,'median':0,'iqr_or_fallback':2.0}
                              for name in A_NAMES}}
        out=v9.raw_a_to_scaled(a,scaler)
        self.assertAlmostEqual(out[0,0],-.5)
        self.assertAlmostEqual(out[1,0],.5)
        self.assertTrue(np.isfinite(out).all())
        scaler['fit_split']='development'
        with self.assertRaises(ValueError):v9.raw_a_to_scaled(a,scaler)

    def test_dev_label_purging(self):
        idx=pd.DataFrame({'split':['development']*3,
             'datetime':pd.to_datetime(['2023-01-03','2024-06-03','2024-12-20']),
             'label_end_date':pd.to_datetime(['2023-01-12','2024-06-12','2024-12-30'])})
        with self.assertRaises(ValueError):v9.check_period(idx)
        big=pd.concat([idx]*400,ignore_index=True)
        self.assertRaises(ValueError,v9.check_period,big) # days insufficient
        big['label_end_date']=pd.Timestamp('2025-01-04')
        self.assertRaises(ValueError,v9.check_period,big)

    def test_strict_A_window(self):
        good=np.ones(100,dtype=bool)
        anchors=pd.DataFrame({'panel_row':[31,45,70]})
        good[10]=False
        valid=v9.cohort_by_a(anchors,good)
        self.assertListEqual(valid.panel_row.tolist(),[45,70])

    def test_same_cohort_pred(self):
        from sklearn.linear_model import Ridge
        arr=np.arange(300,dtype=np.float32).reshape(100,3)
        rows=np.asarray([10,40,75])
        model=Ridge(alpha=1).fit(arr[[9,39,74]],np.asarray([1.,3.,5.]))
        p=v9.make_predictions(model,arr,rows,kind='ridge',window=1,batch=2)
        np.testing.assert_allclose(p,model.predict(arr[rows]),atol=1e-5)
        with self.assertRaises(ValueError):v9.make_predictions(model,arr,rows,kind='ridge',window=2)

    def test_frozen_lstm_architecture_prediction(self):
        import torch
        from lstm_research.representation_v7 import model_for
        model=model_for(8)
        feature=np.zeros((64,8),dtype=np.float32)
        p=v9.make_predictions(model,feature,np.asarray([29,35,60]),kind='lstm',window=30,device='cpu',batch=2)
        self.assertEqual(p.shape,(3,))
        self.assertTrue(np.isfinite(p).all())
        model2=model_for(16)
        f2=np.zeros((64,16),dtype=np.float32)
        p2=v9.make_predictions(model2,f2,np.asarray([29,35]),kind='lstm',window=30,device='cpu',batch=2)
        self.assertEqual(p2.shape,(2,))

    def test_file_registry_requires_genuine_model_paths(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):v9.model_registry(Path(td))

    def test_lstm_checkpoint_metadata(self):
        v9.check_lstm_checkpoint({'channels':8,'window':30,'seed':20261115,'state_dict':{}},8,20261115)
        with self.assertRaises(ValueError):v9.check_lstm_checkpoint({'channels':16,'window':30,'seed':20261115,'state_dict':{}},8,20261115)

    def test_daily_comparisons_require_same_dates(self):
        dates=pd.date_range('2023-01-01',periods=35)
        a=pd.DataFrame({'date':dates,'n_stocks':np.full(35,150),'rankic':np.full(35,.03)})
        b=a.copy();b['rankic']=.01
        diff=v9.compare_daily(a,b)
        self.assertAlmostEqual(diff['mean'],.02)
        b.loc[0,'n_stocks']=145
        with self.assertRaises(ValueError):v9.compare_daily(a,b)

if __name__=='__main__':unittest.main()
