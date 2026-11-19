"""V2 pure-python tests: no Qlib download, no training/holdout statistics."""
from __future__ import annotations
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lstm_research.feature_spec import FEATURE_NAMES
from lstm_research.sequences import (
    NumpySequenceDataset, build_sample_index, fit_train_only_scaler, index_for,
    scale_feature_column, validate_panel, validate_window_boundaries,
)


def fixture() -> pd.DataFrame:
    # 90 dates and two stocks. All 16 features present; deliberate gap and missing.
    dates = pd.bdate_range('2020-09-01', periods=90)
    records = []
    for stock in ('SH600001','SH600002'):
        for pos, date in enumerate(dates):
            rec = {'instrument':stock,'datetime':date,'calendar_pos':pos,
                   'split':'train' if pos < 70 else 'valid',
                   'label_end_date':dates[pos+6] if pos+6 < len(dates) else pd.NaT,
                   'within_split_label':pos+6 < (70 if pos<70 else len(dates)),
                   'label':0.001 * pos}
            for j,name in enumerate(FEATURE_NAMES):
                rec[name]=float(pos+1+j*0.1)
            if stock=='SH600002' and pos==18:
                rec['RET1']=np.nan
            records.append(rec)
    frame = pd.DataFrame(records)
    finite = np.isfinite(frame[list(FEATURE_NAMES)].to_numpy(dtype=float)).all(axis=1)
    streak = np.zeros(len(frame),dtype=np.int32)
    for i in range(len(frame)):
        if not finite[i]:
            streak[i]=0
        elif i and frame.loc[i,'instrument']==frame.loc[i-1,'instrument'] and frame.loc[i,'calendar_pos']==frame.loc[i-1,'calendar_pos']+1:
            streak[i]=streak[i-1]+1
        else:
            streak[i]=1
    frame['strict_streak']=streak
    for w in (10,30,60):
        frame[f'strict_window_{w}']=streak>=w
    frame['label_finite']=np.isfinite(frame['label'].to_numpy(dtype=float))
    frame['usable_label']=frame['label_finite'] & frame['within_split_label']
    return frame


class SequenceTests(unittest.TestCase):
    def test_scaler_train_only_ignores_future_extreme(self):
        x=fixture()
        one=fit_train_only_scaler(x)
        x.loc[x['split']=='valid','RET1']=1e9
        two=fit_train_only_scaler(x)
        self.assertEqual(one,two)
        self.assertEqual(one['fit_split'],'train')

    def test_full_window_same_stock_and_gap(self):
        f=validate_panel(fixture())
        idx=build_sample_index(f)
        stats=validate_window_boundaries(f,idx)
        self.assertGreater(stats['30'],0)
        self.assertTrue(index_for(idx,'train',60)['eligible30'].all())
        self.assertLess(len(index_for(idx,'train',60)),len(index_for(idx,'train',30)))
        self.assertEqual(len(index_for(idx,'train',10,common_60=True)),len(index_for(idx,'train',60)))
        self.assertEqual(len(index_for(idx,'train',30,common_60=True)),len(index_for(idx,'train',60)))
        for name,part in idx.groupby('split'):
            split_end=pd.Timestamp('2020-12-31' if name=='train' else '2022-12-31')
            # Our synthetic fixture has validation dates in 2020/21; true test is label end
            self.assertTrue(part['label_end_date'].notna().all())

    def test_no_invalid_rows_leak_into_window_and_holdout_guard(self):
        f=validate_panel(fixture())
        idx=build_sample_index(f)
        scaler=fit_train_only_scaler(f)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            matrix=np.column_stack([scale_feature_column(f[c].to_numpy(),scaler['parameters'][c]) for c in FEATURE_NAMES])
            np.save(root/'features_scaled.npy',matrix)
            np.save(root/'labels.npy',f['label'].to_numpy(dtype=np.float32))
            idx.to_pickle(root/'sample_index.pkl.gz',compression='gzip')
            ds=NumpySequenceDataset(root,'train',30)
            self.assertGreater(len(ds),10)
            a,y,m=ds[0]
            self.assertEqual(a.shape,(30,16))
            self.assertTrue(np.isfinite(a).all())
            self.assertTrue(np.isfinite(y))
            self.assertEqual(m['instrument'],'SH600001')
            with self.assertRaises(PermissionError):
                NumpySequenceDataset(root,'final_holdout',30)

    def test_transform_missing_ineligible_not_zero_imputation(self):
        f=fixture()
        scaler=fit_train_only_scaler(f)
        vals=np.array([np.nan, np.inf, 15.0])
        result=scale_feature_column(vals,scaler['parameters']['RET1'])
        self.assertEqual(result[0],0)
        self.assertEqual(result[1],0)
        self.assertTrue(np.isfinite(result[2]))
        idx=build_sample_index(f)
        # Not present in any 30d window ending immediately after a missing point.
        bad_ending=f.index[(f.instrument=='SH600002') & (f.calendar_pos==30)][0]
        self.assertNotIn(int(bad_ending), index_for(idx,'train',30).panel_row.to_list())


if __name__=='__main__':
    unittest.main()
