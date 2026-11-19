"""V11 real-interface regression + pure deterministic membership/eligibility/risk tests."""
from __future__ import annotations
import importlib.util
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from lstm_research.v11_validity import (
  read_membership_intervals,membership_info,daily_pool_range,
  eligibility_profile,align_trade_fields,risk_flags,risk_counts,q5_risk_by_model,
)
from lstm_research.v10_economics import MODEL_IDS

class V11ValidityTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'csi300.txt'
        self.path.write_text('SH600000\t2022-12-30\t2023-01-04\n'
                             'SH600000\t2023-01-04\t2023-01-06\n'
                             'SZ000001\t2023-01-03\t2023-01-05\n'
                             'SZ000001\t2023-01-10\t2023-01-11\n')
        self.intervals,self.meta=read_membership_intervals(self.path)

    def test_interval_coalesce_and_inclusive_dates(self):
        self.assertEqual(self.meta['raw_interval_count'],4)
        self.assertEqual(self.meta['overlap_records'],1)
        self.assertEqual(len(self.intervals),3)
        self.assertEqual(self.intervals.iloc[0]['end'],pd.Timestamp('2023-01-06'))

    def test_membership_and_age_reattachment(self):
        cal=pd.date_range('2022-12-30','2023-01-11',freq='B')
        records=pd.DataFrame({'instrument':['SH600000']*3+['SZ000001']*2,
          'datetime':pd.to_datetime(['2023-01-03','2023-01-06','2023-01-09','2023-01-06','2023-01-10'])})
        hits,age,starts=membership_info(records,self.intervals,cal)
        np.testing.assert_array_equal(hits,[True,True,False,False,True])
        self.assertEqual(age[0],2)
        self.assertEqual(age[4],0)
        self.assertEqual(age[2],-1)

    def test_count_active_dates(self):
        days=pd.to_datetime(['2023-01-03','2023-01-06','2023-01-10'])
        c=daily_pool_range(self.intervals,days)
        self.assertEqual(c.n_active.tolist(),[2,1,1])

    def test_bad_interval_rejected(self):
        self.path.write_text('SH600000 2023-01-03 2023-01-02\n')
        with self.assertRaises(ValueError):read_membership_intervals(self.path)

    def test_exante_label_exclusion_count_and_key_match(self):
        d=['2023-01-03','2023-01-04','2024-02-01']
        panel=pd.DataFrame({'instrument':['SH600000']*3,
          'datetime':pd.to_datetime(d),'split':['development']*3,
          'within_split_label':[True]*3,'strict_window_30':[True]*3,
          'label_finite':[True,False,True]})
        selected=panel.iloc[[0,2]][['instrument','datetime']]
        stats,exante,lost=eligibility_profile(panel,selected)
        self.assertEqual(stats['exante_b30_candidates'],3)
        self.assertEqual(stats['excluded_due_to_future_label_nonfinite'],1)
        self.assertEqual(len(lost),1)
        with self.assertRaises(ValueError):eligibility_profile(panel,panel[['instrument','datetime']])

    def test_execution_bar_alignment_and_label_identity(self):
        dates=pd.date_range('2023-01-02',periods=8,freq='B')
        quotes=pd.DataFrame({'instrument':['SH600000']*8,'datetime':dates,
          'open_adj':np.arange(100.,108.),'close_adj':np.arange(100.,108.),
          'volume_adj':[100.]*8,'factor':[1.]*8})
        p=pd.DataFrame({'instrument':['SH600000'],'datetime':[dates[0]],
                        'label_raw':[106./101.-1]})
        details=align_trade_fields(p,quotes,dates)
        self.assertAlmostEqual(details.entry_gap.iloc[0],.01)
        self.assertAlmostEqual(details.exit_gap.iloc[0],106/105-1)
        self.assertTrue(details.entry_volume_finite_positive.iloc[0])
        p['label_raw']=0.4
        with self.assertRaises(ValueError):align_trade_fields(p,quotes,dates)

    def test_execution_proxies_do_not_claim_limit_flags(self):
        df=pd.DataFrame({'entry_gap':[0.101,-0.101,0.2,np.nan],
           'exit_gap':[-.1,0,0,.12],
           'entry_open_finite_positive':[True]*4,
           'entry_volume_finite_positive':[True,True,False,True],
           'exit_open_finite_positive':[True]*4,
           'exit_volume_finite_positive':[True]*4})
        flags=risk_flags(df)
        a=risk_counts(flags,np.ones(4,dtype=bool))
        self.assertEqual(a['entry_gap_up_9_8pct']['count'],2)
        self.assertEqual(a['entry_gap_down_9_8pct']['count'],1)
        self.assertEqual(a['entry_quote_invalid']['count'],1)

    def test_q5_uses_all_frozen_model_names_not_outcome(self):
        p=pd.DataFrame({'instrument':[f'SH600{i:03d}' for i in range(10)],
                        'datetime':[pd.Timestamp('2023-01-03')]*10,
                         **{m+'_mean_raw_pred':np.arange(10.) for m in MODEL_IDS}})
        flags={'risk':np.arange(10)>=8}
        result=q5_risk_by_model(p,flags)
        self.assertEqual(set(result),set(MODEL_IDS))
        for m in MODEL_IDS:
            self.assertEqual(result[m]['n'],2)
            self.assertEqual(result[m]['risk']['count'],2)

    def test_entrypoint_imports_actual_v9_v10_modules(self):
        root=Path(__file__).resolve().parent.parent
        spec=importlib.util.spec_from_file_location('v11_script',root/'11_data_validity_audit.py')
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        from lstm_research.v10_economics import MODEL_IDS as frozen_ids
        self.assertEqual(tuple(module.MODEL_IDS),tuple(frozen_ids))
        self.assertEqual(module.V10.name,'v10_economics')
        self.assertTrue(callable(module.verify_frozen))

if __name__=='__main__':unittest.main()

class V11EndToEndSmokeTest(unittest.TestCase):
    def test_synthetic_full_audit_no_qlib_needed(self):
        """Execute the real V11 entrypoint against frozen-shape synthetic local files."""
        import sys
        from unittest.mock import patch
        from lstm_research.v9_development import file_hash
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'output/v1_data').mkdir(parents=True)
            (root/'output/v10_economics').mkdir(parents=True)
            (root/'market/instruments').mkdir(parents=True)
            (root/'market/calendars').mkdir(parents=True)
            v1=root/'output/v1_data';v10=root/'output/v10_economics'
            market=root/'market'
            names=[f'SH600{i:03d}' for i in range(10)]
            (market/'instruments/csi300.txt').write_text(''.join(f'{s}\t2010-01-04\t2024-12-31\n' for s in names))
            all_cal=pd.date_range('2010-01-04','2024-12-31',freq='B')
            (market/'calendars/day.txt').write_text(''.join(f'{d:%Y-%m-%d}\n' for d in all_cal))
            cal=pd.date_range('2023-01-02','2024-12-31',freq='B')
            signal_days=pd.to_datetime(['2023-01-03','2024-01-03'])
            q=pd.DataFrame([(code,date,100+0.01*i,100+0.01*i,1000.,1.)
                            for code in names for i,date in enumerate(cal)],
                           columns=['instrument','datetime','open_adj','close_adj','volume_adj','factor'])
            def label_for(date):
                k=cal.get_loc(date)
                return (100+0.01*(k+6))/(100+0.01*(k+1))-1
            pred=pd.DataFrame([{'instrument':code,'datetime':date,'label_raw':label_for(date),
                                **{m+'_mean_raw_pred':float(i) for m in MODEL_IDS}}
                               for date in signal_days for i,code in enumerate(names)])
            pred.to_csv(v10/'predict.csv',index=False)
            panel=pred[['instrument','datetime']].copy()
            panel['split']='development';panel['within_split_label']=True
            panel['strict_window_30']=True;panel['label_finite']=True
            panel.to_pickle(v1/'frozen_factor_panel.pkl.gz',compression='gzip')
            quote=v10/'quotes_2023_2024.pkl.gz';q.to_pickle(quote,compression='gzip')
            calendar_file=v10/'calendar_2023_2024.npy';np.save(calendar_file,cal.to_numpy())
            spec=importlib.util.spec_from_file_location('v11_smoke',
                Path(__file__).resolve().parent.parent/'11_data_validity_audit.py')
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            out=root/'output/v11_validity'
            outputs=(out/'v11_membership_audit.json',out/'v11_eligibility_audit.json',
                     out/'v11_execution_risk_audit.json',out/'v11_verdict.json',
                     out/'v11_source_manifest.json')
            def fake_source():
                return (pred,q,cal,{'2023_2024_anchor_sha256':'test-anchor'},
                        {'protocol_sha256':'test-protocol'})
            with patch.object(module,'SRC',market),patch.object(module,'V1',v1),\
                 patch.object(module,'V10',v10),patch.object(module,'OUT',out),\
                 patch.object(module,'PRED',v10/'predict.csv'),\
                 patch.object(module,'QUOTE',quote),\
                 patch.object(module,'CALENDAR',calendar_file),\
                 patch.object(module,'V11_OUTPUTS',outputs),\
                 patch.object(module,'verify_frozen',side_effect=fake_source),\
                 patch.object(module,'provider_snapshots',return_value=[{'date':'2023-01-03','symmetric_difference':0}]),\
                 patch.object(sys,'argv',['11_data_validity_audit.py','--stage','audit']):
                module.main()
                self.assertTrue(all(f.is_file() for f in outputs))
                import json
                v=json.loads(outputs[3].read_text())
                self.assertEqual(v['status'],'MECHANICAL_CHECKS_PASSED_WITH_UNVERIFIED_ASSUMPTIONS')
                self.assertFalse(v['independent_historical_CSI300_PIT_certified'])
                module.main()   # Already-audited guard does not overwrite.
