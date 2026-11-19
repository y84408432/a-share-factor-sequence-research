"""V12 frozen final evaluation: 2025 full / 2026 partial, old API, no model selection."""
import importlib.util
import inspect
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lstm_research import v12_final as v12
from lstm_research import v9_development as v9
from lstm_research.v10_economics import MODEL_IDS,daily_quintiles,run_phase_cycles


class TestV12Final(unittest.TestCase):
    def test_import_real_v12_entrypoint_and_v9_v10_functions(self):
        p=ROOT/'12_final_evaluation.py'
        spec=importlib.util.spec_from_file_location('v12_entry_contract',p)
        mod=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for name in ('audit','final_eval','frozen_final','verify_history','lock_ready'):
            self.assertTrue(callable(getattr(mod,name)))
        self.assertTrue(callable(v9.make_predictions))
        self.assertTrue(callable(v9.check_lstm_checkpoint))
        self.assertEqual(mod.MODEL_IDS,MODEL_IDS)
        self.assertEqual(mod.PRIMARY_PAIRS,v12.PRIMARY_PAIRS)
        self.assertEqual(len(mod.COST_SCENARIOS),4)
        self.assertEqual(v12.YEARS,(2025,2026))

    def test_preflight_audit_cannot_score_targets_or_train(self):
        spec=importlib.util.spec_from_file_location('v12_entry_audit',ROOT/'12_final_evaluation.py')
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        source=inspect.getsource(mod.audit)
        self.assertNotIn("np.load(V2/'labels.npy'",source)
        self.assertNotIn('.fit(',source)
        self.assertNotIn('make_predictions(',source)
        self.assertIn('read_membership_intervals(',source)
        self.assertIn('model_registry',inspect.getsource(mod.verify_history))

    @staticmethod
    def frame():
        date=pd.bdate_range('2025-10-20',periods=100)
        return pd.DataFrame({'split':['final_holdout']*100,
                             'datetime':date,'label_end_date':date+pd.Timedelta(days=9),
                             'panel_row':np.arange(100),
                             'instrument':['SH600000']*100})

    def test_final_period_allows_2026_partial(self):
        v12.check_final_period(self.frame(),full=False)

    def test_final_period_blocks_2024_leakage(self):
        f=self.frame();f.loc[0,'datetime']=pd.Timestamp('2024-12-31')
        with self.assertRaises(ValueError):v12.check_final_period(f,full=False)

    def test_final_period_blocks_label_t_plus_six_after_end(self):
        f=self.frame();f.loc[0,'label_end_date']=pd.Timestamp('2026-11-03')
        with self.assertRaises(ValueError):v12.check_final_period(f,full=False)

    def test_final_period_blocks_wrong_split(self):
        f=self.frame();f.loc[0,'split']='development'
        with self.assertRaises(ValueError):v12.check_final_period(f,full=False)

    def test_final_period_blocks_duplicate_anchor(self):
        f=self.frame();f.loc[1,'panel_row']=0
        with self.assertRaises(ValueError):v12.check_final_period(f,full=False)

    def test_future_label_audit_exact_cohort_match(self):
        panel=pd.DataFrame({'split':['final_holdout']*5,
            'strict_window_30':[False,True,True,True,True],
            'within_split_label':[True]*5,
            'label_finite':[False,True,True,False,True],
            'datetime':pd.to_datetime(['2025-01-02']*5)})
        outcome=v12.future_eligibility(panel,np.array([1,2,4]),np.array([1,4]))
        self.assertEqual(outcome['exante_B30_candidates'],4)
        self.assertEqual(outcome['excluded_due_to_future_label_nonfinite'],1)
        self.assertEqual(outcome['A_missing_window_exclusions'],1)
        with self.assertRaises(ValueError):v12.future_eligibility(panel,np.array([1,2]),np.array([1]))

    def test_final_predictions_require_exact_quintile_columns_and_years(self):
        dates=pd.bdate_range('2025-11-03',periods=90)
        rows=[]
        for d in dates:
            for j in range(120):
                row={'instrument':f'SH{600000+j}', 'datetime':d,'label_raw':j*.0001}
                row.update({m+'_mean_raw_pred':j*.0002 for m in MODEL_IDS})
                rows.append(row)
        frame=pd.DataFrame(rows)
        valid=v12.validate_final_predictions(frame,expected_rows=len(frame),expected_days=len(dates))
        self.assertEqual(len(valid),len(frame))
        bad=frame.copy();bad.loc[0,'datetime']=pd.Timestamp('2024-12-31')
        with self.assertRaises(ValueError):v12.validate_final_predictions(bad,expected_rows=len(bad),expected_days=len(dates))

    def test_actual_daily_model_scoring_with_2026_year_label(self):
        dates=pd.bdate_range('2025-11-03',periods=90)
        dates=np.repeat(dates.to_numpy(),120)
        actual=np.tile(np.linspace(-.02,.02,120),90)
        report,individual,daily,years=v12.model_results(
            'A30_LSTM',dates,actual,[actual,actual,actual],[20261115,20261116,20261117])
        self.assertGreater(report['rankic_seed_mean'],.999)
        self.assertEqual(len(individual),3)
        self.assertEqual(len(daily),90)
        self.assertEqual({x['year'] for x in years},{'2025','2026_YTD'})

    def test_economic_final_uses_2026_ytd_not_development_years(self):
        dates=pd.bdate_range('2025-11-03',periods=90)
        rows=[]
        for d in dates:
            for j in range(120):
                row={'instrument':f'SH{600000+j}', 'datetime':d,'label_raw':j*.00001}
                for m in MODEL_IDS:row[m+'_mean_raw_pred']=j*.001
                rows.append(row)
        pred=v12.validate_final_predictions(pd.DataFrame(rows),
                           expected_rows=len(rows),expected_days=90)
        daily,baskets=daily_quintiles(pred)
        economic=v12.final_quintile_summaries(daily)
        self.assertEqual(len(economic),4*3*8)
        self.assertEqual(set(economic.year),{'2025','2026_YTD','2025-2026_YTD'})
        cal=pd.bdate_range(dates[0],periods=97)
        phased,cycles=run_phase_cycles(daily,baskets,cal)
        self.assertEqual(len(phased),4*3*5*4)
        self.assertEqual(len(cycles),4*3*90*4)
        self.assertTrue((economic.loc[economic.measure=='top_minus_bottom','mean']>0).all())

    def test_economic_invalid_prediction_is_rejected(self):
        dt=pd.to_datetime(['2025-01-02','2026-01-02'])
        frame=pd.DataFrame({'instrument':['SH600000','SH600001'],
                            'datetime':dt,'label_raw':[0.1,np.nan]})
        for name in MODEL_IDS:frame[name+'_mean_raw_pred']=.01
        with self.assertRaises(ValueError):v12.validate_final_predictions(frame,expected_rows=2,expected_days=2)

    def test_frozen_comparisons_cannot_be_selected_from_final(self):
        self.assertEqual(v12.PRIMARY_PAIRS,(
            ('A30_LSTM','A30_Ridge'),('A30_LSTM','B30_LSTM'),('B1_Ridge','A30_LSTM')))


if __name__=='__main__':unittest.main()
