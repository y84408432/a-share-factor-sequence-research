import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lstm_research.v10_economics import (
    MODEL_IDS,PRED_COLS,assign_quintile,validate_predictions,daily_quintiles,
    lag_hac,share_turnover,assert_calendar_contiguous,run_phase_cycles,
    quote_execution_audit,
)


class V10EconomicTests(unittest.TestCase):
    @staticmethod
    def sample(n_dates=25,n_stocks=20):
        dates=pd.bdate_range('2023-01-02',periods=n_dates)
        rows=[]
        for t,dt in enumerate(dates):
            for s in range(n_stocks):
                entry={'datetime':dt,'instrument':f'SH{s:06d}',
                       'label_raw':0.001*(s-10)+0.0001*t}
                for j,k in enumerate(MODEL_IDS):
                    entry[k+'_mean_raw_pred']=(s*(1 if j!=1 else -1))+t*.00001
                rows.append(entry)
        return pd.DataFrame(rows),pd.bdate_range('2023-01-02',periods=n_dates+8)

    def test_quintiles_ties_deterministic(self):
        n=20
        s=np.zeros(n)
        ids=np.array([f'SH{i:03d}' for i in range(n)][::-1])
        q1=assign_quintile(s,ids)
        q2=assign_quintile(s,ids)
        self.assertTrue(np.array_equal(q1,q2))
        self.assertEqual([int((q1==j).sum()) for j in range(1,6)],[4]*5)

    def test_bad_quintile_nonfinite(self):
        with self.assertRaises(ValueError):assign_quintile([0,1,2,3,np.nan],[str(i) for i in range(5)])

    def test_duplicate_and_nonfinite_prediction(self):
        df,_=self.sample()
        x=pd.concat([df,df.iloc[:1]],ignore_index=True)
        with self.assertRaises(ValueError):validate_predictions(x)
        bad=df.copy();bad.loc[0,PRED_COLS[0]]=np.nan
        with self.assertRaises(ValueError):validate_predictions(bad)

    def test_prediction_preserves_quintile_monotonicity(self):
        df,_=self.sample()
        valid=validate_predictions(df)
        daily,baskets=daily_quintiles(valid)
        g=daily[daily.model=='B1_Ridge']
        self.assertTrue((g['Q5']>g['Q1']).all())
        self.assertEqual(len(baskets),25*4)
        self.assertTrue((g.Q5_n==4).all())

    def test_hac_constant_no_error(self):
        stats=lag_hac(np.full(60,.03))
        self.assertAlmostEqual(stats['mean'],.03)
        self.assertAlmostEqual(stats['ci95_lo'],.03,places=10)

    def test_drift_aware_turnover(self):
        first=share_turnover(None,{'A':0.2,'B':-0.1})
        self.assertAlmostEqual(first[0],1.)
        self.assertAlmostEqual(first[1],0.)
        b,s=share_turnover({'A':.20,'B':-.10},{'A':.0,'B':.0})
        self.assertAlmostEqual(b,s)
        self.assertAlmostEqual(b,abs(.5-1.2/2.1),places=12)
        self.assertEqual(share_turnover({'A':0,'B':0},{'A':0,'B':0}),(0.,0.))

    def test_calendar_gap_rejected(self):
        cal=pd.bdate_range('2023-01-02',periods=25)
        with self.assertRaises(ValueError):assert_calendar_contiguous(cal[[0,1,3]],cal)

    def test_five_phase_net_cost_sensitive(self):
        df,cal=self.sample()
        daily,baskets=daily_quintiles(validate_predictions(df))
        phase,cycles=run_phase_cycles(daily,baskets,cal)
        self.assertEqual(len(phase),4*3*5*4)
        self.assertEqual(len(cycles),4*3*25*4)
        sub=phase[(phase.model=='B1_Ridge')&(phase.group=='Q5')&(phase.phase==0)]
        gross=float(sub[sub.scenario=='gross_0bp'].iloc[0].cumulative_return)
        net=float(sub[sub.scenario=='sym_20bp'].iloc[0].cumulative_return)
        self.assertLess(net,gross)
        r=cycles[(cycles.model=='B1_Ridge')&(cycles.group=='Q5')&(cycles.phase==0)&(cycles.scenario=='gross_0bp')]
        self.assertTrue((r.exit_date>r.entry_date).all())

    def test_quote_label_match_and_future_leakage_guard(self):
        cal=pd.bdate_range('2023-01-02',periods=25)
        instruments=['SH600000','SH600001']
        quote=[]
        for i,d in enumerate(cal):
            for j,s in enumerate(instruments):
                px=10.0 + i*.02 + j
                quote.append({'instrument':s,'datetime':d,'open_adj':px,'close_adj':px+.005,
                              'volume_adj':1000,'factor':1.2})
        quote=pd.DataFrame(quote)
        data=[]
        for i in range(12):
            for j,s in enumerate(instruments):
                row={'instrument':s,'datetime':cal[i],
                     'label_raw':(10+(i+6)*.02+j)/(10+(i+1)*.02+j)-1}
                row.update({k:0.001 for k in PRED_COLS})
                data.append(row)
        frame=pd.DataFrame(data)
        result=quote_execution_audit(frame,quote,cal)
        self.assertEqual(result['label_quote_mismatch_over_tolerance'],0)
        self.assertEqual(result['n_signals'],24)
        bad=frame.copy();bad.loc[0,'label_raw']+=.01
        with self.assertRaises(ValueError):quote_execution_audit(bad,quote,cal)

    def test_entrypoint_imports_old_module_api_without_invoking_data(self):
        f=ROOT/'10_economic_diagnostics.py'
        self.assertTrue(f.is_file())
        spec=importlib.util.spec_from_file_location('v10_entry',f)
        mod=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertTrue(callable(mod.frozen_v9))
        self.assertTrue(callable(mod.audit))
        self.assertTrue(callable(mod.evaluate))


if __name__=='__main__':unittest.main()
