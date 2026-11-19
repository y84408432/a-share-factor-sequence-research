"""V12 predeclared final 2025–2026 evaluation helpers.

No fitting, model selection or external data reads. This module deliberately
reuses V9/V10 mathematics, while replacing their 2023/2024-only year labels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .pilot import nw_mean_interval
from .v9_development import evaluate_group
from .v10_economics import MODEL_IDS, PRED_COLS, lag_hac

START = pd.Timestamp('2025-01-01')
END = pd.Timestamp('2026-11-02')
YEARS = (2025, 2026)
PRIMARY_PAIRS = (
    ('A30_LSTM', 'A30_Ridge'),
    ('A30_LSTM', 'B30_LSTM'),
    ('B1_Ridge', 'A30_LSTM'),
)


def check_final_period(index: pd.DataFrame, *, full: bool = True) -> None:
    required = {'split','datetime','label_end_date','panel_row','instrument'}
    if required - set(index.columns) or index.empty:
        raise ValueError('Empty/incomplete final cohort')
    if not (index['split'] == 'final_holdout').all():
        raise ValueError('Final cohort includes data from other periods')
    dates = pd.to_datetime(index['datetime'])
    ends = pd.to_datetime(index['label_end_date'])
    if (dates.isna().any() or ends.isna().any() or dates.min() < START or
        dates.max() > END or (ends <= dates).any() or (ends > END).any()):
        raise ValueError('Final horizon or t+6 label boundary violation')
    if index['panel_row'].duplicated().any() or index[['instrument','datetime']].duplicated().any():
        raise ValueError('Duplicate final sample')
    if full and (set(dates.dt.year.unique()) != set(YEARS) or dates.nunique() < 350 or len(index) < 100000):
        raise ValueError('Insufficient final cohort coverage')


def model_results(name, dates, actual, pred_list, seeds):
    """Date-equal mean of *per-seed* ICs, distinct from ensemble IC."""
    if len(pred_list) != len(seeds):
        raise ValueError('Incorrect seed count')
    metrics, daily = evaluate_group(dates, actual, pred_list)
    scores = np.asarray([x['rankic_daily_equal_weight']['mean'] for x in metrics],float)
    ics = np.asarray([x['ic_daily_equal_weight']['mean'] for x in metrics],float)
    mse = np.asarray([x['mse_raw'] for x in metrics],float)
    hac = nw_mean_interval(daily['rankic'].to_numpy(dtype=float))
    result = {
        'model':name, 'n_seeds':len(seeds), 'n_samples':len(actual), 'n_days':len(daily),
        'rankic_seed_mean':float(scores.mean()),
        'rankic_seed_sd':float(scores.std(ddof=1)) if len(scores)>1 else None,
        'rankic_hac_low':hac['ci95'][0], 'rankic_hac_high':hac['ci95'][1],
        'ic_seed_mean':float(ics.mean()), 'mse_raw_seed_mean':float(mse.mean()),
    }
    yearly=[]
    for year in YEARS:
        group=daily.loc[daily.date.dt.year==year]
        if len(group)<15:
            raise ValueError(f'Insufficient final year {year} evaluation days')
        result[f'rankic_{year}']=float(group.rankic.mean())
        result[f'ic_{year}']=float(group.ic.mean())
        yearly.append({'model':name,'year':f'{year}' + ('_YTD' if year==2026 else ''),
                       'n_days':len(group),'rankic':float(group.rankic.mean()),
                       'ic':float(group.ic.mean())})
    individuals=[{'model':name,'seed':str(seed),
                  'rankic':float(m['rankic_daily_equal_weight']['mean']),
                  'ic':float(m['ic_daily_equal_weight']['mean']),
                  'mse_raw':float(m['mse_raw'])} for seed,m in zip(seeds,metrics)]
    return result, individuals, daily, yearly


def validate_final_predictions(df: pd.DataFrame, *, expected_rows:int, expected_days:int):
    required={'instrument','datetime','label_raw',*PRED_COLS}
    if required-set(df.columns):
        raise ValueError('Final predictions missing required columns')
    out=df.copy()
    out['datetime']=pd.to_datetime(out['datetime'])
    if (out[['instrument','datetime']].isna().any().any() or
        out[['instrument','datetime']].duplicated().any() or
        out.datetime.min()<START or out.datetime.max()>END or
        set(out.datetime.dt.year.unique())!=set(YEARS)):
        raise ValueError('Final stock/date identity or period invalid')
    if len(out)!=expected_rows or out.datetime.nunique()!=expected_days:
        raise ValueError('Final prediction cohort unexpectedly changed')
    if not np.isfinite(out[['label_raw',*PRED_COLS]].to_numpy(float)).all():
        raise ValueError('Nonfinite labels or predictions')
    if (out.groupby('datetime').size()<100).any():
        raise ValueError('Unreasonably small final cross-section')
    return out.sort_values(['datetime','instrument'],kind='mergesort').reset_index(drop=True)


def final_quintile_summaries(daily):
    """Same 5-day/HAC statistics as V10; 2026 is partial-year only."""
    needed={'model','signal_date','Q1','Q2','Q3','Q4','Q5','universe','top_minus_bottom','top_minus_universe'}
    if needed-set(daily.columns):raise ValueError('Missing economic daily columns')
    df=daily.copy();df['signal_date']=pd.to_datetime(df.signal_date)
    if df.signal_date.min()<START or df.signal_date.max()>END:
        raise ValueError('Wrong economic evaluation period')
    rows=[]
    for model,part in df.groupby('model',sort=False):
        if model not in MODEL_IDS:raise ValueError('Unfrozen economic model')
        for period,dd in [('2025-2026_YTD',part),('2025',part[part.signal_date.dt.year==2025]),
                           ('2026_YTD',part[part.signal_date.dt.year==2026])]:
            dd=dd.sort_values('signal_date')
            if len(dd)<15:raise ValueError('Insufficient days for economic period')
            for metric in ('Q1','Q2','Q3','Q4','Q5','universe','top_minus_bottom','top_minus_universe'):
                stats=lag_hac(dd[metric].to_numpy(dtype=float))
                rows.append({'model':model,'year':period,'measure':metric,
                            'units':'5-day adjusted-open return',**stats})
    if len(rows)!=len(MODEL_IDS)*3*8:
        raise ValueError('Economic model or period count changed')
    return pd.DataFrame(rows)


def future_eligibility(panel:pd.DataFrame, final_rows:np.ndarray, a_final_rows:np.ndarray):
    """Quantify label-finiteness condition without accessing label magnitudes."""
    required={'split','strict_window_30','within_split_label','label_finite'}
    if required-set(panel.columns):raise ValueError('Panel missing eligibility flags')
    selected=np.asarray(final_rows,dtype=np.int64)
    common=np.asarray(a_final_rows,dtype=np.int64)
    mask=((panel.split=='final_holdout')&panel.strict_window_30&panel.within_split_label).to_numpy(bool)
    candidate=np.flatnonzero(mask)
    kept=candidate[panel.label_finite.to_numpy(bool)[candidate]]
    if not np.array_equal(np.sort(kept),np.sort(selected)):
        raise ValueError('V2 final B30 index differs from pre-label candidates with observed targets')
    if not np.isin(common,kept).all():
        raise ValueError('A/B common cohort not a subset of label-available B30')
    by_year={}
    years=pd.to_datetime(panel.datetime).dt.year.to_numpy()
    for year in YEARS:
        cp=candidate[years[candidate]==year]
        n_kept=int(np.sum(panel.label_finite.to_numpy(bool)[cp]))
        by_year[str(year)]={'candidate_rows':len(cp), 'missing_future_labels':len(cp)-n_kept,
                           'missing_future_label_fraction':float((len(cp)-n_kept)/len(cp)) if len(cp) else None}
    return {'exante_B30_candidates':len(candidate),'B30_label_available':len(kept),
            'excluded_due_to_future_label_nonfinite':len(candidate)-len(kept),
            'fraction_excluded':float((len(candidate)-len(kept))/len(candidate)),
            'common_A_B_label_eligible':len(common),
            'A_missing_window_exclusions':int(len(kept)-len(common)),
            'by_signal_year':by_year,
            'warning':'Uses V1 existing eligibility flags only, no final return magnitudes. Future target observability remains a selection condition; membership-filtered panel may miss omitted names/bars.'}
