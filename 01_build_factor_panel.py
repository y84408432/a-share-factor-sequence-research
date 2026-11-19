#!/usr/bin/env python3
"""V1 full-market frozen-factor extraction and non-predictive quality assessment.

Run from a_share_lstm_research with the project's .venv.
No model fits; no held-out predictions or outcome statistics are produced.
This builds a *membership-filtered* panel; 60-day strict eligibility is only
an audit and may omit pre-membership histories (see V1_SETUP.md).
"""
from __future__ import annotations
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from lstm_research.feature_spec import (
    FACTORS, FEATURE_EXPRESSIONS, FEATURE_NAMES, HORIZON,
    LABEL_EXPR, RAW_AUDIT, SPLITS, assert_contract,
)
from lstm_research.quality import (
    assign_splits_and_label_end, summarize_feature_missingness,
    summarize_sample_eligibility, train_only_feature_percentiles, with_history_streak,
)

DATA_DIR = Path.home() / '.qlib' / 'qlib_data' / 'cn_data'
OUT = ROOT / 'output' / 'v1_data'
CAL_START = '2015-01-05'
END = '2026-11-02'


def main() -> None:
    import qlib
    from qlib.constant import REG_CN
    from qlib.data import D

    assert_contract()
    if not (DATA_DIR / 'instruments/csi300.txt').is_file():
        raise FileNotFoundError(f'Missing Qlib CSI300 universe file: {DATA_DIR}')
    OUT.mkdir(parents=True, exist_ok=True)
    qlib.init(provider_uri=str(DATA_DIR), region=REG_CN)
    calendar = pd.DatetimeIndex(D.calendar(start_time=CAL_START, end_time=END, freq='day'))
    if len(calendar) < 2500:
        raise ValueError('Trading calendar unexpectedly short')
    print('[1/6] Historical CSI300 dynamic-universe factor extraction...', flush=True)
    fields = list(FEATURE_EXPRESSIONS) + [LABEL_EXPR] + [expr for expr, _ in RAW_AUDIT]
    names = list(FEATURE_NAMES) + ['label'] + [name for _, name in RAW_AUDIT]
    # Dynamic membership at anchor day; Qlib automatically retrieves needed lookbacks.
    panel = D.features(D.instruments(market='csi300'), fields,
                       start_time=CAL_START, end_time=END, freq='day', disk_cache=0)
    if len(panel.columns) != len(fields):
        raise RuntimeError('Unexpected returned field count from Qlib')
    panel.columns = names
    panel = panel.reset_index()
    needed_idx = {'instrument', 'datetime'}
    if not needed_idx.issubset(panel.columns):
        raise RuntimeError(f'Missing expected index fields: {set(panel.columns)}')
    panel['datetime'] = pd.to_datetime(panel['datetime'])
    panel['instrument'] = panel['instrument'].astype(str)
    panel = panel.sort_values(['instrument', 'datetime']).reset_index(drop=True)
    if not panel[['instrument', 'datetime']].duplicated().sum() == 0:
        raise RuntimeError('Duplicate instrument/date pairs in Qlib feature result')
    if panel.empty:
        raise RuntimeError('Qlib factor panel empty')
    print(f'  raw rows={len(panel):,}; stocks={panel.instrument.nunique()}; dates={panel.datetime.nunique()}', flush=True)

    print('[2/6] Explicit future label end dates and purging...', flush=True)
    panel = assign_splits_and_label_end(panel, calendar)
    print('  train/valid/development/final_holdout periods assigned.', flush=True)

    print('[3/6] Finite-factor and strict chronological-window diagnostics...', flush=True)
    panel = with_history_streak(panel)
    missing = summarize_feature_missingness(panel)
    eligible = summarize_sample_eligibility(panel)
    train_percentiles = train_only_feature_percentiles(panel)
    for row in eligible.to_dict('records'):
        print(f"  {row['split']:<14} rows={row['rows']:<9} strict60+label={row['strict_60_usable_label_rows']}", flush=True)

    print('[4/6] Market-data integrity checks (no outcome performance)...', flush=True)
    raw = {}
    for field in ('open_adj', 'high_adj', 'low_adj', 'close_adj', 'volume_adj', 'adjustment_factor'):
        vals = pd.to_numeric(panel[field], errors='coerce').to_numpy(dtype='float64')
        finite = np.isfinite(vals)
        raw[field] = {'nonfinite_count': int((~finite).sum()),
                      'nonpositive_count': int((finite & (vals <= 0)).sum())}
    high = panel['high_adj'].to_numpy(dtype=float)
    low = panel['low_adj'].to_numpy(dtype=float)
    raw['high_below_low_count'] = int(np.sum(np.isfinite(high) & np.isfinite(low) & (high < low)))
    raw['zero_volume_count'] = int(np.sum(panel['volume_adj'].to_numpy(dtype=float) == 0))
    feature_missing = missing.pivot(index='feature', columns='split', values='nonfinite_rate')
    feature_missing.to_csv(OUT / 'missingness_by_split.csv', encoding='utf-8-sig')
    eligible.to_csv(OUT / 'sample_eligibility.csv', index=False, encoding='utf-8-sig')
    train_percentiles.to_csv(OUT / 'train_only_percentiles.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame([{'name': f.name, 'group': f.group, 'chinese': f.zh,
                   'qlib_expression': f.expr, 'source': f.source} for f in FACTORS]).to_csv(
                       OUT / 'frozen_factors_16.csv', index=False, encoding='utf-8-sig')

    print('[5/6] Writing panel for future *controlled* modeling...', flush=True)
    # Persist fields without global scaling, clipping, missing-value filling, or future exposure.
    # data stored in two pieces; avoid accidental loss of labels at split boundaries.
    slim = panel[['instrument', 'datetime', 'split', 'calendar_pos', 'label_end_date',
                  'within_split_label', 'all16_finite', 'strict_streak',
                  'strict_window_10', 'strict_window_30', 'strict_window_60',
                  'label_finite', 'usable_label', *FEATURE_NAMES, 'label']].copy()
    for f in FEATURE_NAMES:
        slim[f] = pd.to_numeric(slim[f], errors='coerce').astype('float32')
    slim['label'] = pd.to_numeric(slim['label'], errors='coerce').astype('float32')
    path = OUT / 'frozen_factor_panel.pkl.gz'
    slim.to_pickle(path, compression='gzip')  # no optional pyarrow dependency required

    print('[6/6] Writing non-predictive audit and reproducibility manifest...', flush=True)
    factor_contract = json.dumps([vars(x) for x in FACTORS], sort_keys=True, ensure_ascii=False)
    manifest = {
        'status': 'V1 factor-data and quality audit only; NOT a trained model / tradable strategy',
        'qlib_version': getattr(qlib, '__version__', 'unknown'),
        'python': sys.version.split()[0], 'machine': platform.machine(),
        'dataset': str(DATA_DIR), 'start': CAL_START, 'end': END,
        'universe': 'Historical daily CSI300 constituency per Qlib provider; pre-membership history may be omitted',
        'features': list(FEATURE_NAMES), 'feature_count': len(FEATURE_NAMES),
        'feature_sha256': hashlib.sha256(factor_contract.encode('utf-8')).hexdigest(),
        'label_formula': LABEL_EXPR,
        'label_end_calendar_shift': HORIZON + 1,
        'split_spec': [{'name': n, 'start': s, 'end': e} for n, s, e in SPLITS],
        'panel_rows': int(len(panel)), 'distinct_stocks': int(panel.instrument.nunique()),
        'distinct_dates': int(panel.datetime.nunique()),
        'raw_quality': raw,
        'sample_eligibility': eligible.to_dict('records'),
        'important_caveats': [
            '2015 is warm-up; 2016-2020 train, 2021-22 validation, 2023-24 development, 2025-26 final evaluation period.',
            'Purging uses label_end_date at t+6, not merely the anchor-day split.',
            'Strict 10/30/60 eligible counts exclude pre-membership history and all rows with any nonfinite factor; these are diagnostics, not final sample selection.',
            'Feature preprocessing, clipping, and imputation are not applied. Do not compute global train+test statistics.',
            '2025-2026 was partly observed during older projects; it is not a pristine market holdout.',
            'Adjusted OHLC/volume and factor provenance, PIT constituency, halts/price limits, realistic execution and fee assumptions still need research review.',
            'Financial return labels are only written to the data panel; no held-out predictive statistics or model results were calculated.'
        ],
    }
    (OUT / 'v1_data_audit.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    if eligible.loc[eligible['split'] == 'train', 'strict_60_usable_label_rows'].sum() <= 0:
        raise RuntimeError('No strict-60 eligible train samples; inspect missingness and constituent history')
    print(f'\nV1_DATA_AUDIT_COMPLETE: {OUT}', flush=True)
    print('Please share: v1_data_audit.json, sample_eligibility.csv, missingness_by_split.csv, and build_run.log.', flush=True)
    print('No model trained. No final-holdout predictions evaluated.', flush=True)


if __name__ == '__main__':
    main()
