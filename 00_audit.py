#!/usr/bin/env python3
"""Read-only, small Qlib audit for the A-share LSTM project.

Reads existing daily Qlib data; does not install, train, or use held-out predictions.
Saves only a small, dated data audit to output/audit.
"""
from __future__ import annotations

import json
import math
import platform
import sys
from pathlib import Path

DATA_DIR = Path.home() / '.qlib' / 'qlib_data' / 'cn_data'
OUT = Path(__file__).resolve().parent / 'output' / 'audit'
SAMPLE_DAY = '2024-06-14'
START, END = '2024-06-03', '2024-07-19'


def label_from_opens(opens: list[float], anchor: int, horizon: int = 5) -> float:
    """At t close, buy at open(t+1), sell at open(t+1+horizon)."""
    if anchor < 0 or anchor + 1 + horizon >= len(opens):
        raise IndexError('not enough future opens')
    start = opens[anchor + 1]
    end = opens[anchor + 1 + horizon]
    if not math.isfinite(start) or not math.isfinite(end) or start <= 0:
        return float('nan')
    return end / start - 1.0


def audit():
    import numpy as np
    import pandas as pd
    import qlib
    from qlib.constant import REG_CN
    from qlib.data import D

    if not (DATA_DIR / 'calendars/day.txt').exists():
        raise SystemExit(f'MISSING Qlib daily data at {DATA_DIR}')
    if not (DATA_DIR / 'instruments/csi300.txt').exists():
        raise SystemExit(f'MISSING historical CSI300 member file at {DATA_DIR}')

    OUT.mkdir(parents=True, exist_ok=True)
    qlib.init(provider_uri=str(DATA_DIR), region=REG_CN)
    calendar = pd.DatetimeIndex(D.calendar(start_time='2015-01-01', end_time='2026-12-31', freq='day'))
    print(f'[1/5] Qlib initialized: {DATA_DIR}', flush=True)
    print(f'  Daily calendar: {len(calendar)} days; first={calendar.min().date()}, last={calendar.max().date()}', flush=True)

    universe = D.instruments(market='csi300')
    snapshots = {}
    for date in ('2018-06-15', SAMPLE_DAY, '2026-06-15'):
        tickers = sorted(set(D.list_instruments(universe, start_time=date, end_time=date, as_list=True)))
        snapshots[date] = {'count': len(tickers), 'head': tickers[:5]}
        print(f'[2/5] Historical CSI300 at {date}: {len(tickers)} stocks; e.g. {tickers[:3]}', flush=True)

    sample_stocks = sorted(set(D.list_instruments(universe, start_time=SAMPLE_DAY, end_time=SAMPLE_DAY, as_list=True)))[:8]
    if not sample_stocks:
        raise RuntimeError('No historical CSI300 constituents retrieved')
    # All expressions use information available by the anchor-day close, except LABEL.
    expressions = ['$open', '$high', '$low', '$close', '$volume',
                   'Ref($close,1)', 'Ref($volume,1)',
                   'Ref($open,-6)/Ref($open,-1)-1']
    field_names = ['open', 'high', 'low', 'close', 'volume', 'close_lag1', 'volume_lag1', 'label_open1_open6']
    print('[3/5] Reading a small historical, mixed-stock panel...', flush=True)
    frame = D.features(sample_stocks, expressions, start_time=START, end_time=END, freq='day')
    frame.columns = field_names
    if frame.empty:
        raise RuntimeError('Qlib returned empty features on 2024 sample dates')
    frame = frame.reset_index()
    frame['datetime'] = pd.to_datetime(frame['datetime'])
    print(f'  Rows={len(frame)}; stocks={frame.instrument.nunique()}; days={frame.datetime.nunique()}', flush=True)

    print('[4/5] Field completeness and future-open label checks:', flush=True)
    null_fractions = frame[field_names].isna().mean().round(6).to_dict()
    for col in field_names:
        vals = pd.to_numeric(frame[col], errors='coerce')
        finite_count = int(np.isfinite(vals.to_numpy(dtype=float)).sum())
        print(f'  {col:<20} finite={finite_count:<5} missing={null_fractions[col]:.2%}', flush=True)
    label = frame['label_open1_open6']
    valid_y = int(np.isfinite(label.to_numpy(dtype=float)).sum())
    if valid_y < 15:
        raise RuntimeError('Too few finite future open/open labels; inspect dataset validity')

    # Compare label expression to directly observed future opens for a sample stock with
    # complete daily observations; only if 7 subsequent rows have consecutive market days.
    one = frame[frame.instrument == sample_stocks[0]].sort_values('datetime').reset_index(drop=True)
    aligned = 0
    discrepancies = []
    calendar_pos = {d: j for j, d in enumerate(calendar)}
    for j in range(max(0, len(one) - 6)):
        dates = one.loc[j:j + 6, 'datetime'].to_list()
        if any(date not in calendar_pos for date in dates):
            continue
        if any(calendar_pos[dates[k + 1]] - calendar_pos[dates[k]] != 1 for k in range(6)):
            continue
        v = label_from_opens(one['open'].tolist(), j, horizon=5)
        got = one.loc[j, 'label_open1_open6']
        if math.isfinite(v) and math.isfinite(got):
            aligned += 1
            discrepancies.append(abs(v - float(got)))
    max_diff = max(discrepancies) if discrepancies else None
    print(f'  Expression/manual label comparison: {aligned} eligible dates; max_abs_difference={max_diff}', flush=True)
    if aligned and max_diff is not None and max_diff > 2e-5:
        print('  WARNING: direct observed opens and Qlib expression differ; inspect before modelling.', flush=True)

    optional = {}
    for expr, name in [('$factor', 'factor'), ('$vwap', 'vwap')]:
        try:
            piece = D.features(sample_stocks[:2], [expr], start_time=START, end_time=END, freq='day')
            finite = int(np.isfinite(piece.to_numpy(dtype=float)).sum())
            optional[name] = {'available': True, 'finite_values': finite}
        except Exception as exc:
            optional[name] = {'available': False, 'error': f'{type(exc).__name__}: {exc}'[:300]}
        print(f'[5/5] Optional {name}: {optional[name]}', flush=True)

    outdata = {
        'protocol_status': 'DRAFT; not a completed research evaluation',
        'machine': {'python': sys.version.split()[0], 'machine': platform.machine(), 'qlib': getattr(qlib, '__version__', 'unknown')},
        'data_dir': str(DATA_DIR),
        'calendar': {'count': len(calendar), 'first': str(calendar.min().date()), 'last': str(calendar.max().date())},
        'historic_csi300': snapshots, 'sample_period': [START, END], 'sample_stocks': sample_stocks,
        'sample_rows': len(frame), 'sample_n_dates': frame.datetime.nunique(),
        'missing_fraction': null_fractions, 'valid_label_rows': valid_y,
        'label_expression': expressions[-1], 'manual_label_compared_days': aligned,
        'max_abs_label_discrepancy': max_diff, 'optional_fields': optional,
        'important_caveats': [
            'Historical membership existence is not a guarantee that the entire data source is point-in-time or survivorship-bias-free.',
            'Price adjustment and halted/limit-up/limit-down tradability require separate auditing before economic backtests.',
            'No final test-period model performance is read by this program.'
        ]
    }
    (OUT / 'audit.json').write_text(json.dumps(outdata, ensure_ascii=False, indent=2), encoding='utf-8')
    frame.head(25).to_csv(OUT / 'sample_rows.csv', index=False)
    print(f'\nAUDIT_COMPLETE: {OUT}\nShare audit.json and terminal log. No model training performed.', flush=True)


if __name__ == '__main__':
    audit()
