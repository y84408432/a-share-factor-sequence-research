"""Pure pandas/numpy quality checks; executable without Qlib."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .feature_spec import DEFAULT_WINDOWS, FEATURE_NAMES, HORIZON, SPLITS


def assign_splits_and_label_end(frame: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Annotate prediction-time split and exact date when label becomes known.

    A label for anchor t uses open(t+1) and open(t+6), hence its information end is
    calendar[t+6]. Remove anchors whose label-information end exceeds split's end.
    Does NOT remove past input observations from an earlier split.
    """
    out = frame.copy()
    positions = calendar.get_indexer(pd.to_datetime(out['datetime']))
    if (positions < 0).any():
        raise ValueError('Some Qlib record dates are not in the trading calendar')
    out['calendar_pos'] = positions.astype(np.int32)
    end_pos = positions + HORIZON + 1
    known = end_pos < len(calendar)
    label_dates = pd.Series(pd.NaT, index=out.index, dtype='datetime64[ns]')
    label_dates.loc[known] = calendar.take(end_pos[known]).to_numpy()
    out['label_end_date'] = label_dates

    out['split'] = 'warmup_or_excluded'
    out['within_split_label'] = False
    for split, start, end in SPLITS:
        anchors = out['datetime'].between(pd.Timestamp(start), pd.Timestamp(end))
        out.loc[anchors, 'split'] = split
        label_ok = anchors & out['label_end_date'].notna() & (out['label_end_date'] <= pd.Timestamp(end))
        out.loc[label_ok, 'within_split_label'] = True
    return out


def with_history_streak(frame: pd.DataFrame, features=FEATURE_NAMES, windows=DEFAULT_WINDOWS) -> pd.DataFrame:
    """Consecutive, finite-feature run within returned rows for each stock.

    This is a *strict diagnostic*. It is NOT the final model's sample mask;
    retrievable pre-index-inclusion stock history and train-only imputations
    may change eligibility. Deliberately no forward filling here.
    """
    out = frame.copy().sort_values(['instrument', 'datetime']).reset_index(drop=True)
    finite = np.isfinite(out[list(features)].to_numpy(dtype=np.float64)).all(axis=1)
    pos = out['calendar_pos'].to_numpy(dtype=np.int64)
    stock = out['instrument'].to_numpy()
    streak = np.zeros(len(out), dtype=np.int32)
    prev_pos = -100
    prev_stock = None
    running = 0
    for i in range(len(out)):
        if finite[i]:
            running = running + 1 if stock[i] == prev_stock and pos[i] == prev_pos + 1 else 1
        else:
            running = 0
        streak[i] = running
        prev_pos, prev_stock = pos[i], stock[i]
    out['all16_finite'] = finite
    out['strict_streak'] = streak
    for w in windows:
        out[f'strict_window_{w}'] = streak >= w
    out['label_finite'] = np.isfinite(pd.to_numeric(out['label'], errors='coerce').to_numpy(dtype=float))
    out['usable_label'] = out['label_finite'] & out['within_split_label']
    return out


def summarize_feature_missingness(frame: pd.DataFrame) -> pd.DataFrame:
    """Report only missing / non-finite rates by split, never holdout returns."""
    records = []
    for split, part in frame.groupby('split', sort=False):
        if split == 'warmup_or_excluded':
            continue
        for feature in FEATURE_NAMES:
            a = pd.to_numeric(part[feature], errors='coerce').to_numpy(dtype='float64')
            records.append({'split': split, 'feature': feature, 'rows': len(part),
                            'finite_count': int(np.isfinite(a).sum()),
                            'nonfinite_rate': round(float((~np.isfinite(a)).mean()), 7)})
    return pd.DataFrame.from_records(records)


def summarize_sample_eligibility(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, part in frame.groupby('split', sort=False):
        if split == 'warmup_or_excluded':
            continue
        rec = {'split': split, 'rows': len(part), 'instruments': part['instrument'].nunique(),
               'days': part['datetime'].nunique(), 'all16_finite_rows': int(part['all16_finite'].sum()),
               'label_finite_rows': int(part['label_finite'].sum()),
               'purged_or_unavailable_label_rows': int((~part['usable_label']).sum())}
        for w in DEFAULT_WINDOWS:
            rec[f'strict_{w}_feature_rows'] = int(part[f'strict_window_{w}'].sum())
            rec[f'strict_{w}_usable_label_rows'] = int((part[f'strict_window_{w}'] & part['usable_label']).sum())
        rows.append(rec)
    return pd.DataFrame(rows)


def train_only_feature_percentiles(frame: pd.DataFrame) -> pd.DataFrame:
    """Quantiles ONLY from train, not leaked from final holdout."""
    train = frame.loc[frame['split'] == 'train']
    rows = []
    for col in FEATURE_NAMES:
        a = pd.to_numeric(train[col], errors='coerce').to_numpy(dtype='float64')
        a = a[np.isfinite(a)]
        if not len(a):
            rows.append({'feature': col, 'count': 0})
        else:
            q = np.quantile(a, [0.001, 0.01, 0.5, 0.99, 0.999])
            rows.append({'feature': col, 'count': int(len(a)), 'p001': float(q[0]),
                         'p01': float(q[1]), 'p50': float(q[2]), 'p99': float(q[3]), 'p999': float(q[4])})
    return pd.DataFrame(rows)
