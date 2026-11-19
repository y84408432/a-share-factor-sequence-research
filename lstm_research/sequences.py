"""V2: leakage-controlled preprocessing and on-demand, shared sequence sample indices.

Only the frozen V1 panel is needed. This module deliberately does not fit models
or compute any predictive metric on validation/development/final_holdout data.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd

from .feature_spec import FEATURE_NAMES, SPLITS

SPLIT_NAMES = tuple(s[0] for s in SPLITS)
WINDOWS = (10, 30, 60)
LOWER_Q = 0.005
UPPER_Q = 0.995


def validate_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Return V1 panel in its original row order after strict contract checks.

    The order matters: all windows below use integer slices in instrument/date
    sorted rows. Never sort here without recreating all V1 streak columns.
    """
    necessary = {
        'instrument', 'datetime', 'split', 'calendar_pos', 'label_end_date',
        'within_split_label', 'label_finite', 'usable_label', 'strict_streak',
        'label', 'strict_window_10', 'strict_window_30', 'strict_window_60',
        *FEATURE_NAMES,
    }
    missing = necessary.difference(panel.columns)
    if missing:
        raise ValueError(f'V1 panel missing required columns: {sorted(missing)}')
    if not len(panel):
        raise ValueError('Empty V1 panel')
    if panel[['instrument', 'datetime']].duplicated().any():
        raise ValueError('Duplicate stock-date records')
    if (panel['instrument'].to_numpy()[1:] < panel['instrument'].to_numpy()[:-1]).any():
        raise ValueError('V1 panel is not sorted by instrument')
    stock = panel['instrument'].to_numpy()
    dates = pd.to_datetime(panel['datetime']).to_numpy()
    within_stock = stock[1:] == stock[:-1]
    if np.any(within_stock & (dates[1:] <= dates[:-1])):
        raise ValueError('V1 panel must have strictly increasing dates per stock')
    if (pd.to_numeric(panel['strict_streak']) < 0).any():
        raise ValueError('Invalid V1 strict streak')
    for w in WINDOWS:
        flag = panel[f'strict_window_{w}'].to_numpy(dtype=bool)
        expected = panel['strict_streak'].to_numpy(dtype=int) >= w
        if not np.array_equal(flag, expected):
            raise ValueError(f'V1 strict_window_{w} flag inconsistent with streak')
    if (~panel['split'].isin((*SPLIT_NAMES, 'warmup_or_excluded'))).any():
        raise ValueError('Unexpected split in V1 data')
    return panel


def fit_train_only_scaler(panel: pd.DataFrame, features: Iterable[str] = FEATURE_NAMES) -> dict:
    """Per-feature 0.5/99.5% clip and robust median/IQR, using TRAIN rows ONLY."""
    features = tuple(features)
    train_mask = panel['split'].to_numpy() == 'train'
    if int(train_mask.sum()) < 50:
        raise ValueError('Insufficient training rows for scaler fit')
    result: dict = {
        'fit_split': 'train', 'clip_quantile_low': LOWER_Q,
        'clip_quantile_high': UPPER_Q, 'center': 'median_after_clip',
        'scale': 'iqr_after_clip', 'missing_policy': 'strict_full_window',
        'features': list(features), 'parameters': {},
    }
    for name in features:
        a = pd.to_numeric(panel.loc[train_mask, name], errors='coerce').to_numpy(dtype=np.float64)
        a = a[np.isfinite(a)]
        if len(a) < 50:
            raise ValueError(f'Insufficient finite training values for {name}')
        low, high = np.quantile(a, [LOWER_Q, UPPER_Q])
        clipped = np.clip(a, low, high)
        med = float(np.median(clipped))
        p25, p75 = np.quantile(clipped, [.25, .75])
        scale = float(p75 - p25)
        used_fallback = False
        if (not np.isfinite(scale)) or scale <= 1e-12:
            scale = float(np.std(clipped))
            used_fallback = True
        if (not np.isfinite(scale)) or scale <= 1e-12:
            scale = 1.0
            used_fallback = True
        result['parameters'][name] = {
            'count_train_finite': int(len(a)), 'clip_low': float(low),
            'clip_high': float(high), 'median': med, 'iqr_or_fallback': scale,
            'used_fallback': used_fallback,
        }
    return result


def scale_feature_column(values: np.ndarray, params: dict) -> np.ndarray:
    """Transform one column. Missing values stored as ZERO, but never eligible.

    Zero is an inert placeholder for invalid panel rows. The strict sequence
    eligibility mask ensures the model will never access those placeholders.
    """
    x = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(x)
    out = np.zeros(x.shape, dtype=np.float32)
    if np.any(mask):
        c = np.clip(x[mask], params['clip_low'], params['clip_high'])
        scaled = (c - params['median']) / params['iqr_or_fallback']
        if not np.isfinite(scaled).all():
            raise ValueError('Non-finite transformed training feature')
        out[mask] = scaled.astype(np.float32)
    return out


def build_sample_index(panel: pd.DataFrame) -> pd.DataFrame:
    """All eligible 10d anchors; 30d/60d flags are nested for FAIR comparisons."""
    valid = panel['usable_label'].to_numpy(dtype=bool)
    eligible10 = panel['strict_window_10'].to_numpy(dtype=bool) & valid
    rows = np.flatnonzero(eligible10)
    cols = ['instrument', 'datetime', 'split', 'calendar_pos', 'label_end_date']
    idx = panel.iloc[rows][cols].copy().reset_index(drop=True)
    idx.insert(0, 'panel_row', rows.astype('int32'))
    idx['eligible30'] = panel['strict_window_30'].to_numpy(dtype=bool)[rows]
    idx['eligible60'] = panel['strict_window_60'].to_numpy(dtype=bool)[rows]
    idx['datetime'] = pd.to_datetime(idx['datetime'])
    idx['label_end_date'] = pd.to_datetime(idx['label_end_date'])
    if idx['label_end_date'].isna().any():
        raise AssertionError('Eligible label lacks end date')
    if not np.array_equal(idx['eligible60'].to_numpy() & idx['eligible30'].to_numpy(),
                          idx['eligible60'].to_numpy()):
        raise AssertionError('A 60d eligible window must also be 30d eligible')
    split_end = {name: pd.Timestamp(end) for name, _start, end in SPLITS}
    for name, part in idx.groupby('split', sort=False):
        if (part['label_end_date'] > split_end[name]).any():
            raise AssertionError(f'Label from {name} reaches across time boundary')
    return idx


def index_for(index: pd.DataFrame, split: str, window: int = 30,
              common_60: bool = False) -> pd.DataFrame:
    """Select one shared set of stock/date anchors for ANY model family."""
    if split not in SPLIT_NAMES:
        raise ValueError('Unknown split')
    if window not in WINDOWS:
        raise ValueError(f'Window not among frozen {WINDOWS}')
    mask = index['split'].to_numpy() == split
    if window >= 30:
        mask &= index['eligible30'].to_numpy(dtype=bool)
    if window >= 60 or common_60:
        mask &= index['eligible60'].to_numpy(dtype=bool)
    return index.loc[mask].reset_index(drop=True)


def validate_window_boundaries(panel: pd.DataFrame, index: pd.DataFrame,
                               windows: tuple[int, ...] = WINDOWS) -> dict:
    """Exhaustively audit EVERY window with vectorized prefix-sum invariants.

    Verify that all reported eligible windows have: a single instrument,
    consecutive daily trading-calendar positions, and all 16 finite raw factors.
    This is a separate check of the V1 streaks, not merely random sampling.
    """
    stocks = panel['instrument'].to_numpy()
    positions = panel['calendar_pos'].to_numpy(dtype=np.int64)
    raw_finite = np.isfinite(panel[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)).all(axis=1)
    n = len(panel)
    edge_invalid = np.zeros(n, dtype=bool)
    edge_invalid[1:] = (stocks[1:] != stocks[:-1]) | (positions[1:] != positions[:-1] + 1)
    edge_prefix = np.concatenate(([0], np.cumsum(edge_invalid, dtype=np.int64)))
    missing_prefix = np.concatenate(([0], np.cumsum(~raw_finite, dtype=np.int64)))
    counts = {}
    for w in windows:
        sub = index_for_all_windows(index, w)
        if sub.empty:
            raise ValueError(f'No {w}d eligible anchors')
        ends = sub['panel_row'].to_numpy(dtype=np.int64)
        starts = ends - w + 1
        if (starts < 0).any():
            raise AssertionError('Window has negative start index')
        if not np.array_equal(stocks[starts], stocks[ends]):
            raise AssertionError('A window crosses instrument boundary')
        bad_edges = edge_prefix[ends+1] - edge_prefix[starts+1]
        bad_values = missing_prefix[ends+1] - missing_prefix[starts]
        if np.any(bad_edges) or np.any(bad_values):
            raise AssertionError(f'{w}d window contains stock/date gap or nonfinite feature')
        counts[str(w)] = int(len(sub))
    return counts


def index_for_all_windows(index: pd.DataFrame, window: int) -> pd.DataFrame:
    if window not in WINDOWS:
        raise ValueError('Invalid window')
    if window == 10:
        return index
    if window == 30:
        return index.loc[index['eligible30']]
    return index.loc[index['eligible60']]


@dataclass
class NumpySequenceDataset:
    """Lazy memmap dataset for SAME anchors across Ridge/MLP/LGBM/LSTM.

    Returns (X: [T,16] float32, y: scalar float32, metadata: dict).
    `window=1` is not exposed here; latest-day baselines use X[-1,:]
    from the identical window and anchor as their sequential competitors.

    IMPORTANT: final_holdout requires explicit allow_final_holdout=True.
    Do not enable for training, hyperparameter tuning or feature selection.
    """
    root: Path
    split: str
    window: int = 30
    common_60: bool = False
    allow_final_holdout: bool = False

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        if self.split == 'final_holdout' and not self.allow_final_holdout:
            raise PermissionError('Holdout locked; only unlock for ONE final evaluation')
        self.features = np.load(self.root / 'features_scaled.npy', mmap_mode='r')
        self.labels = np.load(self.root / 'labels.npy', mmap_mode='r')
        index = pd.read_pickle(self.root / 'sample_index.pkl.gz', compression='gzip')
        self.samples = index_for(index, self.split, self.window, self.common_60)
        if self.features.ndim != 2 or self.features.shape[1] != len(FEATURE_NAMES):
            raise ValueError('Feature-matrix schema mismatch')
        if len(self.features) != len(self.labels):
            raise ValueError('Feature/label row count mismatch')

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, item: int):
        row = self.samples.iloc[int(item)]
        end = int(row['panel_row'])
        start = end - self.window + 1
        x = np.array(self.features[start:end+1, :], dtype=np.float32, copy=True)
        if x.shape != (self.window, len(FEATURE_NAMES)) or not np.isfinite(x).all():
            raise AssertionError('Bad sequence window; investigate V1 streaks')
        y = float(self.labels[end])
        if not np.isfinite(y):
            raise AssertionError('Non-finite prediction label')
        meta = {'instrument': row['instrument'], 'datetime': row['datetime'],
                'label_end_date': row['label_end_date'], 'panel_row': end}
        return x, np.float32(y), meta
