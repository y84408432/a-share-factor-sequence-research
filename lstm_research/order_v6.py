"""V6: controlled historical-order permutation diagnostics for frozen V5 LSTMs.

Randomization: independently permute *past* timesteps per stock/date sample,
keeping the latest-day factor vector untouched and keeping all 16 features within
each day together. For D, train and validation use the same stochastic scheme
but separately sampled permutations; random order is fixed per sample across
training epochs. No new information is introduced and no future-day signal used.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .pilot import evaluate_validation
from .lstm_window_v5 import EVAL_BATCH_SIZE, new_lstm_model, torch_device, check_paired_daily

WINDOWS = (10, 30)
SEEDS = (20261115, 20261116, 20261117)
C_REPEATS = (0, 1, 2)


def shuffle_past(x: np.ndarray, seed: int, *, chunk: int = 10000) -> np.ndarray:
    """Preserve latest snapshot and per-time-step feature vectors exactly.

    Sorting independent uniform random numbers makes a distinct *row-specific*
    permutation with no cross-row mixing. Chunking limits peak RAM usage.
    Identical seed/input/order -> identical output; iteration independent of
    model initialization. Random state is not global.
    """
    if x.ndim != 3 or x.shape[2] != 16 or x.shape[1] < 2:
        raise ValueError('Expected [N, T>=2, 16] ordered feature array')
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite factor in an eligible V5 sample')
    if chunk < 1:
        raise ValueError('Invalid chunk size')
    n, w, f = x.shape
    out = np.empty(x.shape, dtype=np.float32)
    rng = np.random.default_rng(seed)
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        p = rng.random((stop - start, w - 1), dtype=np.float64).argsort(axis=1)
        out[start:stop, :-1, :] = np.take_along_axis(
            x[start:stop, :-1, :], p[:, :, None], axis=1)
        out[start:stop, -1, :] = x[start:stop, -1, :]
    return out


def shuffle_seed(seed: int, *, window: int, experiment: str, repeat: int = 0,
                 split: str = 'valid') -> int:
    """Independent deterministic RNG streams for different comparisons."""
    if window not in WINDOWS or seed not in SEEDS:
        raise ValueError('Unknown V6 seed/window')
    if experiment not in ('C', 'D') or split not in ('train', 'valid'):
        raise ValueError('Unknown experiment/split')
    if not 0 <= repeat < 10:
        raise ValueError('Invalid repeat')
    # Every component contributes a different value, in uint32 RNG seed range.
    return int((seed * 31 + window * 997 + (0 if experiment == 'C' else 100003)
                + repeat * 1009 + (17 if split == 'train' else 53)) % (2**32 - 1))


def predict_checkpoint(x: np.ndarray, checkpoint: Path, *, window: int,
                       seed: int, device_request: str) -> np.ndarray:
    """Evaluate the V5 checkpoint WITHOUT fitting or changing weights."""
    import torch
    if x.ndim != 3 or x.shape[1:] != (window, 16):
        raise ValueError('Input does not match V5 model')
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Missing V5 checkpoint: {checkpoint}')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if (payload.get('window') != window or payload.get('seed') != seed or
            payload.get('input_features') != 16):
        raise ValueError('V5 checkpoint seed/window/features mismatch')
    device = torch_device(device_request)
    model = new_lstm_model()
    model.load_state_dict(payload['state_dict'], strict=True)
    model = model.to(device).eval()
    out = np.empty(len(x), dtype=np.float32)
    with torch.inference_mode():
        for pos in range(0, len(x), EVAL_BATCH_SIZE):
            stop = min(pos + EVAL_BATCH_SIZE, len(x))
            batch = torch.from_numpy(np.ascontiguousarray(x[pos:stop])).to(device)
            out[pos:stop] = model(batch).detach().cpu().numpy()
    if not np.isfinite(out).all():
        raise RuntimeError('Nonfinite V5 checkpoint predictions')
    return out


def compare_daily(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """Mean difference A-B, paired by date and stock counts, HAC lag 10."""
    return check_paired_daily(a, b)


def average_daily(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Mean across checkpoints/repeats *per day*, never pseudo-replicate days."""
    if not frames:
        raise ValueError('No daily metrics')
    a = frames[0].sort_values('date').reset_index(drop=True).copy()
    cs, ics = [], []
    for f in frames:
        b = f.sort_values('date').reset_index(drop=True)
        if not np.array_equal(pd.to_datetime(a.date).to_numpy(), pd.to_datetime(b.date).to_numpy()):
            raise ValueError('Different validation dates')
        if not np.array_equal(a.n_stocks.to_numpy(), b.n_stocks.to_numpy()):
            raise ValueError('Different stock count by date')
        cs.append(b.rankic.to_numpy(dtype=float))
        ics.append(b.ic.to_numpy(dtype=float))
    a['rankic'] = np.stack(cs).mean(axis=0)
    a['ic'] = np.stack(ics).mean(axis=0)
    return a


def cv_eval(dates: np.ndarray, y_raw: np.ndarray, pred_scaled: np.ndarray,
            scaler: dict) -> tuple[dict, pd.DataFrame]:
    from .pilot import pred_to_raw
    return evaluate_validation(dates, y_raw, pred_to_raw(pred_scaled, scaler))
