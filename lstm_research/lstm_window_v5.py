"""V5 preregistered LSTM window diagnostic helpers (validation period only).

The 1/5/10/30-day LSTMs share model structure, all 16 factors, stock/date
anchor cohorts, target scaling, objective, stopping rule, and model capacity.
Only the lookback length and independently specified random seeds differ.
"""
from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd

from .pilot import nw_mean_interval

WINDOWS = (1, 5, 10, 30)
SEEDS = (20261115, 20261116, 20261117)
HIDDEN = 32
EPOCHS = 12
PATIENCE = 3
BATCH_SIZE = 512
EVAL_BATCH_SIZE = 2048
LR = 0.001
WEIGHT_DECAY = 0.001
MIN_DELTA = 1e-5
GRAD_CLIP = 1.0


def window_batch(features: np.ndarray, anchors: np.ndarray, window: int) -> np.ndarray:
    """Build input in temporal order from the same V2 stock/date anchor IDs.

    Consecutive-stock/day/finite eligibility is guaranteed by the previously
    exhaustively audited V2 eligible30 cohort; re-check finite values here.
    """
    if window not in WINDOWS:
        raise ValueError(f'Unregistered window {window}')
    rows = np.asarray(anchors, dtype=np.int64)
    if rows.ndim != 1:
        raise ValueError('Anchors must be 1D')
    if len(rows) and (rows.min() < window - 1 or rows.max() >= len(features)):
        raise ValueError('Sequence exceeds panel row bounds')
    offsets = np.arange(window - 1, -1, -1, dtype=np.int64)
    x = np.asarray(features[rows[:, None] - offsets[None, :]], dtype=np.float32)
    if x.shape != (len(rows), window, 16) or not np.isfinite(x).all():
        raise ValueError('Invalid or nonfinite sequence matrix')
    return x


def materialize_cohort(features: np.ndarray, anchors: np.ndarray,
                       window: int, chunk: int = 8192) -> np.ndarray:
    """One window cached in RAM to avoid repeated memmap random reads per epoch."""
    out = np.empty((len(anchors), window, 16), dtype=np.float32)
    for k in range(0, len(anchors), chunk):
        out[k:k + chunk] = window_batch(features, anchors[k:k + chunk], window)
    return out


def new_lstm_model():
    import torch
    from torch import nn

    class ResearchLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            # Intentionally identical to V3's LSTM architecture.
            self.rnn = nn.LSTM(input_size=16, hidden_size=HIDDEN,
                               num_layers=1, batch_first=True)
            self.head = nn.Sequential(nn.Linear(HIDDEN, 16), nn.ReLU(), nn.Linear(16, 1))

        def forward(self, x):
            if x.ndim != 3 or x.shape[2] != 16:
                raise ValueError('LSTM expects [batch, time, 16]')
            h, _ = self.rnn(x)
            return self.head(h[:, -1, :]).squeeze(-1)

    return ResearchLSTM()


def seeded(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def torch_device(request: str) -> str:
    import torch
    if request == 'auto':
        return 'mps' if torch.backends.mps.is_available() else 'cpu'
    if request not in ('mps', 'cpu'):
        raise ValueError(f'Unrecognized device: {request}')
    if request == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS unavailable, try --device cpu')
    return request


def train_one(xtrain: np.ndarray, train_y: np.ndarray,
              xvalid: np.ndarray, valid_y: np.ndarray,
              *, window: int, seed: int, device_request: str,
              checkpoint: Path, max_epochs: int = EPOCHS,
              patience: int = PATIENCE, batch_size: int = BATCH_SIZE) -> tuple[np.ndarray, dict]:
    """Train from scratch. Checkpoint chosen on VALID MSE, not RankIC.

    Returns predictions in the V3 *scaled-target* convention. Train and
    validation matrices must use exactly the same audited anchor cohort.
    """
    import torch
    from torch import nn
    from torch.nn.utils import clip_grad_norm_

    if window not in WINDOWS or seed not in SEEDS:
        raise ValueError('Experiment not predeclared in V5')
    if xtrain.ndim != 3 or xvalid.ndim != 3 or xtrain.shape[1:] != (window, 16) or xvalid.shape[1:] != (window, 16):
        raise ValueError('Window input tensors have wrong dimensions')
    if len(xtrain) != len(train_y) or len(xvalid) != len(valid_y):
        raise ValueError('X/y counts mismatch')
    if len(xtrain) == 0 or len(xvalid) == 0:
        raise ValueError('Empty split')
    if not (np.isfinite(xtrain).all() and np.isfinite(xvalid).all() and
            np.isfinite(train_y).all() and np.isfinite(valid_y).all()):
        raise ValueError('Nonfinite training/validation input')
    if max_epochs < 1 or patience < 1 or batch_size < 1:
        raise ValueError('Invalid trainer controls')

    seeded(seed)
    device = torch_device(device_request)
    if device == 'cpu':
        torch.set_num_threads(min(4, torch.get_num_threads()))
    model = new_lstm_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    mse = nn.MSELoss()
    best, best_epoch, best_state, bad = float('inf'), 0, None, 0
    trace = []
    # Local RNG for train-row shuffling: independent across window, seed.
    rng = np.random.default_rng(seed)

    for ep in range(1, max_epochs + 1):
        model.train()
        perm = rng.permutation(len(xtrain))
        total = 0.
        for start in range(0, len(perm), batch_size):
            ids = perm[start:start + batch_size]
            xb = torch.from_numpy(np.ascontiguousarray(xtrain[ids])).to(device)
            yb = torch.from_numpy(np.ascontiguousarray(train_y[ids])).to(device)
            opt.zero_grad(set_to_none=True)
            loss = mse(model(xb), yb)
            loss.backward()
            clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            total += float(loss.detach().cpu()) * len(ids)

        model.eval()
        vsum = 0.
        with torch.inference_mode():
            for start in range(0, len(xvalid), EVAL_BATCH_SIZE):
                end = start + EVAL_BATCH_SIZE
                xb = torch.from_numpy(np.ascontiguousarray(xvalid[start:end])).to(device)
                yb = torch.from_numpy(np.ascontiguousarray(valid_y[start:end])).to(device)
                residual = model(xb) - yb
                vsum += float(torch.sum(residual.square()).cpu())
        train_mse = total / len(xtrain)
        val_mse = vsum / len(xvalid)
        trace.append({'epoch': ep, 'train_mse_scaled': train_mse, 'valid_mse_scaled': val_mse})
        print(f'  LSTM-{window:02d} seed={seed}: epoch {ep:02d}/{max_epochs}, '
              f'train MSE={train_mse:.5f}, valid MSE={val_mse:.5f}', flush=True)
        if np.isfinite(val_mse) and val_mse < best - MIN_DELTA:
            best, best_epoch = val_mse, ep
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is None:
        raise RuntimeError('Training failed to produce a finite validation checkpoint')
    model.load_state_dict(best_state)
    model.eval()
    pred = np.empty(len(xvalid), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(xvalid), EVAL_BATCH_SIZE):
            stop = min(len(xvalid), start + EVAL_BATCH_SIZE)
            xb = torch.from_numpy(np.ascontiguousarray(xvalid[start:stop])).to(device)
            pred[start:stop] = model(xb).cpu().numpy()
    if not np.isfinite(pred).all():
        raise RuntimeError('Nonfinite predictions')
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': best_state, 'window': window, 'seed': seed,
                'best_epoch': best_epoch, 'input_features': 16}, checkpoint)
    info = {'best_epoch_by_valid_mse': best_epoch, 'best_valid_mse_scaled': best,
            'epochs_run': len(trace), 'trace': trace, 'device': device,
            'parameter_count': sum(p.numel() for p in model.parameters())}
    return pred, info


def check_paired_daily(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """Newey-West paired daily RankIC difference; exact date/stock counts."""
    from .pilot import paired_rankic
    if not np.array_equal(pd.to_datetime(a.date).to_numpy(), pd.to_datetime(b.date).to_numpy()):
        raise ValueError('Daily validation dates are not identical')
    if not np.array_equal(a.n_stocks.to_numpy(), b.n_stocks.to_numpy()):
        raise ValueError('Comparison uses different cross-sectional stock counts')
    return paired_rankic(a, b)


def across_seeds_daily(daily_files: list[pd.DataFrame]) -> pd.DataFrame:
    """Average per-date *scores*, NOT RankIC of an ensemble of predictions."""
    if len(daily_files) != len(SEEDS):
        raise ValueError('V5 requires exactly three predeclared seeds')
    reference = daily_files[0].sort_values('date').reset_index(drop=True).copy()
    ranks = [reference['rankic'].to_numpy(dtype=float)]
    ics = [reference['ic'].to_numpy(dtype=float)]
    for frame in daily_files[1:]:
        other = frame.sort_values('date').reset_index(drop=True)
        if not np.array_equal(pd.to_datetime(reference.date), pd.to_datetime(other.date)):
            raise ValueError('Different seed evaluated different days')
        if not np.array_equal(reference.n_stocks.to_numpy(), other.n_stocks.to_numpy()):
            raise ValueError('Different seed evaluated different stocks')
        ranks.append(other['rankic'].to_numpy(dtype=float))
        ics.append(other['ic'].to_numpy(dtype=float))
    reference['rankic'] = np.mean(np.stack(ranks), axis=0)
    reference['ic'] = np.mean(np.stack(ics), axis=0)
    return reference


def summarize_window(seed_metrics: list[dict], daily_files: list[pd.DataFrame]) -> dict:
    daily = across_seeds_daily(daily_files)
    r = np.asarray([m['rankic_daily_equal_weight']['mean'] for m in seed_metrics], dtype=float)
    i = np.asarray([m['ic_daily_equal_weight']['mean'] for m in seed_metrics], dtype=float)
    raw_mse = np.asarray([m['mse_raw'] for m in seed_metrics], dtype=float)
    return {
        'n_seeds': len(SEEDS),
        'n_valid_days': int(len(daily)),
        'rankic_seed_mean': float(r.mean()),
        'rankic_seed_std': float(r.std(ddof=1)),
        'rankic_seed_min': float(r.min()),
        'rankic_seed_max': float(r.max()),
        'rankic_daily_seed_mean_hac': nw_mean_interval(daily.rankic.to_numpy()),
        'ic_seed_mean': float(i.mean()),
        'ic_seed_std': float(i.std(ddof=1)),
        'raw_mse_seed_mean': float(raw_mse.mean()),
        'yearly_seed_mean_rankic': {str(int(yr)): float(grp.rankic.mean())
            for yr, grp in daily.groupby(pd.to_datetime(daily.date).dt.year)}
    }
