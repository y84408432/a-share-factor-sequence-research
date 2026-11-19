"""V4: tests of additional lagged-factor information and flattened-MLP regularization.

NO access to development/final evaluation periods. This module is additive to V0-V3.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .pilot import (
    SEED, WINDOW, anchor_sha256, make_batch, nw_mean_interval,
    pred_to_raw, evaluate_validation, paired_rankic, set_seeds, select_torch_device,
)

RIDGE_WINDOWS = (1, 5, 30)
MLP_VARIANTS = {
    'mlp_flat_baseline': {'hidden': (64, 16), 'dropout': 0.1, 'weight_decay': 0.001},
    'mlp_flat_small': {'hidden': (32, 8), 'dropout': 0.1, 'weight_decay': 0.001},
    'mlp_flat_strong_wd': {'hidden': (64, 16), 'dropout': 0.1, 'weight_decay': 0.03},
}


def flat_window(features: np.ndarray, anchors: np.ndarray, window: int, chunk: int = 8192) -> np.ndarray:
    """Flatten lagged features oldest->newest. Same anchor set for 1/5/30.

    Only rows <= anchor are accessed, using strict-30 eligible anchor IDs supplied by caller.
    """
    if window not in RIDGE_WINDOWS:
        raise ValueError('Only predeclared 1/5/30 windows are permitted in V4')
    rows = np.asarray(anchors, dtype=np.int64)
    if rows.ndim != 1 or np.any(rows < window-1):
        raise ValueError('Invalid anchor row')
    nfeat = features.shape[1]
    result = np.empty((len(rows), nfeat*window), dtype=np.float32)
    for k in range(0, len(rows), chunk):
        batch = make_batch(features, rows[k:k+chunk], window=window)
        result[k:k+len(batch)] = batch.reshape(len(batch), window*nfeat)
    if not np.isfinite(result).all():
        raise ValueError('A strict sample contains NaN or Inf')
    return result


def load_v3_baseline(v3: Path, model: str, valid_anchor: str, n_valid: int) -> tuple[dict, pd.DataFrame]:
    """V3 baseline is reused, never retrained or reselected on validation RankIC."""
    if model not in ('ridge', 'mlp_flat'):
        raise ValueError(model)
    met = v3/f'{model}_valid_metrics.json'
    day = v3/f'{model}_daily_rankic.csv'
    if not (met.is_file() and day.is_file()):
        raise FileNotFoundError(f'V3 baseline files missing for {model}; complete original V3 first')
    stats = json.loads(met.read_text(encoding='utf-8'))
    if stats.get('valid_anchor_sha256') != valid_anchor or stats.get('n_samples') != n_valid:
        raise ValueError('V3 baseline anchor fingerprint or sample count mismatch')
    if stats.get('window') != WINDOW or stats.get('seed') != SEED:
        raise ValueError('V3 baseline used different window or random seed')
    daily = pd.read_csv(day, parse_dates=['date'])
    needed = {'date','rankic','ic','n_stocks'}
    if not needed.issubset(daily.columns) or daily.date.duplicated().any():
        raise ValueError('V3 daily baseline columns or date keys invalid')
    return stats, daily.sort_values('date').reset_index(drop=True)


def compare_daily(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """Paired date-level RankIC delta. Verify identical days AND cross-sections."""
    if not np.array_equal(pd.to_datetime(a.date).to_numpy(), pd.to_datetime(b.date).to_numpy()):
        raise ValueError('Mismatched validation trading dates')
    if not np.array_equal(a.n_stocks.to_numpy(), b.n_stocks.to_numpy()):
        raise ValueError('Mismatched cross-section sizes')
    return paired_rankic(a, b)


def make_mlp(variant: str):
    import torch.nn as nn
    cfg = MLP_VARIANTS[variant]
    h1, h2 = cfg['hidden']
    return nn.Sequential(
        nn.Linear(WINDOW*16, h1), nn.ReLU(), nn.Dropout(cfg['dropout']),
        nn.Linear(h1, h2), nn.ReLU(), nn.Linear(h2,1),
    )


def train_mlp_variant(
    variant: str, features: np.ndarray,
    train_rows: np.ndarray, y_train_scaled: np.ndarray,
    valid_rows: np.ndarray, y_valid_scaled: np.ndarray,
    device_request: str, checkpoint: Path,
) -> tuple[np.ndarray, dict]:
    """Matches V3 training procedure except the predeclared variant's width or AdamW decay.

    Checkpoint chosen by VALID scaled MSE, NOT RankIC. Max 12 epochs/patience 3.
    """
    import torch
    from torch import nn
    from torch.nn.utils import clip_grad_norm_

    if variant not in ('mlp_flat_small','mlp_flat_strong_wd'):
        raise ValueError('Only two new frozen V4 variants may be trained')
    set_seeds(SEED)
    device = select_torch_device(device_request)
    if device == 'cpu':
        torch.set_num_threads(min(4,torch.get_num_threads()))
    net = make_mlp(variant).to(device)
    cfg = MLP_VARIANTS[variant]
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=cfg['weight_decay'])
    loss_fn=nn.MSELoss()
    best_valid = float('inf')
    best_epoch = 0
    best_state = None
    stale = 0
    trace = []

    def inputs(rows):
        x=make_batch(features, rows, window=WINDOW)
        return x.reshape(len(rows),WINDOW*16)

    for ep in range(1,13):
        net.train()
        order = np.random.permutation(len(train_rows))
        train_mse = 0.0
        for j in range(0,len(order),512):
            ids = order[j:j+512]
            x = torch.from_numpy(np.array(inputs(train_rows[ids]),copy=True)).to(device)
            y = torch.from_numpy(np.array(y_train_scaled[ids],copy=True)).to(device)
            opt.zero_grad(set_to_none=True)
            prediction = net(x).squeeze(-1)
            loss = loss_fn(prediction,y)
            loss.backward()
            clip_grad_norm_(net.parameters(),1.)
            opt.step()
            train_mse += float(loss.detach().cpu())*len(ids)
        net.eval()
        val_error_sum = 0.0
        with torch.inference_mode():
            for j in range(0,len(valid_rows),2048):
                x=torch.from_numpy(np.array(inputs(valid_rows[j:j+2048]),copy=True)).to(device)
                y=torch.from_numpy(np.array(y_valid_scaled[j:j+2048],copy=True)).to(device)
                pp=net(x).squeeze(-1)
                val_error_sum += float(torch.square(pp-y).sum().cpu())
        vmse = val_error_sum/len(valid_rows)
        tmse = train_mse/len(train_rows)
        trace.append({'epoch':ep,'train_mse_scaled':tmse,'valid_mse_scaled':vmse})
        print(f'  {variant}: epoch {ep:02d}/12, train MSE={tmse:.5f}, valid MSE={vmse:.5f}',flush=True)
        if vmse < best_valid-1e-5:
            best_valid=vmse
            best_epoch=ep
            best_state=copy.deepcopy({k:v.detach().cpu() for k,v in net.state_dict().items()})
            stale=0
        else:
            stale+=1
            if stale >= 3:
                break
    if best_state is None:
        raise RuntimeError('No valid checkpoint')
    net.load_state_dict(best_state)
    checkpoint.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'state_dict':best_state,'kind':variant,'seed':SEED,'window':WINDOW,
                'best_epoch':best_epoch,'config':cfg},checkpoint)
    net.eval()
    predictions=np.empty(len(valid_rows),dtype=np.float32)
    with torch.inference_mode():
        for j in range(0,len(valid_rows),2048):
            rr=valid_rows[j:j+2048]
            x=torch.from_numpy(np.array(inputs(rr),copy=True)).to(device)
            predictions[j:j+len(rr)]=net(x).squeeze(-1).detach().cpu().numpy()
    return predictions,{'trace':trace,'best_epoch_by_valid_mse':best_epoch,
        'epochs_run':len(trace),'best_valid_mse_scaled':best_valid,
        'parameter_count':sum(p.numel() for p in net.parameters()),
        'device':device,'variant_config':cfg}


def metrics_to_row(model: str, obj: dict) -> dict:
    ri=obj['rankic_daily_equal_weight']
    return {'model':model,'n_valid':obj['n_samples'],'n_days':ri['n_days'],
            'rankic':ri['mean'],'rankic_hac_lo':ri['ci95'][0],
            'rankic_hac_hi':ri['ci95'][1],
            'ic':obj['ic_daily_equal_weight']['mean'],'mse_raw':obj['mse_raw']}


def verify_label_scaler(saved: dict, reconstructed: dict) -> None:
    if saved.get('fit_split') != 'train' or reconstructed.get('fit_split') != 'train':
        raise ValueError('Label scaler is not train-only')
    for key in ('low','high','mean','std'):
        if not np.isclose(saved[key], reconstructed[key], rtol=1e-12, atol=1e-12):
            raise ValueError(f'V3 label scaler disagreement on {key}')
