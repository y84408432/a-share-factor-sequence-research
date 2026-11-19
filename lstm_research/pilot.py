"""V3 frozen first-pass model comparison: validation only, no development/holdout access.

V2 feature matrices are stored by instrument/date as a Numpy memmap. All five
models share exactly the same eligible-30 stock-date anchor IDs and labels.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from .sequences import index_for

MODELS = ('ridge', 'mlp_last', 'mlp_flat', 'lightgbm', 'lstm')
SEED = 20261115
WINDOW = 30
MIN_CROSS_SECTION = 30
HAC_LAGS = 10


def anchor_sha256(index: pd.DataFrame) -> str:
    """Fingerprint EXACT subset/order, not aggregate date range or row count."""
    rows = index['panel_row'].to_numpy(dtype=np.int64)
    return hashlib.sha256(rows.tobytes()).hexdigest()


def make_batch(features: np.ndarray, rows: np.ndarray, window: int = WINDOW) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    offsets = np.arange(window-1, -1, -1, dtype=np.int64)
    if len(rows) and rows.min() < window-1:
        raise ValueError('Invalid anchor: insufficient previous rows')
    return np.asarray(features[rows[:, None]-offsets[None, :]], dtype=np.float32)


def stack_sequences(features: np.ndarray, rows: np.ndarray, flatten: bool, chunk=8192) -> np.ndarray:
    """Materialize one cohort at a time; cannot accidentally leak next-day prices."""
    count, f = len(rows), features.shape[1]
    out = np.empty((count, WINDOW*f) if flatten else (count, WINDOW, f), dtype=np.float32)
    for j in range(0, count, chunk):
        x = make_batch(features, rows[j:j+chunk])
        out[j:j+chunk] = x.reshape(len(x), WINDOW*f) if flatten else x
    if not np.isfinite(out).all():
        raise ValueError('Nonfinite values in eligible sample sequence')
    return out


def fit_train_label_scaler(y: np.ndarray) -> dict:
    """Training target robustification is deliberately trained on TRAIN only."""
    y = np.asarray(y, dtype=np.float64)
    if not np.isfinite(y).all():
        raise ValueError('Training label has NaN/inf')
    lo, hi = np.quantile(y, [0.005, 0.995])
    clipped = np.clip(y, lo, hi)
    center, scale = float(np.mean(clipped)), float(np.std(clipped))
    if scale < 1e-9:
        raise ValueError('Train target scale degenerate')
    return {'low':float(lo),'high':float(hi),'mean':center,'std':scale,
            'fit_split':'train','clip_training_label_only':True}


def target_for_train(y: np.ndarray, cfg: dict) -> np.ndarray:
    return ((np.clip(y, cfg['low'], cfg['high'])-cfg['mean'])/cfg['std']).astype(np.float32)


def target_for_valid(y: np.ndarray, cfg: dict) -> np.ndarray:
    """Validation label is NOT winsorized: objective evaluates true outcomes."""
    return ((np.asarray(y)-cfg['mean'])/cfg['std']).astype(np.float32)


def pred_to_raw(p: np.ndarray, cfg: dict) -> np.ndarray:
    return np.asarray(p, dtype=np.float64)*cfg['std']+cfg['mean']


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x, y = x-x.mean(), y-y.mean()
    den = np.sqrt(float(x@x)*float(y@y))
    return float(x@y/den) if den > 1e-16 else float('nan')


def daily_cs_metrics(dates: np.ndarray, actual: np.ndarray, predicted: np.ndarray,
                     min_n: int = MIN_CROSS_SECTION) -> pd.DataFrame:
    """Daily equal-weight cross-sectional RankIC, not pooled row-wise rank correlation."""
    if not len(dates) == len(actual) == len(predicted):
        raise ValueError('Prediction, label and date length mismatch')
    if not (np.isfinite(actual).all() and np.isfinite(predicted).all()):
        raise ValueError('Nonfinite label or prediction')
    df = pd.DataFrame({'date': pd.to_datetime(dates), 'label':actual, 'pred':predicted})
    rows=[]
    for date, sub in df.groupby('date', sort=True):
        if len(sub) < min_n:
            continue
        p, y = sub.pred.to_numpy(), sub.label.to_numpy()
        ric=safe_corr(rankdata(p),rankdata(y))
        ic=safe_corr(p,y)
        rows.append({'date':date, 'n_stocks':len(sub),'rankic':ric,'ic':ic})
    out=pd.DataFrame(rows)
    if out.empty or out.rankic.isna().any():
        raise ValueError('Too few cross sections or degenerate predictions')
    return out


def nw_mean_interval(values: np.ndarray, lags: int = HAC_LAGS) -> dict:
    """Newey-West long-run variance of the equal-weight DAILY mean, fixed lag 10."""
    z=np.asarray(values,dtype=np.float64)
    z=z[np.isfinite(z)]
    n=len(z)
    if n < 15:
        return {'n_days':n,'mean':float(np.mean(z)) if n else None,'se_hac':None,'ci95':None}
    mu=float(z.mean())
    a=z-mu
    v=float(a@a/n)
    for k in range(1,min(lags,n-1)+1):
        weight=1-k/(min(lags,n-1)+1)
        v+=2*weight*float(a[k:]@a[:-k]/n)
    se=float(np.sqrt(max(0.,v)/n))
    return {'n_days':n,'mean':mu,'se_hac':se,'ci95':[mu-1.96*se,mu+1.96*se]}


def evaluate_validation(dates: np.ndarray, y: np.ndarray, p: np.ndarray) -> tuple[dict,pd.DataFrame]:
    daily=daily_cs_metrics(dates,y,p)
    overall={
        'n_samples':int(len(y)),
        'mse_raw':float(np.mean(np.square(y-p))),
        'prediction_std':float(np.std(p)),
        'rankic_daily_equal_weight':nw_mean_interval(daily.rankic.to_numpy()),
        'ic_daily_equal_weight':nw_mean_interval(daily.ic.to_numpy()),
        'yearly':[],
    }
    for year, grp in daily.groupby(daily['date'].dt.year):
        overall['yearly'].append({'year':int(year),'n_days':int(len(grp)),
                                  'rankic':float(grp.rankic.mean()),'ic':float(grp.ic.mean())})
    return overall,daily


def paired_rankic(d1: pd.DataFrame,d2: pd.DataFrame) -> dict:
    merged=d1[['date','rankic']].merge(d2[['date','rankic']],on='date',suffixes=('_a','_b'),validate='one_to_one')
    if len(merged)!=len(d1) or len(merged)!=len(d2):
        raise ValueError('Paired comparison lacks matched trading days')
    return nw_mean_interval(merged.rankic_a.to_numpy()-merged.rankic_b.to_numpy())


def save_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2), encoding='utf-8')


def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def select_torch_device(request: str) -> str:
    import torch
    if request == 'auto':
        return 'mps' if torch.backends.mps.is_available() else 'cpu'
    if request == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS unavailable; rerun with --device cpu')
    return request


def fit_torch_model(kind: str, features: np.ndarray,
                    rows_train: np.ndarray, y_train: np.ndarray,
                    rows_valid: np.ndarray, y_valid: np.ndarray,
                    config: dict, checkpoint: Path, device_request: str='auto') -> tuple[np.ndarray,dict]:
    import copy
    import torch
    from torch import nn
    from torch.nn.utils import clip_grad_norm_
    set_seeds(SEED)
    device=select_torch_device(device_request)
    if device=='cpu':
        torch.set_num_threads(min(8,torch.get_num_threads()))

    class MLP(nn.Module):
        def __init__(self, d: int):
            super().__init__()
            h=32 if d==16 else 64
            self.net=nn.Sequential(nn.Linear(d,h),nn.ReLU(),nn.Dropout(.1),
                    nn.Linear(h,16),nn.ReLU(),nn.Linear(16,1))
        def forward(self,x): return self.net(x).squeeze(-1)

    class LSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn=nn.LSTM(input_size=16,hidden_size=32,num_layers=1,batch_first=True)
            self.head=nn.Sequential(nn.Linear(32,16),nn.ReLU(),nn.Linear(16,1))
        def forward(self,x):
            seq,_=self.rnn(x)
            return self.head(seq[:,-1,:]).squeeze(-1)

    if kind=='lstm':
        model=LSTM().to(device)
    elif kind in ('mlp_last','mlp_flat'):
        model=MLP(16 if kind=='mlp_last' else 16*WINDOW).to(device)
    else:
        raise ValueError(kind)

    # Pure inference / optimization batches. SAME eligible30 rows for all models.
    def x_batch(rows):
        if kind=='mlp_last': return np.asarray(features[rows], dtype=np.float32)
        xx=make_batch(features,rows)
        return xx.reshape(len(rows), -1) if kind=='mlp_flat' else xx

    optimiser=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-3)
    loss_fn=nn.MSELoss()
    batch_size=512
    epochs=12
    patience=3
    best=float('inf')
    best_epoch=0
    best_state=None
    bad=0
    trace=[]
    for epoch in range(1,epochs+1):
        model.train()
        train_perm=np.random.permutation(len(rows_train))
        train_loss_sum=0.
        for i in range(0,len(train_perm),batch_size):
            ids=train_perm[i:i+batch_size]
            x=torch.from_numpy(np.array(x_batch(rows_train[ids]),copy=True)).to(device)
            yy=torch.from_numpy(np.array(y_train[ids],copy=True)).to(device)
            optimiser.zero_grad(set_to_none=True)
            pp=model(x)
            loss=loss_fn(pp,yy)
            loss.backward()
            clip_grad_norm_(model.parameters(),1.)
            optimiser.step()
            train_loss_sum+=float(loss.detach().cpu())*len(ids)
        model.eval()
        val_loss_sum=0.
        with torch.inference_mode():
            for j in range(0,len(rows_valid),2048):
                r=rows_valid[j:j+2048]
                xx=torch.from_numpy(np.array(x_batch(r),copy=True)).to(device)
                yy=torch.from_numpy(np.array(y_valid[j:j+2048],copy=True)).to(device)
                pp=model(xx)
                val_loss_sum+=float(torch.sum(torch.square(pp-yy)).cpu())
        val_mse=val_loss_sum/len(rows_valid)
        train_mse=train_loss_sum/len(rows_train)
        trace.append({'epoch':epoch,'train_mse_scaled':train_mse,'valid_mse_scaled':val_mse})
        print(f'    {kind}: epoch {epoch:02d}/{epochs}, train MSE={train_mse:.5f}, valid MSE={val_mse:.5f}',flush=True)
        if val_mse < best-1e-5:
            best=val_mse;best_epoch=epoch
            best_state=copy.deepcopy({k:v.detach().cpu() for k,v in model.state_dict().items()})
            bad=0
        else:
            bad+=1
            if bad>=patience: break
    if best_state is None:
        raise RuntimeError('No valid checkpoint')
    model.load_state_dict(best_state)
    torch.save({'state_dict':best_state,'kind':kind,'window':WINDOW,'seed':SEED,'epoch':best_epoch,
                'feature_names':list(config['features'])},checkpoint)
    model.eval()
    predictions=np.empty(len(rows_valid),dtype=np.float32)
    with torch.inference_mode():
        for j in range(0,len(rows_valid),2048):
            r=rows_valid[j:j+2048]
            xx=torch.from_numpy(np.array(x_batch(r),copy=True)).to(device)
            predictions[j:j+len(r)]=model(xx).detach().cpu().numpy()
    return predictions,{'epochs_run':len(trace),'best_epoch_by_valid_mse':best_epoch,
                        'best_valid_mse_scaled':best,'device':device,'trace':trace,
                        'parameter_count':sum(p.numel() for p in model.parameters())}

