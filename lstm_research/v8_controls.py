"""V8: exact-cohort, validation-only controls for A30 daily OHLCV sequences.

D-order control: independent within-sample permutations of the first 29 time
steps for BOTH train and validation; last day stays at the last position, and
all 8 channels from a single day move together. Three independently generated
permutation REPEATS and three neural seeds make nine D fits.

Non-recurrent baselines: frozen Ridge(alpha=1000) and V3-style flattened MLP,
using the *same 30x8 A-feature matrices* as the original V7 A30 LSTM.
"""
from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np

from .lstm_window_v5 import (
    SEEDS, EPOCHS, PATIENCE, BATCH_SIZE, EVAL_BATCH_SIZE, LR,
    WEIGHT_DECAY, MIN_DELTA, GRAD_CLIP,
)
from .pilot import nw_mean_interval

WINDOW = 30
FEATURES = 8
REPEATS = (0, 1, 2)
RIDGE_ALPHA = 1000.0
MLP_HIDDEN = (64, 16)
MLP_DROPOUT = 0.1


def perm_seed(seed: int, repeat: int, split: str) -> int:
    """Predictable independent RNG stream; no global RNG contamination."""
    if seed not in SEEDS or repeat not in REPEATS or split not in ('train', 'valid'):
        raise ValueError('Unregistered seed, repeat or split')
    return int((seed * 37 + repeat * 1009 + (100003 if split == 'train' else 200003)) % (2**32-1))


def permute_past(x: np.ndarray, seed: int, chunk: int=8192) -> np.ndarray:
    """Permute rows' first 29 *whole time-step vectors*, not channels/stocks.

    The same static permutation applies across all training epochs for a given
    stock-date, but train and validation permutation arrays are independently
    drawn from the same sampling mechanism. Output preserves latest day.
    """
    if x.ndim!=3 or x.shape[1:]!=(WINDOW, FEATURES) or x.dtype!=np.float32:
        raise ValueError('Expected float32 [N,30,8]')
    if chunk<1 or not np.isfinite(x).all():
        raise ValueError('Invalid permutation input')
    n=len(x)
    out=np.empty_like(x)
    rng=np.random.default_rng(seed)
    for lo in range(0,n,chunk):
        hi=min(lo+chunk,n)
        permutation=rng.random((hi-lo,WINDOW-1)).argsort(axis=1)
        out[lo:hi,:-1,:]=np.take_along_axis(x[lo:hi,:-1,:],permutation[:,:,None],axis=1)
        out[lo:hi,-1,:]=x[lo:hi,-1,:]
    return out


def flatten_a30(x: np.ndarray) -> np.ndarray:
    if x.ndim!=3 or x.shape[1:]!=(WINDOW,FEATURES):
        raise ValueError('Expected [N,30,8]')
    out=np.ascontiguousarray(x.reshape(len(x),WINDOW*FEATURES),dtype=np.float32)
    if not np.isfinite(out).all():
        raise ValueError('Nonfinite flattened A30 input')
    return out


def fit_ridge(xtr: np.ndarray, ytr: np.ndarray, xva: np.ndarray):
    """V3 Ridge configuration, now with exactly 240 lagged A inputs."""
    from sklearn.linear_model import Ridge
    if xtr.ndim!=2 or xva.ndim!=2 or xtr.shape[1]!=WINDOW*FEATURES or xva.shape[1]!=WINDOW*FEATURES:
        raise ValueError('Expected [N,240] flat A30 Ridge inputs')
    if len(xtr)!=len(ytr):
        raise ValueError('Ridge X/y length mismatch')
    model=Ridge(alpha=RIDGE_ALPHA,solver='lsqr',fit_intercept=True,tol=1e-3)
    model.fit(xtr,ytr)
    p=np.asarray(model.predict(xva),dtype=np.float32)
    if not np.isfinite(p).all():
        raise RuntimeError('Nonfinite Ridge predictions')
    return p,model


def flat_mlp():
    from torch import nn
    # Match the V3 flattened MLP's architecture (240->64->16->1),
    # NOT the V7 LSTM's different 8->32 recurrent architecture.
    return nn.Sequential(
        nn.Linear(WINDOW*FEATURES,MLP_HIDDEN[0]),nn.ReLU(),nn.Dropout(MLP_DROPOUT),
        nn.Linear(MLP_HIDDEN[0],MLP_HIDDEN[1]),nn.ReLU(),nn.Linear(MLP_HIDDEN[1],1),
    )


def fit_flat_mlp(xtr:np.ndarray,ytr:np.ndarray,xva:np.ndarray,yva:np.ndarray,
                 *,seed:int,device:str,checkpoint:Path,
                 max_epochs:int=EPOCHS,patience:int=PATIENCE,batch_size:int=BATCH_SIZE):
    """Frozen V3-like flattened MLP with V7 target scaling and validation-MSE stop.

    A30 LSTM is reused, not tuned based on V8. Predictions are scaled-target.
    """
    import torch
    from torch import nn
    from torch.nn.utils import clip_grad_norm_
    if seed not in SEEDS or xtr.ndim!=2 or xva.ndim!=2 or xtr.shape[1]!=240 or xva.shape[1]!=240:
        raise ValueError('Unregistered seed or wrong MLP input')
    if not (len(xtr)==len(ytr) and len(xva)==len(yva)):
        raise ValueError('Invalid MLP X/y sizes')
    if not (np.isfinite(xtr).all() and np.isfinite(xva).all() and
            np.isfinite(ytr).all() and np.isfinite(yva).all()):
        raise ValueError('Nonfinite MLP data')
    if max_epochs<1 or patience<1 or batch_size<1:
        raise ValueError('Bad training parameters')
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.backends.mps.is_available(): torch.mps.manual_seed(seed)
    if device=='cpu':torch.set_num_threads(min(4,torch.get_num_threads()))
    model=flat_mlp().to(device)
    optim=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    mse=nn.MSELoss()
    best=float('inf');best_state=None;best_epoch=0;bad=0;trace=[]
    rng=np.random.default_rng(seed)
    for ep in range(1,max_epochs+1):
        model.train();total=0.
        order=rng.permutation(len(xtr))
        for j in range(0,len(order),batch_size):
            ids=order[j:j+batch_size]
            xb=torch.from_numpy(np.ascontiguousarray(xtr[ids])).to(device)
            yy=torch.from_numpy(np.ascontiguousarray(ytr[ids])).to(device)
            optim.zero_grad(set_to_none=True)
            loss=mse(model(xb).squeeze(-1),yy)
            loss.backward();clip_grad_norm_(model.parameters(),GRAD_CLIP)
            optim.step();total+=float(loss.detach().cpu())*len(ids)
        model.eval();valid_sum=0.
        with torch.inference_mode():
            for j in range(0,len(xva),EVAL_BATCH_SIZE):
                stop=min(j+EVAL_BATCH_SIZE,len(xva))
                xb=torch.from_numpy(np.ascontiguousarray(xva[j:stop])).to(device)
                yy=torch.from_numpy(np.ascontiguousarray(yva[j:stop])).to(device)
                valid_sum+=float((model(xb).squeeze(-1)-yy).square().sum().cpu())
        vmse=valid_sum/len(yva);tmse=total/len(ytr)
        trace.append({'epoch':ep,'train_mse_scaled':tmse,'valid_mse_scaled':vmse})
        print(f'   MLP seed={seed}: epoch {ep:02d}/{max_epochs}, train MSE={tmse:.5f}, valid MSE={vmse:.5f}',flush=True)
        if np.isfinite(vmse) and vmse<best-MIN_DELTA:
            best=vmse;best_epoch=ep
            best_state=copy.deepcopy({k:v.detach().cpu() for k,v in model.state_dict().items()})
            bad=0
        else:
            bad+=1
            if bad>=patience:break
    if best_state is None:raise RuntimeError('No finite MLP validation checkpoint')
    model.load_state_dict(best_state);model.eval()
    pred=np.empty(len(xva),dtype=np.float32)
    with torch.inference_mode():
        for j in range(0,len(xva),EVAL_BATCH_SIZE):
            stop=min(j+EVAL_BATCH_SIZE,len(xva))
            xb=torch.from_numpy(np.ascontiguousarray(xva[j:stop])).to(device)
            pred[j:stop]=model(xb).squeeze(-1).cpu().numpy()
    if not np.isfinite(pred).all():raise RuntimeError('Nonfinite MLP prediction')
    checkpoint.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'state_dict':best_state,'model':'A30_flat_MLP','seed':seed,
                'window':WINDOW,'channels':FEATURES,'best_epoch':best_epoch},checkpoint)
    return pred,{'best_epoch':best_epoch,'valid_mse_scaled':best,
                 'trace':trace,'epochs_run':len(trace),'device':device,
                 'parameter_count':sum(p.numel() for p in model.parameters())}


def same_day_metrics(frames):
    """Average DAILY RankIC across models; do not pseudo-replicate dates."""
    import pandas as pd
    if not frames:raise ValueError('No daily input metrics')
    first=frames[0].sort_values('date').reset_index(drop=True).copy()
    rank=[];ic=[]
    for f in frames:
        f=f.sort_values('date').reset_index(drop=True)
        if (not np.array_equal(pd.to_datetime(first.date),pd.to_datetime(f.date)) or
                not np.array_equal(first.n_stocks.to_numpy(),f.n_stocks.to_numpy())):
            raise ValueError('Daily model results are not aligned')
        rank.append(f.rankic.to_numpy(float));ic.append(f.ic.to_numpy(float))
    first['rankic']=np.stack(rank).mean(axis=0)
    first['ic']=np.stack(ic).mean(axis=0)
    return first


def mean_daily_diff(a,b):
    import pandas as pd
    aa=a.sort_values('date').reset_index(drop=True)
    bb=b.sort_values('date').reset_index(drop=True)
    if (not np.array_equal(pd.to_datetime(aa.date),pd.to_datetime(bb.date)) or
            not np.array_equal(aa.n_stocks.to_numpy(),bb.n_stocks.to_numpy())):
        raise ValueError('Paired daily comparisons require exact same dates and stock counts')
    return nw_mean_interval(aa.rankic.to_numpy(float)-bb.rankic.to_numpy(float))
