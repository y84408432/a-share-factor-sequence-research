"""V7: minimal-OHLCV vs frozen-rolling-factor representation experiment.

Only train/validation period outcomes are eligible to be used in this stage.
The A features are intentionally short-memory daily features (no rolling statistics
beyond one prior trading day) and are evaluated against the predeclared B factors.
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .lstm_window_v5 import (
    SEEDS, HIDDEN, EPOCHS, PATIENCE, BATCH_SIZE, EVAL_BATCH_SIZE,
    LR, WEIGHT_DECAY, MIN_DELTA, GRAD_CLIP,
)
from .pilot import nw_mean_interval

WINDOWS = (1, 30)
REPRESENTATIONS = ('A', 'B')


@dataclass(frozen=True)
class RawFeature:
    name: str
    chinese: str
    expression: str
    rationale: str


A_FEATURES = (
    RawFeature('A_RET1', '收盘价单日收益率', '$close/Ref($close,1)-1', '每日收盘价格变化'),
    RawFeature('A_GAP1', '隔夜开盘缺口', '$open/Ref($close,1)-1', '隔夜信息反应；收盘后预测时已知'),
    RawFeature('A_INTRADAY', '日内开收收益率', '($close-$open)/$open', '当日价格方向'),
    RawFeature('A_RANGE', '日内相对振幅', '($high-$low)/$open', '当日价格波动区间'),
    RawFeature('A_UPPER_WICK', '上影线占振幅比例', '($high-Greater($open,$close))/($high-$low+1e-12)', '日内高价相对开收盘的位置'),
    RawFeature('A_LOWER_WICK', '下影线占振幅比例', '(Less($open,$close)-$low)/($high-$low+1e-12)', '日内低价相对开收盘的位置'),
    RawFeature('A_VWAP_DEV', 'VWAP相对收盘偏离', '$vwap/$close-1', '成交量加权均价与收盘价偏离'),
    RawFeature('A_VOL_CHANGE', '相邻日成交量对称变化率', '($volume-Ref($volume,1))/($volume+Ref($volume,1)+1e-12)', '当日量相对前日量的变化，取值约[-1,1]'),
)
A_NAMES = tuple(x.name for x in A_FEATURES)


def check_a_spec():
    assert len(A_FEATURES) == 8 == len(set(A_NAMES))
    assert not any('Mean(' in f.expression or 'Std(' in f.expression or 'Corr(' in f.expression for f in A_FEATURES)
    assert not any(',-' in f.expression or ', -' in f.expression for f in A_FEATURES)


def consecutive_window_mask(finite_daily: np.ndarray, anchors: np.ndarray, window: int) -> np.ndarray:
    """Whether ALL daily observations in a requested window are finite.

    Prior V2 eligible30 anchors already ensure chronological and stock continuity.
    This mask only checks the added A features, including preceding context days.
    """
    good = np.asarray(finite_daily, dtype=bool)
    rows = np.asarray(anchors, dtype=np.int64)
    if good.ndim != 1 or rows.ndim != 1 or window < 1:
        raise ValueError('Bad input dimensions/window')
    if len(rows) and (rows.min() < window - 1 or rows.max() >= len(good)):
        raise ValueError('Window anchor out of bounds')
    prefix = np.concatenate(([0], np.cumsum(~good, dtype=np.int64)))
    return prefix[rows+1] == prefix[rows-window+1]


def materialize(features: np.ndarray, rows: np.ndarray, window: int, chunk: int=8192) -> np.ndarray:
    if window not in WINDOWS:
        raise ValueError('Window not predeclared')
    rows = np.asarray(rows, dtype=np.int64)
    n_channels = int(features.shape[1])
    if features.ndim != 2 or n_channels not in (8,16):
        raise ValueError('Unexpected representation channel count')
    if len(rows) and (rows.min() < window-1 or rows.max() >= len(features)):
        raise ValueError('Out-of-range anchor')
    out = np.empty((len(rows), window, n_channels), dtype=np.float32)
    offsets = np.arange(window-1, -1, -1, dtype=np.int64)
    for start in range(0,len(rows),chunk):
        take = rows[start:start+chunk]
        out[start:start+chunk] = np.asarray(features[take[:,None] - offsets[None,:]],dtype=np.float32)
    if not np.isfinite(out).all():
        raise ValueError('Input contains nonfinite values; inspect A/B common mask')
    return out


def source_consistency(a_raw: np.ndarray, b_raw, train_and_valid_rows: np.ndarray,
                       b_names: tuple[str, ...], atol=5e-5) -> dict:
    """Independent Qlib expression audit on overlapping daily features.

    We use A_RET1=B_RET1, A_INTRADAY=B_KMID, A_RANGE=B_KLEN, and
    A_VWAP_DEV=(B_VWAP0 - 1) to guard against alignment or adjustment errors.
    """
    checks={'A_RET1':'RET1','A_INTRADAY':'KMID','A_RANGE':'KLEN','A_VWAP_DEV':'VWAP0'}
    indices=np.asarray(train_and_valid_rows,dtype=np.int64)
    if indices.size == 0:
        raise ValueError('No rows for overlap check')
    output={}
    for a_name,b_name in checks.items():
        ai=A_NAMES.index(a_name)
        target=np.asarray(b_raw[b_name],dtype=np.float64)[indices]
        if a_name=='A_VWAP_DEV':
            target=target-1.0
        candidate=np.asarray(a_raw[indices,ai],dtype=np.float64)
        finite=np.isfinite(target)&np.isfinite(candidate)
        if int(finite.sum())<100:
            raise ValueError(f'Insufficient common finite values to audit {a_name}')
        discrepancy=np.abs(candidate[finite]-target[finite])
        maximum=float(np.max(discrepancy))
        output[a_name]={'compared_rows':int(finite.sum()),'max_abs_diff':maximum}
        if maximum>atol:
            raise ValueError(f'{a_name} fails A/B formula-alignment check: {maximum}')
    return output


def model_for(channels: int):
    from torch import nn
    if channels not in (8,16):
        raise ValueError('Expected 8 or 16 channels')
    class LSTMCompare(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn=nn.LSTM(input_size=channels, hidden_size=HIDDEN, num_layers=1, batch_first=True)
            self.head=nn.Sequential(nn.Linear(HIDDEN,16),nn.ReLU(),nn.Linear(16,1))
        def forward(self,x):
            if x.ndim!=3 or x.shape[2]!=channels:
                raise ValueError(f'Expected [batch,time,{channels}]')
            h,_=self.rnn(x)
            return self.head(h[:,-1,:]).squeeze(-1)
    return LSTMCompare()


def fit_and_predict(x_train:np.ndarray,y_train:np.ndarray,x_valid:np.ndarray,y_valid:np.ndarray,
                    *,seed:int,device:str,checkpoint:Path,max_epochs:int=EPOCHS,
                    patience:int=PATIENCE,batch_size:int=BATCH_SIZE):
    """Same optimizer, patience, checkpoint metric, hidden width, head as V5.

    8/16 input dimensions produce slightly different parameter counts; log that.
    """
    import torch
    from torch import nn
    from torch.nn.utils import clip_grad_norm_
    if seed not in SEEDS or x_train.ndim!=3 or x_valid.ndim!=3:
        raise ValueError('Unregistered seed or bad X dimensions')
    if x_train.shape[1:]!=x_valid.shape[1:]:
        raise ValueError('Train/valid input shapes differ')
    if len(x_train)!=len(y_train) or len(x_valid)!=len(y_valid):
        raise ValueError('Mismatched X/y')
    if not (np.isfinite(x_train).all() and np.isfinite(x_valid).all() and
            np.isfinite(y_train).all() and np.isfinite(y_valid).all()):
        raise ValueError('Nonfinite data')
    if max_epochs<1 or patience<1 or batch_size<1:
        raise ValueError('Invalid trainer arguments')
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.backends.mps.is_available():torch.mps.manual_seed(seed)
    if device=='cpu':torch.set_num_threads(min(4,torch.get_num_threads()))
    model=model_for(x_train.shape[2]).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    mse=nn.MSELoss()
    best=float('inf');best_state=None;best_ep=0;bad=0;trace=[]
    rng=np.random.default_rng(seed)
    for ep in range(1,max_epochs+1):
        model.train(); total=0.0
        perm=rng.permutation(len(x_train))
        for k in range(0,len(perm),batch_size):
            idx=perm[k:k+batch_size]
            xb=torch.from_numpy(np.ascontiguousarray(x_train[idx])).to(device)
            yb=torch.from_numpy(np.ascontiguousarray(y_train[idx])).to(device)
            opt.zero_grad(set_to_none=True)
            loss=mse(model(xb),yb)
            loss.backward();clip_grad_norm_(model.parameters(),GRAD_CLIP)
            opt.step();total+=float(loss.detach().cpu())*len(idx)
        model.eval();vsum=0.0
        with torch.inference_mode():
            for k in range(0,len(x_valid),EVAL_BATCH_SIZE):
                stop=min(k+EVAL_BATCH_SIZE,len(x_valid))
                xb=torch.from_numpy(np.ascontiguousarray(x_valid[k:stop])).to(device)
                yb=torch.from_numpy(np.ascontiguousarray(y_valid[k:stop])).to(device)
                vsum+=float((model(xb)-yb).square().sum().cpu())
        vmse=vsum/len(y_valid);tmse=total/len(y_train)
        trace.append({'epoch':ep,'train_mse_scaled':tmse,'valid_mse_scaled':vmse})
        print(f'   epoch {ep:02d}/{max_epochs}, train MSE={tmse:.5f}, valid MSE={vmse:.5f}',flush=True)
        if np.isfinite(vmse) and vmse<best-MIN_DELTA:
            best=float(vmse);best_ep=ep
            best_state=copy.deepcopy({k:v.detach().cpu() for k,v in model.state_dict().items()})
            bad=0
        else:
            bad+=1
            if bad>=patience:break
    if best_state is None:
        raise RuntimeError('No finite validation checkpoint')
    model.load_state_dict(best_state);model.eval()
    preds=np.empty(len(x_valid),dtype=np.float32)
    with torch.inference_mode():
        for k in range(0,len(x_valid),EVAL_BATCH_SIZE):
            end=min(len(x_valid),k+EVAL_BATCH_SIZE)
            xb=torch.from_numpy(np.ascontiguousarray(x_valid[k:end])).to(device)
            preds[k:end]=model(xb).cpu().numpy()
    if not np.isfinite(preds).all():raise RuntimeError('Nonfinite predictions')
    checkpoint.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'state_dict':best_state,'channels':x_train.shape[2],
                'window':x_train.shape[1],'seed':seed,'best_epoch':best_ep},checkpoint)
    return preds,{'best_epoch':best_ep,'validation_mse_scaled':best,
                  'epochs_run':len(trace),'trace':trace,'device':device,
                  'parameter_count':sum(p.numel() for p in model.parameters())}


def average_daily(dailies:list):
    import pandas as pd
    if len(dailies)!=len(SEEDS):
        raise ValueError('Expected exactly three seed daily results')
    ref=dailies[0].sort_values('date').reset_index(drop=True).copy()
    vectors=[ref.rankic.to_numpy(float)]
    ics=[ref.ic.to_numpy(float)]
    for d in dailies[1:]:
        d=d.sort_values('date').reset_index(drop=True)
        if not np.array_equal(pd.to_datetime(d.date),pd.to_datetime(ref.date)) or not np.array_equal(d.n_stocks,ref.n_stocks):
            raise ValueError('Misaligned model daily cross-sections')
        vectors.append(d.rankic.to_numpy(float));ics.append(d.ic.to_numpy(float))
    ref['rankic']=np.mean(np.stack(vectors),axis=0)
    ref['ic']=np.mean(np.stack(ics),axis=0)
    return ref


def mean_rankic_contrast(a,b):
    import pandas as pd
    a=a.sort_values('date').reset_index(drop=True)
    b=b.sort_values('date').reset_index(drop=True)
    if not np.array_equal(pd.to_datetime(a.date),pd.to_datetime(b.date)) or not np.array_equal(a.n_stocks,b.n_stocks):
        raise ValueError('Paired dates/stock counts differ')
    return nw_mean_interval(a.rankic.to_numpy(float)-b.rankic.to_numpy(float))


def difference_in_differences(a1,a30,b1,b30):
    """Per-day (A30-A1)-(B30-B1) *average daily RankIC* contrasts.

    It is a descriptive contrast of model performance differences, not causal.
    """
    dates=[x.sort_values('date').reset_index(drop=True) for x in (a1,a30,b1,b30)]
    for x in dates[1:]:
        if not np.array_equal(pd_to_dates(x.date),pd_to_dates(dates[0].date)):
            raise ValueError('Contrasts use mismatched dates')
        if not np.array_equal(x.n_stocks.to_numpy(),dates[0].n_stocks.to_numpy()):
            raise ValueError('Contrasts use mismatched cross sections')
    x1,x30,y1,y30=[d.rankic.to_numpy(float) for d in dates]
    return nw_mean_interval((x30-x1)-(y30-y1))


def pd_to_dates(series):
    import pandas as pd
    return pd.to_datetime(series).to_numpy()
