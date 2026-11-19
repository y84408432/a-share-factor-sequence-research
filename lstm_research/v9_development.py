"""V9 frozen development-period helpers. Imports do not read holdout labels."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

from .pilot import anchor_sha256, evaluate_validation, paired_rankic, nw_mean_interval
from .representation_v7 import A_FEATURES, A_NAMES, consecutive_window_mask, model_for
from .sequences import scale_feature_column

MODEL_IDS=('B1_Ridge','A30_Ridge','A30_LSTM','B30_LSTM')
SEEDS=(20261115,20261116,20261117)


def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for part in iter(lambda:f.read(2**20),b''):
            h.update(part)
    return h.hexdigest()


def model_registry(root:Path):
    root=Path(root)
    paths={
        'B1_Ridge':root/'output/v4_hypothesis/ridge_1_model.joblib',
        'A30_Ridge':root/'output/v8_a30_controls/checkpoints/ridge_A30_flat.pt',
        'A30_LSTM':[(root/f'output/v7_representations/checkpoints/a_w30_s{s}.pt') for s in SEEDS],
        'B30_LSTM':[(root/f'output/v7_representations/checkpoints/b_w30_s{s}.pt') for s in SEEDS],
    }
    flat={key:([str(p) for p in value] if isinstance(value,list) else [str(value)]) for key,value in paths.items()}
    for key,group in flat.items():
        for item in group:
            if not Path(item).is_file():
                raise FileNotFoundError(f'{key} missing frozen checkpoint: {item}')
    return {k:[{'path':str(Path(x).relative_to(root)),'sha256':file_hash(x)} for x in v] for k,v in flat.items()}


def raw_a_to_scaled(raw, scaler):
    raw=np.asarray(raw,dtype=np.float32)
    if raw.ndim!=2 or raw.shape[1]!=8 or scaler.get('features')!=list(A_NAMES):
        raise ValueError('A8 input/scaler specification changed')
    if scaler.get('fit_split')!='train':
        raise ValueError('A scaler not from training data')
    scaled=np.zeros(raw.shape,dtype=np.float32)
    for j,name in enumerate(A_NAMES):
        scaled[:,j]=scale_feature_column(raw[:,j],scaler['parameters'][name])
    return scaled


def audit_overlap(raw_a,b_panel,positions,feature_names,atol=5e-5):
    """Independently verify Qlib A expressions versus B frozen formulas (no labels)."""
    checks={'A_RET1':'RET1','A_INTRADAY':'KMID',
            'A_RANGE':'KLEN','A_VWAP_DEV':'VWAP0'}
    pos=np.asarray(positions,dtype=np.int64)
    if len(pos)<100: raise ValueError('Insufficient overlap rows')
    report={}
    for an,bn in checks.items():
        a=np.asarray(raw_a[pos,A_NAMES.index(an)],dtype=np.float64)
        b=np.asarray(b_panel[bn],dtype=np.float64)[pos]
        if an=='A_VWAP_DEV':b=b-1
        good=np.isfinite(a)&np.isfinite(b)
        if int(good.sum())<100:raise ValueError(f'Insufficient comparable {an}')
        err=float(np.max(np.abs(a[good]-b[good])))
        if err>atol:raise ValueError(f'A/B formula mismatch {an}: max error={err}')
        report[an]={'compared':int(good.sum()),'max_abs_diff':err}
    return report


def check_period(index):
    if index.empty or not (index.split=='development').all():
        raise ValueError('Evaluation sample not wholly development')
    times=pd.to_datetime(index.datetime)
    ends=pd.to_datetime(index.label_end_date)
    if times.min()<pd.Timestamp('2023-01-01') or times.max()>pd.Timestamp('2024-12-31'):
        raise ValueError('Development signal date out of bounds')
    if ends.isna().any() or (ends>pd.Timestamp('2024-12-31')).any() or (ends<=times).any():
        raise ValueError('Label end boundary / purge failure')
    if len(index)<100000 or index.datetime.nunique()<400:
        raise ValueError('Unexpectedly small development sample')


def cohort_by_a(eligible,index_feature_finite):
    rows=eligible.panel_row.to_numpy(np.int64)
    ok=consecutive_window_mask(index_feature_finite,rows,30)
    return eligible.loc[ok].reset_index(drop=True)


def make_predictions(model, features, rows, *,kind,window=30,device='cpu',batch=2048):
    """Exact existing checkpoint inference; never call fit or optimizer."""
    rows=np.asarray(rows,dtype=np.int64)
    if len(rows)==0:raise ValueError('No cohort')
    if kind=='ridge':
        count=int(features.shape[1])*window
        if int(model.n_features_in_)!=count:
            raise ValueError(f'Ridge expected {model.n_features_in_} inputs, got {count}')
    elif kind=='lstm':
        import torch
        model=model.to(device).eval()
    else:raise ValueError(kind)
    out=np.empty(len(rows),dtype=np.float32)
    offsets=np.arange(window-1,-1,-1,dtype=np.int64)
    for i in range(0,len(rows),batch):
        rr=rows[i:i+batch]
        x=np.asarray(features[rr[:,None]-offsets[None,:]],dtype=np.float32)
        if not np.isfinite(x).all():raise ValueError('Nonfinite frozen feature in inference window')
        if kind=='ridge':
            pp=model.predict(x.reshape(len(rr),-1))
        else:
            with torch.inference_mode():
                pp=model(torch.from_numpy(np.ascontiguousarray(x)).to(device)).cpu().numpy()
        out[i:i+len(rr)]=np.asarray(pp,dtype=np.float32).reshape(-1)
    if not np.isfinite(out).all():raise ValueError('Nonfinite model predictions')
    return out


def check_lstm_checkpoint(obj,channels,seed):
    if (obj.get('channels')!=channels or obj.get('window')!=30 or obj.get('seed')!=seed
            or not isinstance(obj.get('state_dict'),dict)):
        raise ValueError('Saved V7 LSTM checkpoint metadata mismatch')


def evaluate_group(dates,actual,preds):
    """Return per-seed metrics AND mean of daily cross-sectional metrics.

    This is not the same as computing IC on a mean-prediction ensemble.
    """
    all_metrics=[]; daily=[]
    for pred in preds:
        m,d=evaluate_validation(dates,actual,pred)
        all_metrics.append(m);daily.append(d)
    ref=daily[0][['date','n_stocks']].reset_index(drop=True)
    avg=ref.copy()
    for col in ('rankic','ic'):
        values=np.stack([d[col].to_numpy() for d in daily])
        if any(not d[['date','n_stocks']].reset_index(drop=True).equals(ref) for d in daily):
            raise ValueError('Seed-specific evaluation date or stock mismatch')
        avg[col]=values.mean(axis=0)
    return all_metrics, avg


def summarize_group(name,dates,actual,preds,seed_labels):
    met,avg=evaluate_group(dates,actual,preds)
    ranks=np.asarray([m['rankic_daily_equal_weight']['mean'] for m in met])
    ic=np.asarray([m['ic_daily_equal_weight']['mean'] for m in met])
    mse=np.asarray([m['mse_raw'] for m in met])
    report={'model':name,'n_seeds':len(preds),'n_samples':len(actual),'n_days':len(avg),
            'rankic_seed_mean':float(ranks.mean()),
            'rankic_seed_sd':float(ranks.std(ddof=1)) if len(ranks)>1 else None,
            'rankic_hac_low':nw_mean_interval(avg.rankic.to_numpy())['ci95'][0],
            'rankic_hac_high':nw_mean_interval(avg.rankic.to_numpy())['ci95'][1],
            'ic_seed_mean':float(ic.mean()),'mse_raw_seed_mean':float(mse.mean()),
            'rankic_2023':float(avg.loc[avg.date.dt.year==2023,'rankic'].mean()),
            'rankic_2024':float(avg.loc[avg.date.dt.year==2024,'rankic'].mean())}
    per_seed=[{'model':name,'seed':str(seed),'rankic':float(m['rankic_daily_equal_weight']['mean']),
               'ic':float(m['ic_daily_equal_weight']['mean']),'mse_raw':float(m['mse_raw'])}
              for seed,m in zip(seed_labels,met)]
    return report,per_seed,avg


def compare_daily(a,b):
    aa=a[['date','n_stocks','rankic']].reset_index(drop=True)
    bb=b[['date','n_stocks','rankic']].reset_index(drop=True)
    if not aa[['date','n_stocks']].equals(bb[['date','n_stocks']]):
        raise ValueError('Different evaluation dates/cross section sizes')
    return paired_rankic(aa,bb)
