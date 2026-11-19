#!/usr/bin/env python3
"""V4 validation-only experiments, frozen 30-day eligible anchors and labels.

Ridge 1/5/30: same train/valid anchor cohort, alpha=1000, solver='lsqr'.
MLP flat: reuse V3 baseline; run reduced width and stronger AdamW decay controls.
2023-2026 outcomes are NEVER accessed. Run --experiment ridge, then --experiment mlp.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from lstm_research.feature_spec import FEATURE_NAMES
from lstm_research.sequences import index_for
from lstm_research.pilot import (
    SEED,WINDOW,anchor_sha256,fit_train_label_scaler,target_for_train,target_for_valid,
    pred_to_raw,evaluate_validation,save_json,
)
from lstm_research.hypothesis_v4 import (
    RIDGE_WINDOWS,MLP_VARIANTS,flat_window,load_v3_baseline,compare_daily,
    train_mlp_variant,metrics_to_row,verify_label_scaler,
)

V2=ROOT/'output'/'v2_sequences'
V3=ROOT/'output'/'v3_pilot'
OUT=ROOT/'output'/'v4_hypothesis'


def main():
    parser=argparse.ArgumentParser(description='V4 research hypothesis controls (2021-22 validation only)')
    parser.add_argument('--experiment',choices=['ridge','mlp','both'],default='ridge')
    parser.add_argument('--device',choices=['auto','cpu','mps'],default='auto')
    parser.add_argument('--overwrite',action='store_true',help='rerun V4 variants; never overwrites V3 baseline')
    args=parser.parse_args()

    print('[1/5] Read V2 contract and V3 saved baseline; NO 2023-2026 access',flush=True)
    v2=json.loads((V2/'v2_sequence_audit.json').read_text(encoding='utf-8'))
    if v2['features']!=list(FEATURE_NAMES) or v2['default_window']!=30 or v2['n_features']!=16:
        raise ValueError('Frozen V2 16 x 30 input schema mismatch')
    features=np.load(V2/'features_scaled.npy',mmap_mode='r')
    targets=np.load(V2/'labels.npy',mmap_mode='r')
    index=pd.read_pickle(V2/'sample_index.pkl.gz',compression='gzip')
    train=index_for(index,'train',30)
    valid=index_for(index,'valid',30)
    if len(train)!=327555 or len(valid)!=139220:
        raise ValueError('Unexpected V2 common eligible30 cohorts')
    if set(train['split'])!={'train'} or set(valid['split'])!={'valid'}:
        raise ValueError('Unexpected training/validation subset')
    if train.datetime.max()>pd.Timestamp('2020-12-31') or valid.datetime.min()<pd.Timestamp('2021-01-01') or valid.datetime.max()>pd.Timestamp('2022-12-31'):
        raise ValueError('Unexpected split calendar boundary')
    tr=train.panel_row.to_numpy(dtype=np.int64)
    va=valid.panel_row.to_numpy(dtype=np.int64)
    ytr=np.asarray(targets[tr],dtype=np.float64)
    yva=np.asarray(targets[va],dtype=np.float64)
    if not (np.isfinite(ytr).all() and np.isfinite(yva).all()):
        raise ValueError('Invalid label in eligible cohort')
    vhash=anchor_sha256(valid)
    thash=anchor_sha256(train)
    v3_protocol=json.loads((V3/'pilot_protocol.json').read_text(encoding='utf-8'))
    if v3_protocol.get('valid_anchor_sha256') != vhash or v3_protocol.get('train_anchor_sha256') != thash:
        raise ValueError('V3 cohort differs from V2: cannot reuse baselines')
    scaler=fit_train_label_scaler(ytr)
    saved_scaler=json.loads((V3/'label_scaler_train_only.json').read_text(encoding='utf-8'))
    verify_label_scaler(saved_scaler,scaler)
    ytr_scaled=target_for_train(ytr,scaler)
    yva_scaled=target_for_valid(yva,scaler)
    OUT.mkdir(parents=True,exist_ok=True)
    manifest={
        'stage':'V4 two mechanistic hypotheses; validation-only, NOT holdout',
        'train_period':'2016-2020','valid_period':'2021-2022',
        'development_2023_2024':'NOT READ','final_2025_2026':'LOCKED: NOT READ',
        'train_anchor_sha256':thash,'valid_anchor_sha256':vhash,
        'n_train':len(train),'n_valid':len(valid),'seed':SEED,
        'features':list(FEATURE_NAMES),'v2_scaler_sha256':v2['scaler_sha256'],
        'v3_protocol_sha256':hashlib.sha256((V3/'pilot_protocol.json').read_bytes()).hexdigest(),
        'ridge':{'windows':[1,5,30],'alpha':1000.0,'solver':'lsqr',
                 'cohort':'identical eligible30 train/valid anchors; 30d reused from V3'},
        'mlp':{'baseline':'V3 mlp_flat unchanged',
               'variants':MLP_VARIANTS,
               'epochs_max':12,'patience':3,'batch_size':512,'optimizer':'AdamW','lr':0.001,
               'checkpoint':'best validation scaled MSE, not validation RankIC'},
        'primary_metric':'equal-weight mean daily cross-sectional RankIC',
        'uncertainty':'paired date differences, Newey-West 10 lags; descriptive',
        'limitations':[
            'Ridge penalty alpha fixed while dimension changes; effective regularization not identical across windows.',
            'A latest-day rolling factor contains historical price/volume data; 1d is NOT history-free.',
            'Validation MSE used for MLP checkpoint choice; validation RankIC is model selection evidence, not final independent estimate.',
            'One seed, two predeclared MLP modifications: this is a diagnostic, not hyperparameter proof.',
            'No development or final_holdout outcome is touched.',
        ],
        'environment':{'python':sys.version.split()[0],'arch':platform.machine()},
    }
    save_json(OUT/'v4_protocol.json',manifest)
    models=['ridge_1','ridge_5','ridge_30'] if args.experiment=='ridge' else (
        ['mlp_flat_baseline','mlp_flat_small','mlp_flat_strong_wd'] if args.experiment=='mlp'
        else ['ridge_1','ridge_5','ridge_30','mlp_flat_baseline','mlp_flat_small','mlp_flat_strong_wd'])
    print(f'  train={len(train):,}; valid={len(valid):,}; experiment={args.experiment}',flush=True)
    print('[2/5] Check existing results and train ONLY new variants',flush=True)
    for model in models:
        metric_path=OUT/f'{model}_valid_metrics.json'
        daily_path=OUT/f'{model}_daily_rankic.csv'
        if model in ('ridge_30','mlp_flat_baseline'):
            old_name='ridge' if model=='ridge_30' else 'mlp_flat'
            stats,daily=load_v3_baseline(V3,old_name,vhash,len(valid))
            stats=stats.copy()
            stats['v4_source']='original_v3_baseline_reused_without_retraining'
            save_json(metric_path,stats)
            daily.to_csv(daily_path,index=False,encoding='utf-8-sig')
            print('   '+model+': copied V3 original validation evidence; no retraining',flush=True)
            continue
        if metric_path.exists() and daily_path.exists() and not args.overwrite:
            old=json.loads(metric_path.read_text(encoding='utf-8'))
            if old.get('valid_anchor_sha256')!=vhash or old.get('v4_variant')!=model or old.get('seed')!=SEED:
                raise RuntimeError('Stale V4 result does not match protocol; inspect before overwrite')
            print('   '+model+': completed, skip',flush=True)
            continue
        print('   Training '+model+' ...',flush=True)
        if model.startswith('ridge_'):
            from sklearn.linear_model import Ridge
            import joblib
            width=int(model.split('_')[-1])
            xtr=flat_window(features,tr,width)
            xva=flat_window(features,va,width)
            reg=Ridge(alpha=1000.0,solver='lsqr',fit_intercept=True,tol=1e-3)
            reg.fit(xtr,ytr_scaled)
            pred_scaled=reg.predict(xva)
            joblib.dump(reg,OUT/f'{model}_model.joblib')
            details={'window':width,'n_features_flat':16*width,'alpha':1000.0,'solver':'lsqr'}
            del xtr,xva,reg
        else:
            pred_scaled,details=train_mlp_variant(
                model,features,tr,ytr_scaled,va,yva_scaled,args.device,OUT/f'{model}_model.pt')
        pred_raw=pred_to_raw(pred_scaled,scaler)
        stats,daily=evaluate_validation(valid.datetime.to_numpy(),yva,pred_raw)
        stats.update({'model':model,'v4_variant':model,'seed':SEED,'valid_anchor_sha256':vhash,
                      'train_anchor_sha256':thash,'validation_period':'2021-2022',
                      'details':details})
        save_json(metric_path,stats)
        daily.to_csv(daily_path,index=False,encoding='utf-8-sig')
        np.save(OUT/f'{model}_valid_predictions.npy',np.asarray(pred_raw,dtype=np.float32))
        print(f'     {model}: RankIC={stats["rankic_daily_equal_weight"]["mean"]:+.6f}, IC={stats["ic_daily_equal_weight"]["mean"]:+.6f}',flush=True)

    print('[3/5] Summarizing exact-cohort validation results...',flush=True)
    all_models=['ridge_1','ridge_5','ridge_30','mlp_flat_baseline','mlp_flat_small','mlp_flat_strong_wd']
    rows=[]
    daily_map={}
    for model in all_models:
        mfile=OUT/f'{model}_valid_metrics.json'
        dfile=OUT/f'{model}_daily_rankic.csv'
        if not (mfile.is_file() and dfile.is_file()):
            continue
        stats=json.loads(mfile.read_text(encoding='utf-8'))
        if stats.get('valid_anchor_sha256')!=vhash or stats.get('n_samples')!=len(valid):
            raise ValueError('Existing V4 metrics differ from current shared validation sample')
        rows.append(metrics_to_row(model,stats))
        daily_map[model]=pd.read_csv(dfile,parse_dates=['date']).sort_values('date').reset_index(drop=True)
    pd.DataFrame(rows).to_csv(OUT/'v4_model_summary.csv',index=False,encoding='utf-8-sig')
    comparisons={}
    for a,b in [('ridge_5','ridge_1'),('ridge_30','ridge_1'),('ridge_30','ridge_5'),
                ('mlp_flat_small','mlp_flat_baseline'),('mlp_flat_strong_wd','mlp_flat_baseline')]:
        if a in daily_map and b in daily_map:
            comparisons[f'{a}_minus_{b}']=compare_daily(daily_map[a],daily_map[b])
    save_json(OUT/'v4_paired_rankic.json',comparisons)
    print('[4/5] Diagnostic contrast results (mean RankIC differences):',flush=True)
    for name,c in comparisons.items():
        print(f'   {name}: {c["mean"]:+.6f}; HAC95={c["ci95"]}',flush=True)
    print('[5/5] Finished. Development/final outcomes never read.',flush=True)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
    print('V4_HYPOTHESIS_EXPERIMENT_COMPLETE:',OUT,flush=True)


if __name__=='__main__':
    main()
