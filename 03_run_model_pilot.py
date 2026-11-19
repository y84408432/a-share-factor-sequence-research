#!/usr/bin/env python3
"""V3 pilot: baseline + LSTM, fixed 30d anchors, 2016-20 train / 2021-22 valid.

Absolutely no 2023-24 development or 2025-26 final_holdout outcome access.
No tuning on later periods. Predictions are persisted only for VALID split.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
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
    MODELS,WINDOW,SEED,anchor_sha256,stack_sequences,
    fit_train_label_scaler,target_for_train,target_for_valid,pred_to_raw,
    evaluate_validation, paired_rankic,save_json,fit_torch_model,
)

V2=ROOT/'output'/'v2_sequences'
OUT=ROOT/'output'/'v3_pilot'


def check_source_contract():
    meta=json.loads((V2/'v2_sequence_audit.json').read_text(encoding='utf-8'))
    if meta['features'] != list(FEATURE_NAMES) or meta['default_window']!=WINDOW:
        raise ValueError('V2 features/window do not match frozen 16x30 protocol')
    if meta['n_features']!=16 or meta['n_panel_rows']!=861570:
        raise ValueError('Unexpected V2 factor panel size')
    return meta


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--models',default=','.join(MODELS),help='comma separated: '+','.join(MODELS))
    p.add_argument('--device',default='auto',choices=['auto','cpu','mps'])
    p.add_argument('--overwrite',action='store_true',help='retrain existing models; old results get replaced')
    args=p.parse_args()
    requested=[m.strip() for m in args.models.split(',') if m.strip()]
    if not requested or len(requested)!=len(set(requested)) or any(m not in MODELS for m in requested):
        p.error('Invalid model selection: '+str(requested))

    print('[1/5] Load V2 16-factor / 30d model-comparison contract...',flush=True)
    v2=check_source_contract()
    feat=np.load(V2/'features_scaled.npy',mmap_mode='r')
    all_labels=np.load(V2/'labels.npy',mmap_mode='r')
    index=pd.read_pickle(V2/'sample_index.pkl.gz',compression='gzip')
    train=index_for(index,'train',WINDOW)
    valid=index_for(index,'valid',WINDOW)
    expected={x['split']:x['eligible_30'] for x in v2['counts']}
    if len(train)!=expected['train'] or len(valid)!=expected['valid']:
        raise ValueError('Eligible sample counts differ from audited V2')
    if set(train['split'])!={'train'} or set(valid['split'])!={'valid'}:
        raise ValueError('Unexpected subset in pilot')
    if valid['datetime'].min() < pd.Timestamp('2021-01-01') or valid['datetime'].max() > pd.Timestamp('2022-12-31'):
        raise ValueError('Unexpected validation period')
    train_rows=train.panel_row.to_numpy(dtype=np.int64)
    valid_rows=valid.panel_row.to_numpy(dtype=np.int64)
    train_y_raw=np.asarray(all_labels[train_rows],dtype=np.float64)
    valid_y_raw=np.asarray(all_labels[valid_rows],dtype=np.float64)
    if not np.isfinite(train_y_raw).all() or not np.isfinite(valid_y_raw).all():
        raise ValueError('Nonfinite target on eligible anchor')
    dates=valid.datetime.to_numpy()
    label_scaler=fit_train_label_scaler(train_y_raw)
    tr_y=target_for_train(train_y_raw,label_scaler)
    va_y=target_for_valid(valid_y_raw,label_scaler)
    print(f'  samples: train={len(train):,}, valid={len(valid):,}; VALID only, no development/holdout evaluation',flush=True)
    print('  valid anchors sha256:',anchor_sha256(valid),flush=True)

    OUT.mkdir(parents=True,exist_ok=True)
    save_json(OUT/'label_scaler_train_only.json',label_scaler)
    manifest={'stage':'V3 validation-only PILOT, not final research performance',
              'seed':SEED,'window':WINDOW,'features':list(FEATURE_NAMES),
              'preprocessing_sha256':v2['scaler_sha256'],
              'train_anchor_sha256':anchor_sha256(train),
              'valid_anchor_sha256':anchor_sha256(valid),
              'n_train':len(train),'n_valid':len(valid),
              'train_period':'2016-2020','valid_period':'2021-2022',
              'development_2023_2024':'not accessed','final_2025_2026':'LOCKED: not accessed',
              'target_train_only':label_scaler,'metric':'mean per-date cross-sectional RankIC',
              'metrics_nw_lags':10,'min_stocks_per_date':30,
              'model_selection':'validation MSE checkpoint/early stopping; RankIC reported but not used for checkpoint',
              'hardware':platform.machine(),'python':sys.version.split()[0],
              'model_configs':{
                  'ridge':{'alpha':1000.0,'solver':'lsqr','input':'flatten_30x16'},
                  'mlp_last':{'hidden':[32,16],'epochs_max':12,'patience':3,'lr':1e-3,'weight_decay':1e-3,'input':'last_16'},
                  'mlp_flat':{'hidden':[64,16],'epochs_max':12,'patience':3,'lr':1e-3,'weight_decay':1e-3,'input':'flatten_30x16'},
                  'lightgbm':{'n_estimators':400,'learning_rate':0.05,'num_leaves':15,
                              'min_child_samples':200,'reg_lambda':10.0,'max_bin':63,
                              'early_stopping_rounds':30,'input':'flatten_30x16'},
                  'lstm':{'hidden_size':32,'num_layers':1,'epochs_max':12,'patience':3,'lr':1e-3,
                          'weight_decay':1e-3,'input':'ordered_30x16'},
              },
              'limitations':['One seed and fixed configurations only: screening, not robust model superiority.',
                             'Labels from adjusted Qlib opens; market-price executable backtest not yet audited.',
                             'V3 does not establish PIT safety of historical constituents or adjusted price data.',
                             'Valid MSE determines neural checkpoints/LightGBM round; validation RankIC is not independent.']}
    save_json(OUT/'pilot_protocol.json',manifest)
    print('[2/5] Frozen train-only label scaling; preparing shared anchors...',flush=True)

    # Reuse outputs when rerun without --overwrite; never accidentally mix cohorts.
    for k,model in enumerate(requested,1):
        metric_path=OUT/f'{model}_valid_metrics.json'
        daily_path=OUT/f'{model}_daily_rankic.csv'
        if metric_path.exists() and not args.overwrite:
            saved=json.loads(metric_path.read_text(encoding='utf-8'))
            if saved.get('valid_anchor_sha256')!=anchor_sha256(valid):
                raise RuntimeError('Old result has different anchors. Investigate rather than silently overwrite.')
            print(f'[3/5] [{k}/{len(requested)}] {model}: already finished, skipping (use --overwrite to rerun)',flush=True)
            continue
        print(f'[3/5] [{k}/{len(requested)}] Fitting {model}...',flush=True)
        extras={}
        if model=='mlp_last':
            pred_scaled,extras=fit_torch_model(model,feat,train_rows,tr_y,valid_rows,va_y,
                {'features':FEATURE_NAMES},OUT/f'{model}_model.pt',args.device)
        elif model in ('mlp_flat','lstm'):
            pred_scaled,extras=fit_torch_model(model,feat,train_rows,tr_y,valid_rows,va_y,
                {'features':FEATURE_NAMES},OUT/f'{model}_model.pt',args.device)
        elif model in ('ridge','lightgbm'):
            print('    constructing identical flat 30x16 panels for training and validation',flush=True)
            xtr=stack_sequences(feat,train_rows,flatten=True)
            xva=stack_sequences(feat,valid_rows,flatten=True)
            if model=='ridge':
                import joblib
                from sklearn.linear_model import Ridge
                reg=Ridge(alpha=1000.0,solver='lsqr',fit_intercept=True,tol=1e-3)
                reg.fit(xtr,tr_y)
                pred_scaled=reg.predict(xva)
                joblib.dump(reg,OUT/'ridge_model.joblib')
                extras={'effective_features':480,'alpha':1000,'solver':'lsqr'}
            else:
                import lightgbm as lgb
                reg=lgb.LGBMRegressor(n_estimators=400,learning_rate=.05,num_leaves=15,
                        min_child_samples=200,reg_lambda=10.,max_bin=63,
                        random_state=SEED,n_jobs=8,verbosity=-1)
                reg.fit(xtr,tr_y,eval_set=[(xva,va_y)],eval_metric='l2',
                        callbacks=[lgb.early_stopping(30,verbose=False),lgb.log_evaluation(0)])
                pred_scaled=reg.predict(xva,num_iteration=reg.best_iteration_)
                reg.booster_.save_model(str(OUT/'lightgbm_model.txt'))
                extras={'best_iteration_by_valid_mse':int(reg.best_iteration_),
                        'early_stopping_metric':'valid_l2_scaled',
                        'effective_features':480}
            del xtr,xva,reg
            gc.collect()
        else:
            raise RuntimeError('Unhandled model')
        pred_raw=pred_to_raw(pred_scaled,label_scaler)
        metrics,daily=evaluate_validation(dates,valid_y_raw,pred_raw)
        metrics.update({'model':model,'window':WINDOW,'seed':SEED,
                        'valid_anchor_sha256':anchor_sha256(valid),
                        'validation_period':'2021-2022','checkpoint_extra':extras})
        save_json(metric_path,metrics)
        daily.to_csv(daily_path,index=False,encoding='utf-8-sig')
        print(f'    VALID daily RankIC={metrics["rankic_daily_equal_weight"]["mean"]:+.6f}',flush=True)
        print(f'    VALID IC={metrics["ic_daily_equal_weight"]["mean"]:+.6f}',flush=True)

    print('[4/5] Summarizing completed VALID-only results...',flush=True)
    summaries=[]
    dailies={}
    for model in MODELS:
        fp=OUT/f'{model}_valid_metrics.json'
        daily_fp=OUT/f'{model}_daily_rankic.csv'
        if fp.exists() and daily_fp.exists():
            stats=json.loads(fp.read_text(encoding='utf-8'))
            if stats['valid_anchor_sha256']!=anchor_sha256(valid):
                raise ValueError(f'Result cohort mismatch in {model}')
            r=stats['rankic_daily_equal_weight']
            summaries.append({'model':model,'n_valid':stats['n_samples'],'n_days':r['n_days'],
                              'rankic':r['mean'],'rankic_hac_lo':r['ci95'][0] if r['ci95'] else None,
                              'rankic_hac_hi':r['ci95'][1] if r['ci95'] else None,
                              'ic':stats['ic_daily_equal_weight']['mean'],
                              'mse_raw':stats['mse_raw']})
            dailies[model]=pd.read_csv(daily_fp,parse_dates=['date'])
    pd.DataFrame(summaries).to_csv(OUT/'model_valid_summary.csv',index=False,encoding='utf-8-sig')
    paired={}
    if 'lstm' in dailies:
        for baseline in ('mlp_flat','lightgbm','ridge','mlp_last'):
            if baseline in dailies:
                paired[f'lstm_minus_{baseline}']=paired_rankic(dailies['lstm'],dailies[baseline])
    save_json(OUT/'paired_valid_rankic.json',paired)
    print('[5/5] Pilot complete, no development/holdout outcomes inspected.',flush=True)
    print(pd.DataFrame(summaries).to_string(index=False),flush=True)
    print('V3_VALID_PILOT_COMPLETE:',OUT,flush=True)
    print('Share: model_valid_summary.csv, paired_valid_rankic.json, pilot_protocol.json, and run_v3.log',flush=True)

if __name__=='__main__': main()
