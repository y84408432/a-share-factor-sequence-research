#!/usr/bin/env python3
"""V7 controlled representation comparison: 2 (A/B) × 2 (1d/30d) × 3 seeds.

A = 8 lightly engineered daily OHLCV features; B = frozen 16 rolling factors.
Same V2-eligible30 anchor cohort INTERSECTED with A-eligible30, same target,
optimizer, 2021–2022 validation period. Never uses 2023–2026 target values.
This is an exploration of feature representations, not a clean causal test of
information redundancy (A and B feature information sets and input widths differ).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from lstm_research.feature_spec import FEATURE_NAMES, LABEL_EXPR
from lstm_research.sequences import (
    fit_train_only_scaler,scale_feature_column,index_for,
)
from lstm_research.pilot import (
    anchor_sha256,evaluate_validation,fit_train_label_scaler,
    target_for_train,target_for_valid,pred_to_raw,save_json,
)
from lstm_research.lstm_window_v5 import (
    SEEDS,HIDDEN,EPOCHS,PATIENCE,BATCH_SIZE,EVAL_BATCH_SIZE,
    LR,WEIGHT_DECAY,MIN_DELTA,GRAD_CLIP,torch_device,
)
from lstm_research.representation_v7 import (
    A_FEATURES,A_NAMES,WINDOWS,REPRESENTATIONS,check_a_spec,
    consecutive_window_mask,source_consistency,materialize,
    fit_and_predict,average_daily,mean_rankic_contrast,difference_in_differences,
)

V1=ROOT/'output'/'v1_data'
V2=ROOT/'output'/'v2_sequences'
V3=ROOT/'output'/'v3_pilot'
OUT=ROOT/'output'/'v7_representations'
RAW_PANEL=V1/'frozen_factor_panel.pkl.gz'
DATA_FILES=('a_features_scaled.npy','common_train_rows.npy','common_valid_rows.npy',
            'v7_a_scaler_train_only.json','v7_data_audit.json')


def canonical_sha(obj):
    return hashlib.sha256(json.dumps(obj,ensure_ascii=False,sort_keys=True,
            separators=(',',':')).encode('utf-8')).hexdigest()


def load_sources():
    v2=json.loads((V2/'v2_sequence_audit.json').read_text(encoding='utf-8'))
    v3=json.loads((V3/'pilot_protocol.json').read_text(encoding='utf-8'))
    if v2.get('features')!=list(FEATURE_NAMES) or v2.get('n_features')!=16:
        raise ValueError('V2 frozen B feature set changed')
    if v2.get('n_panel_rows')!=861570 or v3.get('features')!=list(FEATURE_NAMES):
        raise ValueError('Source panel/protocol changed from audited V1–V3')
    if v3.get('train_anchor_sha256') is None or v3.get('valid_anchor_sha256') is None:
        raise ValueError('V3 cohort fingerprint missing')
    b_scaled=np.load(V2/'features_scaled.npy',mmap_mode='r')
    labels=np.load(V2/'labels.npy',mmap_mode='r')
    idx=pd.read_pickle(V2/'sample_index.pkl.gz',compression='gzip')
    base_train=index_for(idx,'train',30)
    base_valid=index_for(idx,'valid',30)
    if len(base_train)!=327555 or len(base_valid)!=139220:
        raise ValueError('V2 eligible-30 cohort size changed')
    if anchor_sha256(base_train)!=v3['train_anchor_sha256'] or anchor_sha256(base_valid)!=v3['valid_anchor_sha256']:
        raise ValueError('V2/V3 anchor fingerprints differ')
    if b_scaled.shape!=(861570,16) or labels.shape!=(861570,):
        raise ValueError('V2 matrix size changed')
    return {'v2':v2,'v3':v3,'b':b_scaled,'labels':labels,
            'index':idx,'train':base_train,'valid':base_valid}


def build_a_data(src):
    check_a_spec()
    print('[1/7] Constructing A daily price-volume expressions using Qlib (through 2022 ONLY)...',flush=True)
    from qlib.constant import REG_CN
    from qlib.data import D
    import qlib
    source=Path.home()/'.qlib'/'qlib_data'/'cn_data'
    if not source.exists():raise FileNotFoundError(source)
    qlib.init(provider_uri=str(source),region=REG_CN)
    expressions=[f.expression for f in A_FEATURES]
    a_df=D.features(D.instruments(market='csi300'),expressions,
                    start_time='2015-01-05',end_time='2022-12-31',freq='day',disk_cache=0)
    if a_df.shape[1]!=len(A_NAMES):
        raise RuntimeError('Qlib A feature count mismatch')
    a_df.columns=list(A_NAMES)
    a_df=a_df.reset_index()
    if not {'instrument','datetime'}.issubset(a_df.columns):
        raise RuntimeError('Qlib must return instrument/datetime index')
    a_df['instrument']=a_df['instrument'].astype(str)
    a_df['datetime']=pd.to_datetime(a_df['datetime'])
    if a_df[['instrument','datetime']].duplicated().any():
        raise RuntimeError('Qlib A feature returned duplicate stock-date rows')
    a_df=a_df.set_index(['instrument','datetime']).sort_index()
    print(f'  Qlib A features={len(A_NAMES)}, market rows={len(a_df):,}',flush=True)

    # V1 panel supplies ONLY stock/date alignment and existing feature definitions.
    # The historical labels from 2023–2026 are never used or inspected.
    if not RAW_PANEL.is_file():raise FileNotFoundError(RAW_PANEL)
    panel=pd.read_pickle(RAW_PANEL,compression='gzip')
    if len(panel)!=src['v2']['n_panel_rows']:
        raise ValueError('V1 panel row count changed')
    if panel[['instrument','datetime']].duplicated().any():
        raise ValueError('V1 panel has duplicate stock-date rows')
    panel['datetime']=pd.to_datetime(panel['datetime'])
    base_mask=(panel.datetime<=pd.Timestamp('2022-12-31')).to_numpy()
    base_pos=np.flatnonzero(base_mask)
    target=pd.MultiIndex.from_frame(panel.iloc[base_pos][['instrument','datetime']])
    aligned=a_df.reindex(target)
    a_raw=np.full((len(panel),len(A_NAMES)),np.nan,dtype=np.float32)
    a_raw[base_pos]=aligned.to_numpy(dtype=np.float32)
    # Record coverage before sample masking; avoid silent wrong market/index join.
    joined_rows=int(np.isfinite(a_raw[base_pos]).any(axis=1).sum())
    if joined_rows<int(0.95*len(base_pos)):
        raise RuntimeError(f'A/B alignment unexpectedly low ({joined_rows}/{len(base_pos)})')
    same=source_consistency(a_raw,panel,
            np.concatenate([src['train'].panel_row.to_numpy(dtype=np.int64),
                            src['valid'].panel_row.to_numpy(dtype=np.int64)]),
            FEATURE_NAMES)
    print('  Independent A/B overlap checks passed:',same,flush=True)

    print('[2/7] Build COMMON, strictly finite 30-day cohorts for both A and B...',flush=True)
    a_finite=np.isfinite(a_raw).all(axis=1)
    shared={}
    base_counts={}
    for split in ('train','valid'):
        base=src[split]
        rows=base.panel_row.to_numpy(dtype=np.int64)
        mask=consecutive_window_mask(a_finite,rows,30)
        common=base.loc[mask].reset_index(drop=True)
        if (len(common)<100000 if split=='train' else len(common)<50000):
            raise RuntimeError(f'A/B common {split} cohort unexpectedly small: {len(common)}')
        shared[split]=common
        base_counts[split]={'b_eligible30':int(len(base)), 'a_and_b_common':int(len(common)),
                            'excluded_for_a_missing':int(len(base)-len(common)),
                            'different_dates':int(common.datetime.nunique())}
        print(f'  {split}: B={len(base):,}; A∩B={len(common):,}; lost={len(base)-len(common):,}',flush=True)
    for name,sub in shared.items():
        if (sub.datetime>pd.Timestamp('2022-12-31')).any():
            raise RuntimeError('Later-period anchors forbidden')
        if set(sub.split)!={name}:
            raise RuntimeError('Split mismatch')
    print('[3/7] A-only robust scaler fitted on TRAIN 2016–2020 (V2 rules)...',flush=True)
    scaler_df=pd.DataFrame({'split':panel['split']})
    for i,name in enumerate(A_NAMES):
        scaler_df[name]=a_raw[:,i]
    a_scaler=fit_train_only_scaler(scaler_df,features=A_NAMES)
    if a_scaler.get('fit_split')!='train':
        raise RuntimeError('Scaler fitted outside train')
    # Use the original B train-only scaler from V2 without refitting.
    b_scaler=json.loads((V2/'scaler_train_only.json').read_text(encoding='utf-8'))
    if b_scaler.get('fit_split')!='train':
        raise ValueError('V2 B scaler was not trained on train only')
    a_scaled=np.zeros_like(a_raw,dtype=np.float32)
    for j,name in enumerate(A_NAMES):
        a_scaled[:,j]=scale_feature_column(a_raw[:,j],a_scaler['parameters'][name])
    for split,sub in shared.items():
        # Full-window recheck after scale and cohort construction.
        ends=sub.panel_row.to_numpy(dtype=np.int64)
        for block_start in range(0,len(ends),8192):
            xx=materialize(a_scaled,ends[block_start:block_start+8192],30)
            if not np.isfinite(xx).all():raise RuntimeError('A output has nonfinite eligible window')

    OUT.mkdir(parents=True,exist_ok=True)
    # Fail on partially built caches from a previous unsuccessful run.
    if any((OUT/name).exists() for name in DATA_FILES):
        raise RuntimeError('V7 data cache already partially exists; inspect it rather than overwriting')
    np.save(OUT/'a_features_scaled.npy',a_scaled)
    for name,cohort in shared.items():
        np.save(OUT/f'common_{name}_rows.npy',cohort.panel_row.to_numpy(dtype=np.int64))
    save_json(OUT/'v7_a_scaler_train_only.json',a_scaler)
    audit={
        'stage':'V7 A-feature extraction and shared cohort, VALID-only; no new model performance examined',
        'A_features':[vars(f) for f in A_FEATURES],
        'B_features':list(FEATURE_NAMES),
        'A_shape':list(a_scaled.shape),'B_shape':list(src['b'].shape),
        'A_days_last':'2022-12-31','A_joined_market_rows':joined_rows,
        'A_B_duplicate_formula_checks':same,
        'common_cohort':base_counts,
        'train_anchor_sha256':anchor_sha256(shared['train']),
        'valid_anchor_sha256':anchor_sha256(shared['valid']),
        'b_source_train_only_scaler_sha256':src['v2']['scaler_sha256'],
        'a_scaler_sha256':canonical_sha(a_scaler),
        'A_nonfinite_daily_rows_before_2023':int((~a_finite & base_mask).sum()),
        'limitations':['Inputs A and B differ in information set and channel count; differences are descriptive, not causal evidence of rolling factor redundancy.',
                       'Historical CSI300 membership, adjusted prices and tradability are not fully validated.',
                       'The V1 pickle contains labels from later periods, but no labels beyond 2022 are indexed/used for this experiment.']
    }
    save_json(OUT/'v7_data_audit.json',audit)
    print('  V7 data cache and audit saved; no 2023–2026 label accessed',flush=True)
    del panel,scaler_df,a_raw,a_scaled
    return audit


def ensure_a_cache(src):
    present=[(OUT/n).is_file() for n in DATA_FILES]
    if not any(present):
        return build_a_data(src)
    if not all(present):
        raise RuntimeError('Partial V7 cache: inspect/remove only output/v7_representations files after backup')
    audit=json.loads((OUT/'v7_data_audit.json').read_text(encoding='utf-8'))
    if audit.get('A_features')!=[vars(f) for f in A_FEATURES] or audit.get('B_features')!=list(FEATURE_NAMES):
        raise ValueError('Cached V7 feature contract differs from code')
    if audit.get('b_source_train_only_scaler_sha256')!=src['v2']['scaler_sha256']:
        raise ValueError('Changed B scaler from V2 after V7 construction')
    scaler=json.loads((OUT/'v7_a_scaler_train_only.json').read_text(encoding='utf-8'))
    if canonical_sha(scaler)!=audit.get('a_scaler_sha256'):
        raise ValueError('V7 A train-only scaler cache hash mismatch')
    a=np.load(OUT/'a_features_scaled.npy',mmap_mode='r')
    if a.shape!=(861570,8):raise ValueError('V7 A feature cache shape mismatch')
    for split in ('train','valid'):
        rows=np.load(OUT/f'common_{split}_rows.npy')
        base=src[split].panel_row.to_numpy(dtype=np.int64)
        if len(rows)==0 or not np.isin(rows,base).all():
            raise ValueError('Common A/B cohort not subset of frozen B anchors')
        # Order of row indices is as stored in original indexed panel, not re-sorted.
        if np.any(rows[1:]<=rows[:-1]):
            raise ValueError('Common cohort order invalid')
        sub=src[split].set_index('panel_row').loc[rows].reset_index()
        if audit[f'{split}_anchor_sha256']!=anchor_sha256(sub):
            raise ValueError('Common cohort changed since initial audit')
    print('[1/7] Existing V7 A data cache verified, no Qlib recalc',flush=True)
    return audit


def build_protocol(audit,device,src,label_scaler):
    core={
        'stage':'V7 preregistered representation diagnostic; 2021–22 VALID only',
        'question':'Does adding daily history help LSTM more with unaggregated OHLCV inputs (A) than with rolling-factor inputs (B)?',
        'hypothesis':'B rolling factors may overlap in historical information; A daily channels may make temporal structure easier to use.',
        'A_feature_spec':[vars(f) for f in A_FEATURES],
        'B_frozen_features':list(FEATURE_NAMES),
        'representations':list(REPRESENTATIONS),'windows':list(WINDOWS),'seeds':list(SEEDS),
        'cohort':'intersection of V2 B eligible30 and A strict30, identical for all 4 model configurations',
        'n_train':audit['common_cohort']['train']['a_and_b_common'],
        'n_valid':audit['common_cohort']['valid']['a_and_b_common'],
        'train_anchor_sha256':audit['train_anchor_sha256'],
        'valid_anchor_sha256':audit['valid_anchor_sha256'],
        'A_scaler_hash':audit['a_scaler_sha256'],
        'B_scaler_hash':audit['b_source_train_only_scaler_sha256'],
        'label_scaler':label_scaler,
        'architecture':{'kind':'LSTM','hidden_size':HIDDEN,'layers':1,'head':[32,16,1],
                        'input_width_A':8,'input_width_B':16,'parameter_counts_differ_by_input_width':True},
        'optimizer':{'kind':'AdamW','lr':LR,'weight_decay':WEIGHT_DECAY,'batch_size':BATCH_SIZE,
                     'max_epochs':EPOCHS,'patience':PATIENCE,'min_delta':MIN_DELTA,
                     'gradient_clip_norm':GRAD_CLIP,'stop_metric':'VALID MSE, not RankIC'},
        'device':device,'primary_metric':'daily equal-weight cross-sectional mean RankIC',
        'support_metrics':['IC','raw MSE','2021/2022 RankIC','seed SD','paired HAC differences'],
        'comparisons':['A30-A1','B30-B1','A30-B30','A1-B1','(A30-A1)-(B30-B1)'],
        'inference':'HAC Newey-West lag 10 for daily contrasts, descriptive after multiple validations',
        'train_dates':'2016-2020','valid_dates':'2021-2022',
        'not_accessed_outcomes':['2023-2024','2025-2026'],
        'data_quality_limits':[
            'A and B differ in channel count (8 vs 16), feature construction and information content.',
            'A/B differences cannot by themselves establish that B rolling-factor histories are redundant.',
            'Common 30d cohort is a conditioned sample (may discard missing A rows).',
            'Early stopping uses VALID MSE; VALID RankIC is repeatedly examined across V3–V7.',
            'No portfolio / after-cost tradability claim; adjusted prices/index PIT remain to be audited.',
        ],
        'software':{'python':sys.version.split()[0],'machine':platform.machine()},
    }
    core['protocol_sha256']=canonical_sha(core)
    file=OUT/'v7_protocol.json'
    if file.is_file():
        existing=json.loads(file.read_text(encoding='utf-8'))
        if existing!=core:raise ValueError('V7 protocol modified; do not silently overwrite earlier run')
    else:
        save_json(file,core)
    return core


def shared_runtime(src,audit):
    rows={name:np.load(OUT/f'common_{name}_rows.npy').astype(np.int64)
          for name in ('train','valid')}
    tr,va=rows['train'],rows['valid']
    labels=src['labels']
    y_train_raw=np.asarray(labels[tr],dtype=np.float64)
    y_valid_raw=np.asarray(labels[va],dtype=np.float64)
    if not (np.isfinite(y_train_raw).all() and np.isfinite(y_valid_raw).all()):
        raise ValueError('Nonfinite common labels')
    label_scaler=fit_train_label_scaler(y_train_raw)
    tr_sample=src['train'].set_index('panel_row').loc[tr]
    va_sample=src['valid'].set_index('panel_row').loc[va]
    if anchor_sha256(tr_sample.reset_index())!=audit['train_anchor_sha256'] or anchor_sha256(va_sample.reset_index())!=audit['valid_anchor_sha256']:
        raise ValueError('V7 train/valid sample changed')
    return rows,y_train_raw,y_valid_raw,label_scaler,va_sample.datetime.to_numpy()


def run_experiments(src,audit,device):
    rows,ytr_raw,yva_raw,label_scaler,dates=shared_runtime(src,audit)
    prot=build_protocol(audit,device,src,label_scaler)
    ytr=target_for_train(ytr_raw,label_scaler)
    yva=target_for_valid(yva_raw,label_scaler)
    mats={'A':np.load(OUT/'a_features_scaled.npy',mmap_mode='r'),
          'B':src['b']}
    summary=[]; daily_maps={}; seed_results=[]
    print('[4/7] Training matched A/B LSTM × 1/30d × 3 seed; ONLY TRAIN and VALID...',flush=True)
    for representation in REPRESENTATIONS:
        for window in WINDOWS:
            names=[f'{representation.lower()}_w{window:02d}_s{seed}' for seed in SEEDS]
            tasks=[]
            for seed,name in zip(SEEDS,names):
                mfile=OUT/f'{name}_metrics.json'
                dfile=OUT/f'{name}_daily.csv'
                weights=OUT/'checkpoints'/f'{name}.pt'
                exist=[p.is_file() for p in (mfile,dfile,weights)]
                if all(exist):
                    m=json.loads(mfile.read_text(encoding='utf-8'))
                    if m.get('protocol_sha256')!=prot['protocol_sha256'] or m.get('seed')!=seed or m.get('window')!=window or m.get('representation')!=representation:
                        raise ValueError('Cached model result conflicts with V7 protocol')
                else:
                    tasks.append((seed,name,mfile,dfile,weights))
            if tasks:
                xtr=materialize(mats[representation],rows['train'],window)
                xva=materialize(mats[representation],rows['valid'],window)
                for seed,name,mfile,dfile,weights in tasks:
                    print(f'   Training {name} | {xtr.shape[1:]} | seed={seed}...',flush=True)
                    pred_scaled,fit=fit_and_predict(xtr,ytr,xva,yva,seed=seed,device=device,checkpoint=weights)
                    pred=pred_to_raw(pred_scaled,label_scaler)
                    metrics,daily=evaluate_validation(dates,yva_raw,pred)
                    metrics.update({'representation':representation,'window':window,'seed':seed,
                                    'name':name,'fit':fit,'protocol_sha256':prot['protocol_sha256']})
                    daily.to_csv(dfile,index=False,encoding='utf-8-sig')
                    save_json(mfile,metrics)
                    print(f'    {name}: RankIC={metrics["rankic_daily_equal_weight"]["mean"]:+.6f} '
                          f'IC={metrics["ic_daily_equal_weight"]["mean"]:+.6f} '
                          f'best_epoch={fit["best_epoch"]}',flush=True)
                del xtr,xva
            else:
                print(f'   Reused all {representation}-window{window} checkpoints.',flush=True)
            stored=[]
            for seed,name in zip(SEEDS,names):
                m=json.loads((OUT/f'{name}_metrics.json').read_text(encoding='utf-8'))
                if m.get('protocol_sha256')!=prot['protocol_sha256']:
                    raise ValueError('V7 cached metrics protocol mismatch')
                d=pd.read_csv(OUT/f'{name}_daily.csv',parse_dates=['date'])
                stored.append(d)
                seed_results.append({'representation':representation,'window':window,'seed':seed,
                                     'rankic':m['rankic_daily_equal_weight']['mean'],
                                     'ic':m['ic_daily_equal_weight']['mean'],'mse_raw':m['mse_raw'],
                                     'best_epoch':m['fit']['best_epoch'],
                                     'n_parameters':m['fit']['parameter_count']})
            daily_maps[(representation,window)]=average_daily(stored)
            current=[m for m in seed_results if m['representation']==representation and m['window']==window]
            ranks=np.asarray([m['rankic'] for m in current],dtype=float)
            ints=np.asarray([m['ic'] for m in current],dtype=float)
            mses=np.asarray([m['mse_raw'] for m in current],dtype=float)
            av=daily_maps[(representation,window)]
            entry={'representation':representation,'window':window,
                   'n_features':mats[representation].shape[1],
                   'n_train':len(rows['train']),'n_valid':len(rows['valid']),
                   'n_valid_days':len(av),'seeds':len(SEEDS),
                   'rankic_mean_3seed':float(ranks.mean()),'rankic_sd_seed':float(ranks.std(ddof=1)),
                   'ic_mean_3seed':float(ints.mean()),'mse_raw_mean_3seed':float(mses.mean()),
                   'n_parameters':int(current[0]['n_parameters']),
                   'rankic_2021':float(av.loc[av.date.dt.year==2021,'rankic'].mean()),
                   'rankic_2022':float(av.loc[av.date.dt.year==2022,'rankic'].mean())}
            summary.append(entry)
    print('[5/7] Paired daily RankIC, including representation × history-length interaction...',flush=True)
    a1=daily_maps[('A',1)];a30=daily_maps[('A',30)]
    b1=daily_maps[('B',1)];b30=daily_maps[('B',30)]
    paired={
        'A30_minus_A1':mean_rankic_contrast(a30,a1),
        'B30_minus_B1':mean_rankic_contrast(b30,b1),
        'A30_minus_B30':mean_rankic_contrast(a30,b30),
        'A1_minus_B1':mean_rankic_contrast(a1,b1),
        'diff_in_diff_A_history_minus_B_history':difference_in_differences(a1,a30,b1,b30),
    }
    pd.DataFrame(summary).sort_values(['representation','window']).to_csv(
        OUT/'v7_representation_summary.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(seed_results).sort_values(['representation','window','seed']).to_csv(
        OUT/'v7_seed_results.csv',index=False,encoding='utf-8-sig')
    save_json(OUT/'v7_paired_rankic.json',{
        'protocol_sha256':prot['protocol_sha256'],
        'paired_valid_2021_2022':paired,
        'caveat':'All differences are exploratory after V3–V6 validation reuse; different A/B information sets and widths.'})
    print('[6/7] Matched experiments summary:',flush=True)
    print(pd.DataFrame(summary).to_string(index=False),flush=True)
    print('[7/7] V7 comparison complete; no 2023–2026 predictions evaluated.',flush=True)
    print('V7_REPRESENTATION_EXPERIMENT_COMPLETE:',OUT,flush=True)
    print('Share: v7_data_audit.json, v7_representation_summary.csv, v7_seed_results.csv, '
          'v7_paired_rankic.json, run_v7.log',flush=True)


def main():
    ap=argparse.ArgumentParser(description='V7 controlled A/B representation comparison (TRAIN/VALID only)')
    ap.add_argument('--stage',choices=['data','all'],default='all')
    ap.add_argument('--device',choices=['auto','mps','cpu'],default='auto')
    args=ap.parse_args()
    src=load_sources()
    audit=ensure_a_cache(src)
    if args.stage=='data':
        print('V7_A_DATA_AUDIT_COMPLETE:',OUT,flush=True)
        return
    dev=torch_device(args.device)
    print('Resolved device:',dev,'; no 2023–2026 outcomes accessed.',flush=True)
    run_experiments(src,audit,dev)


if __name__=='__main__':
    main()
