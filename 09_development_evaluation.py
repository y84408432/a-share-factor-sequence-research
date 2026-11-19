#!/usr/bin/env python3
"""V9 once-only 2023–2024 development evaluation of FOUR previously fitted models.

Preflight: audit 2023–24 A features WITHOUT consulting development outcomes.
Evaluation: verify frozen preflight/model hashes; predict and evaluate 2023–24.
Never select/refit/tune any models using development results; 2025–26 locked.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from lstm_research.feature_spec import FEATURE_NAMES, LABEL_EXPR
from lstm_research.sequences import index_for
from lstm_research.pilot import anchor_sha256,pred_to_raw,save_json
from lstm_research.lstm_window_v5 import torch_device
from lstm_research.representation_v7 import A_FEATURES,A_NAMES,model_for
from lstm_research.v9_development import (
    MODEL_IDS,SEEDS,file_hash,model_registry,raw_a_to_scaled,audit_overlap,
    check_period,cohort_by_a,make_predictions,check_lstm_checkpoint,
    summarize_group,compare_daily,
)

V1=ROOT/'output/v1_data'
V2=ROOT/'output/v2_sequences'
V3=ROOT/'output/v3_pilot'
V4=ROOT/'output/v4_hypothesis'
V7=ROOT/'output/v7_representations'
V8=ROOT/'output/v8_a30_controls'
OUT=ROOT/'output/v9_development'
SPECFILE=OUT/'v9_preflight.json'
A_FILE=OUT/'a_features_2022_2024_scaled.npy'
ROW_FILE=OUT/'development_common_rows.npy'
PROTOCOL_FILE=OUT/'v9_frozen_protocol.json'


def read_json(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha_obj(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def preconditions():
    for fp in (V1/'frozen_factor_panel.pkl.gz',V2/'sample_index.pkl.gz',
               V2/'v2_sequence_audit.json',V2/'features_scaled.npy',V2/'labels.npy',
               V7/'v7_data_audit.json',V7/'v7_protocol.json',
               V7/'v7_a_scaler_train_only.json',V7/'a_features_scaled.npy',
               V4/'ridge_1_valid_metrics.json',V8/'v8_protocol.json',
               V8/'ridge_A30_flat_metrics.json',V3/'label_scaler_train_only.json'):
        if not fp.is_file():raise FileNotFoundError(f'Missing completed V1–V8 artifact: {fp}')
    a7=read_json(V7/'v7_data_audit.json')
    prot7=read_json(V7/'v7_protocol.json')
    v8=read_json(V8/'v8_protocol.json')
    v2=read_json(V2/'v2_sequence_audit.json')
    if (v2['features']!=list(FEATURE_NAMES) or len(A_NAMES)!=8 or
            a7['A_features']!=[vars(x) for x in A_FEATURES]):
        raise ValueError('Frozen A8/B16 feature definition changed')
    if (prot7['A_scaler_hash']!=a7['a_scaler_sha256'] or
            prot7['B_scaler_hash']!=v2['scaler_sha256'] or
            v8.get('v7_protocol_sha256')!=prot7['protocol_sha256']):
        raise ValueError('V7/V8 source protocol/scaler mismatch')
    if prot7.get('train_dates')!='2016-2020' or prot7.get('valid_dates')!='2021-2022':
        raise ValueError('Unexpected model training period')
    sl=read_json(V7/'v7_a_scaler_train_only.json')
    if sl.get('fit_split')!='train' or sl.get('features')!=list(A_NAMES):
        raise ValueError('Wrong original A scaler')
    label_scale=prot7['label_scaler']
    v3_scale=read_json(V3/'label_scaler_train_only.json')
    for k in ('low','high','mean','std'):
        if abs(float(label_scale[k])-float(v3_scale[k]))>1e-10:
            raise ValueError('V3/V7 target scaler mismatch')
    v4_met=read_json(V4/'ridge_1_valid_metrics.json')
    if v4_met.get('model')!='ridge_1' or v4_met.get('seed')!=20261115:
        raise ValueError('Wrong B1 Ridge validation source')
    v8_met=read_json(V8/'ridge_A30_flat_metrics.json')
    if v8_met.get('type')!='ridge' or v8_met.get('n_samples')!=139220:
        raise ValueError('Wrong A30 Ridge validation source')
    if not all((V7/f'a_w30_s{s}_metrics.json').exists() and (V7/f'b_w30_s{s}_metrics.json').exists() for s in SEEDS):
        raise ValueError('Missing V7 per-seed model metric')
    reg=model_registry(ROOT)
    return a7,prot7,sl,label_scale,reg


def audit():
    print('[1/5] Verify V1–V8 frozen model files, scalers and agreements...',flush=True)
    audit7,prot7,scaler,target_scaler,registry=preconditions()
    v2meta=read_json(V2/'v2_sequence_audit.json')
    idx=pd.read_pickle(V2/'sample_index.pkl.gz',compression='gzip')
    dev=index_for(idx,'development',30)
    check_period(dev)
    expected=next(r['eligible_30'] for r in v2meta['counts'] if r['split']=='development')
    if len(dev)!=expected:
        raise ValueError(f'V2 development B30 samples changed: {len(dev)} vs {expected}')
    print(f'  frozen B30 development anchors={len(dev):,}; label boundaries checked',flush=True)
    print('[2/5] Recompute ONLY A8 daily FEATURES for 2022-09 through 2024-12...',flush=True)
    import qlib
    from qlib.data import D
    from qlib.constant import REG_CN
    source=Path.home()/'.qlib/qlib_data/cn_data'
    if not source.exists():raise FileNotFoundError(source)
    qlib.init(provider_uri=str(source),region=REG_CN)
    expressions=[f.expression for f in A_FEATURES]
    query=D.features(D.instruments(market='csi300'),expressions,
                     start_time='2022-09-01',end_time='2024-12-31',freq='day',disk_cache=0)
    if query.shape[1]!=8:raise ValueError('Wrong Qlib A feature count')
    query.columns=list(A_NAMES)
    query=query.reset_index()
    if not {'instrument','datetime'}.issubset(query.columns):
        raise ValueError('Unexpected Qlib feature row index')
    query['instrument']=query['instrument'].astype(str)
    query['datetime']=pd.to_datetime(query['datetime'])
    if query[['instrument','datetime']].duplicated().any():
        raise ValueError('Qlib duplicate instrument dates')
    qdf=query.set_index(['instrument','datetime']).sort_index()
    print(f'  Qlib returned {len(query):,} feature rows through 2024 only',flush=True)
    print('[3/5] Strict source overlap / pre-2023 A-cache consistency checks...',flush=True)
    panel=pd.read_pickle(V1/'frozen_factor_panel.pkl.gz',compression='gzip')
    if len(panel)!=v2meta['n_panel_rows']:
        raise ValueError('V1 panel length differs from V2')
    panel['datetime']=pd.to_datetime(panel['datetime'])
    is_context=(panel.datetime>=pd.Timestamp('2022-09-01'))&(panel.datetime<=pd.Timestamp('2024-12-31'))
    positions=np.flatnonzero(is_context.to_numpy())
    target=pd.MultiIndex.from_frame(panel.iloc[positions][['instrument','datetime']])
    araw=np.full((len(panel),8),np.nan,dtype=np.float32)
    araw[positions]=qdf.reindex(target).to_numpy(dtype=np.float32)
    eval_rows=dev.panel_row.to_numpy(dtype=np.int64)
    overlap=audit_overlap(araw,panel,eval_rows,FEATURE_NAMES)
    scaled=raw_a_to_scaled(araw,scaler)
    v7_scaled=np.load(V7/'a_features_scaled.npy',mmap_mode='r')
    context=(panel.datetime>=pd.Timestamp('2022-10-01'))&(panel.datetime<=pd.Timestamp('2022-12-31'))
    within=np.flatnonzero(context.to_numpy())
    comparable=within[np.isfinite(araw[within]).all(axis=1)]
    if len(comparable)<500:
        raise ValueError('Insufficient 2022 overlap for A scaler continuity audit')
    err=float(np.max(np.abs(scaled[comparable]-v7_scaled[comparable])))
    if err>2e-5:raise ValueError(f'2022 historical A feature/scaling mismatch, max={err}')
    finite=np.isfinite(araw).all(axis=1)
    common=cohort_by_a(dev,finite)
    check_period(common)
    if len(common)<120000:
        raise ValueError('Unexpected loss of common 30-day A/B samples')
    # V2 eligible30 guarantees B continuity and factor finiteness; A mask is extra.
    r=common.panel_row.to_numpy(dtype=np.int64)
    offsets=np.arange(29,-1,-1,dtype=np.int64)
    for k in range(0,len(r),4096):
        x=scaled[r[k:k+4096,None]-offsets[None,:]]
        if not np.isfinite(x).all():raise ValueError('Bad A input window')
    print(f'  A/B common development anchors={len(common):,}, A exclusions={len(dev)-len(common):,}; formula/scaler checks passed',flush=True)
    print('[4/5] Freeze exact samples, feature cache and model hashes BEFORE outcomes...',flush=True)
    OUT.mkdir(parents=True,exist_ok=True)
    if any(p.exists() for p in (SPECFILE,A_FILE,ROW_FILE,PROTOCOL_FILE)):
        raise RuntimeError('V9 preflight artifacts already exist. Use --stage evaluate or inspect files; no silent overwrites.')
    np.save(A_FILE,scaled)
    np.save(ROW_FILE,r)
    m={
        'status':'V9 preflight only; 2023–2024 labels not indexed or evaluated',
        'stage':'2023-2024 frozen development evaluation',
        'cohort':'V2 development B eligible30 INTERSECT A strict30',
        'B_eligible30_n':len(dev),'A_B_common_n':len(common),
        'A_exclusions':int(len(dev)-len(common)),
        'A_B_overlap_formula':overlap,'A_2022_scaled_max_abs_diff':err,
        'A_query_start':'2022-09-01','A_query_end':'2024-12-31',
        'model_ids':list(MODEL_IDS),'model_registry':registry,
        'source_A_train_scaler_hash':audit7['a_scaler_sha256'],
        'source_B_train_scaler_hash':v2meta['scaler_sha256'],
        'source_label_scaler':target_scaler,
        '2023_2024_anchor_sha256':anchor_sha256(common),
        '2023_2024_dates':{'min':str(common.datetime.min().date()),'max':str(common.datetime.max().date()),
                           'n':int(common.datetime.nunique())},
        'final_holdout':'NOT evaluated; no 2025-2026 Qlib A feature query',
        'quality_limitations':['Historical CSI300 membership PIT not independently verified.',
          'Qlib adjusted open/open label is not an executable trade simulation.',
          'Development was chosen after extensive exploratory 2021–2022 validation experiments; reported HAC is conditional.',
          '2025–2026 remains locked and was partly seen in older projects.']
    }
    m['A_cache_sha256']=file_hash(A_FILE)
    m['cohort_rows_file_sha256']=file_hash(ROW_FILE)
    save_json(SPECFILE,m)
    prot={
        'stage':'V9 one-shot out-of-time DEVELOPMENT evaluation, 2023–24 ONLY',
        'A8_spec':[vars(x) for x in A_FEATURES],
        'B16_feature_order':list(FEATURE_NAMES),
        'training':'2016–2020, frozen original checkpoints (V4,V7,V8)',
        'validation':'2021–2022 previously used to choose checkpoints; no re-selection',
        'development':'2023–2024 one-shot; do not tune or rank-select after viewing',
        'final_holdout':'2025–2026 locked',
        'models':registry,'models_order':list(MODEL_IDS),
        'cohort_sha256':m['2023_2024_anchor_sha256'],
        'n_development':len(common),
        'A_scaler_hash':m['source_A_train_scaler_hash'],
        'B_scaler_hash':m['source_B_train_scaler_hash'],
        'label_scaler':target_scaler,
        'metric_primary':'daily cross-sectional mean RankIC, equally weighted over days',
        'lstm_three_seed_reporting':'mean of per-seed daily RankIC, NOT RankIC of average predictions',
        'supplemental':['daily Pearson IC','raw MSE','year 2023 and 2024 separately','LSTM seed SD'],
        'inference':'date-wise paired RankIC contrasts, Newey-West lag 10, exploratory conditional on V3–V8 selection',
        'primary_pairwise':['A30_LSTM-A30_Ridge','A30_LSTM-B30_LSTM','B1_Ridge-A30_LSTM'],
        'no_training_no_fit_no_preprocessing_update':True,
        'do_not_select_model_after_development':True,
    }
    prot['protocol_sha256']=sha_obj(prot)
    save_json(PROTOCOL_FILE,prot)
    print('[5/5] Preflight complete. 2023–2024 results NOT read; next step --stage evaluate.',flush=True)
    print('V9_PREFLIGHT_COMPLETE:',OUT,flush=True)


def frozen_inputs():
    audit7,prot7,scaler,label_scaler,registry=preconditions()
    if not (SPECFILE.is_file() and PROTOCOL_FILE.is_file() and A_FILE.is_file() and ROW_FILE.is_file()):
        raise FileNotFoundError('V9 preflight incomplete; run --stage audit first')
    meta=read_json(SPECFILE);protocol=read_json(PROTOCOL_FILE)
    if (meta['model_registry']!=registry or protocol['models']!=registry or
            meta.get('A_cache_sha256')!=file_hash(A_FILE) or
            meta.get('cohort_rows_file_sha256')!=file_hash(ROW_FILE) or
            protocol['protocol_sha256']!=sha_obj({k:v for k,v in protocol.items() if k!='protocol_sha256'})):
        raise ValueError('V9 model registry/protocol changed; cannot evaluate')
    rows=np.load(ROW_FILE)
    scaled=np.load(A_FILE,mmap_mode='r')
    if (len(rows)!=meta['A_B_common_n'] or scaled.shape!=(861570,8) or
            not np.array_equal(rows,np.unique(rows))):
        raise ValueError('Frozen V9 cohort or A feature cache invalid')
    idx=pd.read_pickle(V2/'sample_index.pkl.gz',compression='gzip')
    dev=index_for(idx,'development',30)
    sub=dev.set_index('panel_row').loc[rows].reset_index()
    check_period(sub)
    if anchor_sha256(sub)!=meta['2023_2024_anchor_sha256']:
        raise ValueError('Development sample fingerprint differs from preflight')
    return meta,protocol,sub,scaled,label_scaler


def valid_reference():
    v4=read_json(V4/'ridge_1_valid_metrics.json')
    v8=read_json(V8/'ridge_A30_flat_metrics.json')
    ref={'B1_Ridge':v4['rankic_daily_equal_weight']['mean'],
         'A30_Ridge':v8['rankic_daily_equal_weight']['mean']}
    for group,prefix in [('A30_LSTM','a'),('B30_LSTM','b')]:
        rr=[]
        for seed in SEEDS:
            m=read_json(V7/f'{prefix}_w30_s{seed}_metrics.json')
            if m.get('representation')!=prefix.upper() or m.get('window')!=30 or m.get('seed')!=seed:
                raise ValueError('Frozen V7 per-seed metrics changed')
            rr.append(m['rankic_daily_equal_weight']['mean'])
        ref[group]=float(np.mean(rr))
    return ref


def eval_once(device):
    from lstm_research.v9_development import evaluate_group
    print('[1/6] Confirm frozen preflight and 4 model hashes; READ development labels only now...',flush=True)
    meta,protocol,sub,aa,label_scaler=frozen_inputs()
    result_files=[OUT/'v9_model_summary.csv',OUT/'v9_yearly_rankic.csv',
                  OUT/'v9_paired_rankic.json',OUT/'v9_model_seed_detail.csv',
                  OUT/'v9_development_predictions.csv.gz']
    if any(p.exists() for p in result_files):
        if all(p.is_file() for p in result_files):
            print('V9_ALREADY_EVALUATED: results exist; do not rerun or tune.',flush=True)
            return
        raise RuntimeError('Partial V9 outputs exist; inspect before proceeding (do not overwrite)')
    rows=sub.panel_row.to_numpy(np.int64)
    # Only development rows are selected. The numpy label file includes future dates,
    # but 2025–2026 indices are never constructed or selected in this runner.
    raw_y=np.asarray(np.load(V2/'labels.npy',mmap_mode='r')[rows],dtype=np.float64)
    if not np.isfinite(raw_y).all():raise ValueError('Nonfinite dev target on eligible cohort')
    if not np.array_equal(pd.to_datetime(sub.datetime).dt.year.unique(),[2023,2024]):
        raise ValueError('Expected calendar years 2023 and 2024')
    dates=sub.datetime.to_numpy()
    bb=np.load(V2/'features_scaled.npy',mmap_mode='r')
    if bb.shape!=(861570,16):raise ValueError('B frozen feature shape mismatch')
    import joblib
    b1=joblib.load(V4/'ridge_1_model.joblib')
    ar=joblib.load(V8/'checkpoints/ridge_A30_flat.pt')
    print(f'  development common cohort {len(rows):,}; prediction device {device}',flush=True)
    pred_scaled={}
    print('[2/6] Frozen B1 Ridge (16 features, latest date only)...',flush=True)
    pred_scaled['B1_Ridge']=[make_predictions(b1,bb,rows,kind='ridge',window=1)]
    print('[3/6] Frozen A30 Ridge (240 features, oldest -> newest)...',flush=True)
    pred_scaled['A30_Ridge']=[make_predictions(ar,aa,rows,kind='ridge',window=30)]
    import torch
    for name,channels,prefix,mat in [
        ('A30_LSTM',8,'a',aa),('B30_LSTM',16,'b',bb)]:
        print(f'[4/6] Frozen {name} × {len(SEEDS)} seeds (no fitting)...',flush=True)
        pp=[]
        for seed in SEEDS:
            ckpt=V7/f'checkpoints/{prefix}_w30_s{seed}.pt'
            try:
                state=torch.load(ckpt,map_location='cpu',weights_only=True)
            except TypeError:
                state=torch.load(ckpt,map_location='cpu')
            check_lstm_checkpoint(state,channels,seed)
            model=model_for(channels)
            model.load_state_dict(state['state_dict'],strict=True)
            model.eval()
            pp.append(make_predictions(model,mat,rows,kind='lstm',window=30,device=device))
            print('   inferred',name,'seed',seed,flush=True)
            del model,state
        pred_scaled[name]=pp
    print('[5/6] Evaluate frozen predictions with daily RankIC, year and paired HAC...',flush=True)
    summaries=[];seed_out=[];daily={};raw_predictions={}
    valid=valid_reference()
    for key in MODEL_IDS:
        pp=[pred_to_raw(x,label_scaler) for x in pred_scaled[key]]
        report,individual,day=summarize_group(key,dates,raw_y,pp,
                                             list(SEEDS) if len(pp)>1 else ['fixed'])
        report['valid_rankic_2021_2022']=valid[key]
        report['development_minus_validation']=report['rankic_seed_mean']-valid[key]
        summaries.append(report)
        seed_out.extend(individual)
        daily[key]=day
        raw_predictions[key]=np.stack(pp,axis=0).mean(axis=0)
        day.to_csv(OUT/f'{key}_daily_rankic.csv',index=False,encoding='utf-8-sig')
        print(f'   {key}: RankIC={report["rankic_seed_mean"]:+.6f} '
              f'(2023={report["rankic_2023"]:+.6f}, 2024={report["rankic_2024"]:+.6f})',flush=True)
    pairs=[('A30_LSTM','A30_Ridge'),('A30_LSTM','B30_LSTM'),('B1_Ridge','A30_LSTM')]
    contrasts={f'{a}_minus_{b}':compare_daily(daily[a],daily[b]) for a,b in pairs}
    summary=pd.DataFrame(summaries)
    summary.to_csv(OUT/'v9_model_summary.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(seed_out).to_csv(OUT/'v9_model_seed_detail.csv',index=False,encoding='utf-8-sig')
    by_year=[]
    for name in MODEL_IDS:
        for year in (2023,2024):
            dd=daily[name]
            subset=dd[dd.date.dt.year==year]
            by_year.append({'model':name,'year':year,'n_days':len(subset),
                 'rankic':float(subset.rankic.mean()),'ic':float(subset.ic.mean())})
    pd.DataFrame(by_year).to_csv(OUT/'v9_yearly_rankic.csv',index=False,encoding='utf-8-sig')
    save_json(OUT/'v9_paired_rankic.json',{
        'protocol_sha256':protocol['protocol_sha256'],
        'n_unique_development_days':int(sub.datetime.nunique()),
        'comparisons':contrasts,
        'caveat':'Exploratory, conditional on V3–V8 repeated 2021–2022 validation; no multiple-testing correction.'})
    result=pd.DataFrame({'instrument':sub.instrument.to_numpy(),
                         'datetime':sub.datetime.to_numpy(),
                         'label_raw':raw_y})
    for k in MODEL_IDS:result[k+'_mean_raw_pred']=raw_predictions[k]
    result.to_csv(OUT/'v9_development_predictions.csv.gz',index=False,compression='gzip')
    print('[6/6] DEVELOPMENT-only evaluation complete. 2025–2026 outcomes untouched.',flush=True)
    print(summary[['model','rankic_seed_mean','rankic_2023','rankic_2024','valid_rankic_2021_2022']].to_string(index=False),flush=True)
    print('V9_DEVELOPMENT_EVALUATION_COMPLETE:',OUT,flush=True)
    print('Upload v9_preflight.json, v9_model_summary.csv, v9_yearly_rankic.csv, v9_paired_rankic.json and run_v9.log',flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',required=True,choices=['audit','evaluate'])
    p.add_argument('--device',choices=['auto','cpu','mps'],default='auto')
    args=p.parse_args()
    if args.stage=='audit':audit()
    else:eval_once(torch_device(args.device))


if __name__=='__main__':main()
