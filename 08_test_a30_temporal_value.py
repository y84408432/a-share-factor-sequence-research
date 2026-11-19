#!/usr/bin/env python3
"""V8 exploratory controls for V7 A30: order D + equal-information Ridge/MLP.

Only 2016–2020 TRAIN and 2021–2022 VALID are indexed. V7 trained A30
checkpoints/daily prediction metrics are REUSED, not reselected or overwritten.
Do not infer independent significance after V3–V7 validation re-use.
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

# V7's IO / research-contract functions live in the V7 runner script,
# not in lstm_research.representation_v7 (which owns feature/model helpers).
# Load that already-frozen runner as a module without executing main().
import importlib.util
_v7_spec=importlib.util.spec_from_file_location(
    "_frozen_v7_runner", ROOT / "07_compare_representations.py"
)
if _v7_spec is None or _v7_spec.loader is None:
    raise ImportError("Cannot load frozen V7 runner script")
v7=importlib.util.module_from_spec(_v7_spec)
_v7_spec.loader.exec_module(v7)
from lstm_research.pilot import (evaluate_validation,pred_to_raw,
                                  target_for_train,target_for_valid,save_json)
from lstm_research.lstm_window_v5 import SEEDS,torch_device
from lstm_research.v8_controls import (
    REPEATS,WINDOW,FEATURES,RIDGE_ALPHA,MLP_HIDDEN,MLP_DROPOUT,
    perm_seed,permute_past,flatten_a30,fit_ridge,fit_flat_mlp,
    same_day_metrics,mean_daily_diff,
)

V7=ROOT/'output'/'v7_representations'
OUT=ROOT/'output'/'v8_a30_controls'


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def hash_json(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False,
        separators=(',',':')).encode('utf-8')).hexdigest()


def validate_v7_results(audit,src,device):
    original_protocol=read_json(V7/'v7_protocol.json')
    if original_protocol.get('protocol_sha256')!=v7.canonical_sha({k:v for k,v in original_protocol.items() if k!='protocol_sha256'}):
        raise ValueError('V7 protocol self-hash mismatch')
    if (original_protocol.get('valid_anchor_sha256')!=audit['valid_anchor_sha256'] or
            original_protocol.get('train_anchor_sha256')!=audit['train_anchor_sha256']):
        raise ValueError('V7 protocol cohort mismatch')
    if (original_protocol.get('A_scaler_hash')!=audit['a_scaler_sha256'] or
            original_protocol.get('B_scaler_hash')!=audit['b_source_train_only_scaler_sha256']):
        raise ValueError('V7 scaler fingerprints differ')
    if original_protocol.get('device')!=device:
        raise ValueError('V8 neural device differs from V7; use same mps/cpu device to compare reliably')
    if (original_protocol.get('windows')!=[1,30] or
            original_protocol.get('seeds')!=list(SEEDS) or
            original_protocol.get('n_train')!=327555 or
            original_protocol.get('n_valid')!=139220):
        raise ValueError('V7 training protocol changed unexpectedly')
    if original_protocol.get('label_scaler',{}).get('fit_split')!='train':
        raise ValueError('V7 labels not train-only normalized')
    srcrows,ytr_raw,yva_raw,scaler,dates=v7.shared_runtime(src,audit)
    for key in ('low','high','mean','std'):
        if not np.isclose(original_protocol['label_scaler'][key],scaler[key],rtol=1e-12,atol=1e-12):
            raise ValueError('Target scaler changed since V7')
    originals={}
    for seed in SEEDS:
        name=f'a_w30_s{seed}'
        metric=V7/f'{name}_metrics.json'
        daily=V7/f'{name}_daily.csv'
        checkpoint=V7/'checkpoints'/f'{name}.pt'
        if not all(f.is_file() for f in (metric,daily,checkpoint)):
            raise FileNotFoundError(f'Missing V7 A30 artifacts for {name}; finish V7 first')
        m=read_json(metric)
        if (m.get('protocol_sha256')!=original_protocol['protocol_sha256'] or
                m.get('window')!=30 or m.get('representation')!='A' or
                m.get('seed')!=seed or m.get('n_samples')!=len(srcrows['valid'])):
            raise ValueError(f'Original A30 metric mismatches frozen V7 protocol: {name}')
        d=pd.read_csv(daily,parse_dates=['date'])
        originals[seed]=d
    v7_daily=same_day_metrics(list(originals.values()))
    if len(v7_daily)!=479 or not np.isfinite(v7_daily.rankic.to_numpy()).all():
        raise ValueError('Unexpected original V7 daily data')
    return original_protocol,srcrows,ytr_raw,yva_raw,scaler,dates,originals,v7_daily


def write_protocol(audit,v7_protocol,device):
    data={
        'stage':'V8 exploratory, VALID-only, AFTER V3-V7 validation exploration',
        'hypothesis':'A30 improvement may come from additional daily information, not necessarily chronological order or LSTM recurrence',
        'input':'A30: 30 ordered days × 8 daily OHLCV features; identical samples and labels to V7',
        'v7_protocol_sha256':v7_protocol['protocol_sha256'],
        'v7_a_scaler_sha256':audit['a_scaler_sha256'],
        'train_anchor_sha256':audit['train_anchor_sha256'],
        'valid_anchor_sha256':audit['valid_anchor_sha256'],
        'train_count':327555,'valid_count':139220,
        'seeds':list(SEEDS),'repeats_order_D':list(REPEATS),
        'order_D':'Fixed-per-sample shuffled first 29 whole-day vectors in BOTH train and valid, latest day preserved; independently generated across samples/splits/seeds/repeats; original A30 reused',
        'ridge':{'alpha':RIDGE_ALPHA,'solver':'lsqr','tol':0.001,'flatten':240,'fit_intercept':True},
        'mlp_flat':{'hidden':list(MLP_HIDDEN),'dropout':MLP_DROPOUT,
                    'loss':'MSE scaled labels','optimizer':'AdamW','lr':0.001,
                    'weight_decay':0.001,'epochs_max':12,'patience':3,
                    'early_stop':'2021–2022 validation MSE, not RankIC','seeds':list(SEEDS)},
        'reused_lstm_A30':{'hidden_size':32,'layers':1,'head':[32,16,1],
                          'seeds':list(SEEDS),'from':'V7 checkpoint-selected by valid MSE'},
        'statistic':'mean daily cross-sectional RankIC with HAC Newey-West 10-lag date-wise paired contrasts',
        'comparisons':['Dshuffle_A30_minus_original_A30',
                       'original_A30_minus_Ridge_flat_A30',
                       'original_A30_minus_MLP_flat_A30',
                       'MLP_flat_A30_minus_Ridge_flat_A30'],
        'train_period':'2016-2020','valid_period':'2021-2022',
        'excluded_periods':['2023-2024','2025-2026'],
        'device':device,
        'limitations':[
            'A30 heldout relative to future years only; repeatedly inspected 2021–2022 validation is exploratory, not independent inference.',
            'D order shuffle destroys chronological patterns but changes optimization dynamics; an effect is not causal identification of time dependence.',
            'Ridge, flat MLP and LSTM have different capacity, regularization inductive biases and training procedures; same information set does not imply capacity match.',
            'Training stopping and neural checkpoint selection use VALID MSE; rank metric reported, not used to select weights.',
            'Label uses Qlib adjusted next-open to sixth-open; historical index point-in-time, price adjustment and tradability still require audit.',
            'D repeats share same market sample: date count is 479, never 479×seeds×repeats pseudo-observations.',
        ],
        'python':sys.version.split()[0], 'machine':platform.machine(),
    }
    data['protocol_sha256']=hash_json(data)
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'v8_protocol.json'
    if path.exists():
        if read_json(path)!=data:raise ValueError('Existing V8 protocol differs; refuse to mix results')
    else:
        save_json(path,data)
    return data


def model_paths(name):
    return (OUT/f'{name}_metrics.json',OUT/f'{name}_daily.csv',
            OUT/'checkpoints'/f'{name}.pt')


def check_cache(name,protocol,*,expect_checkpoint=True):
    metric,daily,ckpt=model_paths(name)
    existing=[metric.is_file(),daily.is_file(),ckpt.is_file()]
    if any(existing) and not all(existing):
        raise RuntimeError(f'Incomplete saved result {name}; inspect it before proceeding (do not blindly overwrite)')
    if all(existing):
        mm=read_json(metric)
        if mm.get('protocol_sha256')!=protocol['protocol_sha256'] or mm.get('name')!=name:
            raise ValueError(f'V8 cached result does not match protocol: {name}')
        return True
    return False


def write_model(name,p_scaled,train_note,dates,yva_raw,label_scaler,protocol,*,extra):
    raw=pred_to_raw(p_scaled,label_scaler)
    m,daily_frame=evaluate_validation(dates,yva_raw,raw)
    m.update({'name':name,'protocol_sha256':protocol['protocol_sha256'],
              'train_note':train_note,**extra})
    metric,daily_path,_=model_paths(name)
    daily_frame.to_csv(daily_path,index=False,encoding='utf-8-sig')
    save_json(metric,m)
    print(f'  {name}: RankIC={m["rankic_daily_equal_weight"]["mean"]:+.6f} '
          f'IC={m["ic_daily_equal_weight"]["mean"]:+.6f}',flush=True)


def do_order(xtr,xva,ytr,yva,dates,yva_raw,label_scaler,protocol,device):
    print('[2/5] D control: train+valid shuffled history, 3 seeds × 3 permutation repeats...',flush=True)
    from lstm_research.representation_v7 import fit_and_predict
    for seed in SEEDS:
        for repeat in REPEATS:
            name=f'D_A30_s{seed}_r{repeat}'
            if check_cache(name,protocol):
                print('  reused',name,flush=True);continue
            print(f'  fitting {name}',flush=True)
            t=permute_past(xtr,perm_seed(seed,repeat,'train'))
            v=permute_past(xva,perm_seed(seed,repeat,'valid'))
            ckpt=model_paths(name)[2]
            p,fit=fit_and_predict(t,ytr,v,yva,seed=seed,device=device,checkpoint=ckpt)
            write_model(name,p,'V7 same LSTM trainer with independently shuffled history in BOTH splits',
                dates,yva_raw,label_scaler,protocol,
                extra={'type':'order_D','seed':seed,'repeat':repeat,
                       'best_epoch':fit['best_epoch'],'fit':fit})
            del t,v


def do_baselines(xtr,xva,ytr,yva,dates,yva_raw,label_scaler,protocol,device):
    print('[3/5] Equal-A30-input flat Ridge and three MLP seeds...',flush=True)
    train=flatten_a30(xtr);valid=flatten_a30(xva)
    name='ridge_A30_flat'
    if not check_cache(name,protocol):
        print('  fitting Ridge 240 features...',flush=True)
        import joblib
        pred,model=fit_ridge(train,ytr,valid)
        ckpt=model_paths(name)[2]
        ckpt.parent.mkdir(parents=True,exist_ok=True)
        joblib.dump(model,ckpt)
        write_model(name,pred,'Ridge(alpha=1000) on exactly same A30 flat data',
            dates,yva_raw,label_scaler,protocol,
            extra={'type':'ridge','n_parameters':len(model.coef_)+1})
    else: print('  reused Ridge',flush=True)
    for seed in SEEDS:
        name=f'mlp_A30_flat_s{seed}'
        if check_cache(name,protocol):
            print('  reused',name,flush=True);continue
        pred,fit=fit_flat_mlp(train,ytr,valid,yva,seed=seed,device=device,
                             checkpoint=model_paths(name)[2])
        write_model(name,pred,'V3-style flattened MLP, seed-controlled',
                    dates,yva_raw,label_scaler,protocol,
                    extra={'type':'mlp_flat','seed':seed,'best_epoch':fit['best_epoch'],'fit':fit})
    del train,valid


def load_day(name):
    return pd.read_csv(model_paths(name)[1],parse_dates=['date'])


def finish_summary(protocol,originals,original_mean):
    """Write only fully completed experiments; request all when --experiment all."""
    experiments=[]; compared={}
    for seed in SEEDS:
        for repeat in REPEATS:
            name=f'D_A30_s{seed}_r{repeat}'
            if check_cache(name,protocol):experiments.append(name)
    count_d=len(experiments)
    mlps=[f'mlp_A30_flat_s{s}' for s in SEEDS]
    have_mlp=all(check_cache(n,protocol) for n in mlps)
    have_ridge=check_cache('ridge_A30_flat',protocol)
    results=[]
    def row(name,kind):
        m=read_json(model_paths(name)[0]);v=m['rankic_daily_equal_weight']
        return {'name':name,'group':kind,'rankic':v['mean'],'ic':m['ic_daily_equal_weight']['mean'],
                'mse_raw':m['mse_raw'],'best_epoch':m.get('best_epoch')}
    for seed in SEEDS:
        results.append({'name':f'a_w30_s{seed}','group':'V7_original',
                        'rankic':float(originals[seed].rankic.mean()),
                        'ic':float(originals[seed].ic.mean()),'mse_raw':None,'best_epoch':None})
    for n in experiments:results.append(row(n,'D_shuffled'))
    if have_ridge:results.append(row('ridge_A30_flat','ridge'))
    if have_mlp:results.extend(row(n,'mlp_flat') for n in mlps)
    pd.DataFrame(results).to_csv(OUT/'v8_individual_results.csv',index=False,encoding='utf-8-sig')
    report=[{'group':'V7_A30_LSTM_ordered','rankic':float(original_mean.rankic.mean()),
             'ic':float(original_mean.ic.mean()),'n_models':3,'n_days':len(original_mean)}]
    compared_daily={'V7_A30_LSTM_ordered':original_mean}
    if count_d==len(SEEDS)*len(REPEATS):
        # Nested per-seed then per-repeat average: exactly 479 unique calendar dates.
        seed_means=[]
        for seed in SEEDS:
            seed_means.append(same_day_metrics([load_day(f'D_A30_s{seed}_r{r}') for r in REPEATS]))
        d_avg=same_day_metrics(seed_means)
        compared_daily['D_A30_shuffle_both']=d_avg
        report.append({'group':'D_A30_shuffle_both','rankic':float(d_avg.rankic.mean()),
                       'ic':float(d_avg.ic.mean()),'n_models':9,'n_days':len(d_avg)})
        compared['Dshuffle_A30_minus_original_A30']=mean_daily_diff(d_avg,original_mean)
        # Individual repeat-by-seed comparisons show whether sign is stable.
        compared['D_repeat_seed_deltas']={str(seed):{
            str(r):mean_daily_diff(load_day(f'D_A30_s{seed}_r{r}'),originals[seed])
            for r in REPEATS} for seed in SEEDS}
    if have_ridge:
        r=load_day('ridge_A30_flat')
        compared_daily['Ridge_A30_flat']=r
        report.append({'group':'Ridge_A30_flat','rankic':float(r.rankic.mean()),
                       'ic':float(r.ic.mean()),'n_models':1,'n_days':len(r)})
        compared['original_A30_minus_Ridge_flat_A30']=mean_daily_diff(original_mean,r)
    if have_mlp:
        m=same_day_metrics([load_day(n) for n in mlps])
        compared_daily['MLP_A30_flat']=m
        report.append({'group':'MLP_A30_flat','rankic':float(m.rankic.mean()),
                       'ic':float(m.ic.mean()),'n_models':3,'n_days':len(m)})
        compared['original_A30_minus_MLP_flat_A30']=mean_daily_diff(original_mean,m)
        if have_ridge:
            compared['MLP_flat_A30_minus_Ridge_flat_A30']=mean_daily_diff(m,load_day('ridge_A30_flat'))
    pd.DataFrame(report).to_csv(OUT/'v8_model_summary.csv',index=False,encoding='utf-8-sig')
    save_json(OUT/'v8_paired_rankic.json',{
        'protocol_sha256':protocol['protocol_sha256'],
        'n_unique_validation_days':len(original_mean),
        'comparisons':compared,
        'note':'EXPLORATORY after validation reuse; HAC lag 10 describes within-period uncertainty, not independent blind confirmation.'})
    print(pd.DataFrame(report).to_string(index=False),flush=True)
    return count_d,have_ridge,have_mlp


def main():
    arg=argparse.ArgumentParser(description='V8 A30 order D and equal-input Ridge/MLP controls')
    arg.add_argument('--experiment',default='all',choices=('order','baselines','all'))
    arg.add_argument('--device',default='auto',choices=('auto','mps','cpu'))
    args=arg.parse_args()
    print('[1/5] Loading V7 A30 common-cohort data and frozen reference metrics...',flush=True)
    device=torch_device(args.device)
    src=v7.load_sources()
    audit=v7.ensure_a_cache(src)
    original_protocol,rows,ytr_raw,yva_raw,label_scaler,dates,originals,original_daily=validate_v7_results(audit,src,device)
    protocol=write_protocol(audit,original_protocol,device)
    xtrain=v7.materialize(np.load(V7/'a_features_scaled.npy',mmap_mode='r'),rows['train'],30)
    xvalid=v7.materialize(np.load(V7/'a_features_scaled.npy',mmap_mode='r'),rows['valid'],30)
    if (xtrain.shape!=(327555,30,8) or xvalid.shape!=(139220,30,8)):
        raise ValueError('Incorrect A30 matrices; cancel experiment')
    ytrain=target_for_train(ytr_raw,label_scaler)
    yvalid=target_for_valid(yva_raw,label_scaler)
    print('A30 input:',xtrain.shape,xvalid.shape,'device=',device,flush=True)
    if args.experiment in ('order','all'):
        do_order(xtrain,xvalid,ytrain,yvalid,dates,yva_raw,label_scaler,protocol,device)
    if args.experiment in ('baselines','all'):
        do_baselines(xtrain,xvalid,ytrain,yvalid,dates,yva_raw,label_scaler,protocol,device)
    print('[4/5] Daily RankIC contrasts and aggregate summaries...',flush=True)
    nd,have_ridge,have_mlp=finish_summary(protocol,originals,original_daily)
    if args.experiment=='all' and (nd!=9 or not have_ridge or not have_mlp):
        raise RuntimeError('V8 was not fully completed; check logged failure')
    print('[5/5] No 2023–2026 outcomes accessed.',flush=True)
    print('V8_A30_CONTROLS_COMPLETE:',OUT,flush=True)
    print('Share: v8_model_summary.csv, v8_individual_results.csv, v8_paired_rankic.json, run_v8.log',flush=True)


if __name__=='__main__':main()
