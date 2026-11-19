#!/usr/bin/env python3
"""V12 final 2025–2026 out-of-time evaluation (ONE SHOT).

--stage audit: lock original model hashes, A8 frozen preprocessing and common
               final stock/date cohort; no final return values or prediction metrics.
--stage evaluate: load frozen models without fitting, evaluate final RankIC and
                  V10-equivalent HYPOTHETICAL adjusted-open economic diagnostics.
Never tune/choose/retrain/rerun after final evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from lstm_research.feature_spec import FEATURE_NAMES
from lstm_research.pilot import anchor_sha256,pred_to_raw,save_json
from lstm_research.sequences import index_for
from lstm_research.lstm_window_v5 import torch_device
from lstm_research.representation_v7 import A_FEATURES,A_NAMES,model_for
from lstm_research.v9_development import (
    MODEL_IDS,SEEDS,file_hash,model_registry,raw_a_to_scaled,audit_overlap,
    cohort_by_a,make_predictions,check_lstm_checkpoint,compare_daily,
)
from lstm_research.v10_economics import (
    COST_SCENARIOS,daily_quintiles,run_phase_cycles,quote_execution_audit,
    assert_calendar_contiguous,
)
from lstm_research.v11_validity import (
    OPAQUE_SOURCE_IDENTIFIERS,read_membership_intervals,membership_info,
    daily_pool_range,align_trade_fields,risk_flags,risk_counts,q5_risk_by_model,
)
from lstm_research.v12_final import (
    START,END,YEARS,PRIMARY_PAIRS,check_final_period,model_results,
    validate_final_predictions,final_quintile_summaries,future_eligibility,
)

V1=ROOT/'output/v1_data'
V2=ROOT/'output/v2_sequences'
V4=ROOT/'output/v4_hypothesis'
V7=ROOT/'output/v7_representations'
V8=ROOT/'output/v8_a30_controls'
V9=ROOT/'output/v9_development'
V10=ROOT/'output/v10_economics'
V11=ROOT/'output/v11_validity'
SRC=Path.home()/'.qlib/qlib_data/cn_data'
OUT=ROOT/'output/v12_final'
PREF=OUT/'v12_preflight.json'
PROT=OUT/'v12_frozen_protocol.json'
AFEAT=OUT/'a_features_final_scaled.npy'
ROWS=OUT/'final_common_rows.npy'
QUOTE=OUT/'quotes_2025_2026.pkl.gz'
CALENDAR=OUT/'calendar_2025_2026.npy'

RESULT_NAMES=(
    'v12_model_summary.csv','v12_yearly_rankic.csv','v12_model_seed_detail.csv',
    'v12_paired_rankic.json','v12_quintile_summary.csv','v12_quintile_daily.csv',
    'v12_phase_summary.csv','v12_phase_cycles.csv',
    'v12_execution_risk_audit.json','v12_final_manifest.json',
    'v12_final_predictions.csv.gz',
)
RESULTS=[OUT/f for f in RESULT_NAMES]
REVIEW=OUT/'v12_review_bundle.zip'
SMALL=(PREF,'v12_model_summary.csv','v12_yearly_rankic.csv',
       'v12_model_seed_detail.csv','v12_paired_rankic.json',
       'v12_quintile_summary.csv','v12_phase_summary.csv',
       'v12_execution_risk_audit.json','v12_final_manifest.json')


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def digest_json(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False,
                    separators=(',',':')).encode('utf-8')).hexdigest()


def label_file_identity(path:Path):
    # Do not read final target values during preflight.
    s=path.stat()
    return {'size_bytes':s.st_size,'mtime_ns':s.st_mtime_ns}


def verify_history():
    """V12 is conditioned on the already-completed V9–V11 protocol and hashes."""
    files=(SRC/'instruments/csi300.txt',SRC/'calendars/day.txt',
           V1/'frozen_factor_panel.pkl.gz',V2/'sample_index.pkl.gz',
           V2/'features_scaled.npy',V2/'labels.npy',V2/'v2_sequence_audit.json',
           V7/'v7_a_scaler_train_only.json',V9/'v9_preflight.json',
           V9/'v9_frozen_protocol.json',V9/'a_features_2022_2024_scaled.npy',
           V9/'v9_model_summary.csv',V11/'v11_verdict.json',V11/'v11_source_manifest.json',
           V10/'v10_result_manifest.json',V10/'v10_protocol.json')
    for f in files:
        if not f.is_file():raise FileNotFoundError(f'Missing frozen research file: {f}')
    # Import V9 runner's proven real-interface checks; it does not score models.
    import importlib.util
    spec=importlib.util.spec_from_file_location('v12_dep_v9_runner',ROOT/'09_development_evaluation.py')
    if not spec or not spec.loader:raise RuntimeError('V9 entry point not importable')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    a7,pr7,scaler,label_scale,registry=module.preconditions()
    v9=read_json(V9/'v9_preflight.json')
    p9=read_json(V9/'v9_frozen_protocol.json')
    verdict=read_json(V11/'v11_verdict.json')
    manifest=read_json(V11/'v11_source_manifest.json')
    p10=read_json(V10/'v10_protocol.json')
    m10=read_json(V10/'v10_result_manifest.json')
    if verdict['status']!='MECHANICAL_CHECKS_PASSED_WITH_UNVERIFIED_ASSUMPTIONS' or verdict['hard_inconsistencies']:
        raise ValueError('V11 reported unresolved hard inconsistencies; final evaluation blocked')
    if not isinstance(verdict.get('independent_historical_CSI300_PIT_certified'),bool) or verdict['independent_historical_CSI300_PIT_certified']:
        raise ValueError('V11 PIT limitations silently changed')
    if v9['model_registry']!=registry or p9['models']!=registry or p10['source_v9_model_registry']!=registry:
        raise ValueError('V9/V10 model registry differs from frozen checkpoints')
    if p9['protocol_sha256']!=digest_json({k:v for k,v in p9.items() if k!='protocol_sha256'}):
        raise ValueError('Frozen V9 protocol altered')
    if p10['protocol_sha256']!=digest_json({k:v for k,v in p10.items() if k!='protocol_sha256'}):
        raise ValueError('Frozen V10 economic protocol altered')
    expected_costs=[{'name':name,'buy_rate':buy,'sell_rate':sell} for name,buy,sell in COST_SCENARIOS]
    if p10['cost_scenarios']!=expected_costs:
        raise ValueError('Frozen V10 transaction cost scenarios changed')
    if v9['A_cache_sha256']!=file_hash(V9/'a_features_2022_2024_scaled.npy'):
        raise ValueError('V9 A cache modified after development evaluation')
    if p10['source_v9_protocol_sha256']!=p9['protocol_sha256'] or m10['protocol_sha256']!=p10['protocol_sha256']:
        raise ValueError('V10→V9 chain changed')
    sources=manifest['source_files_sha256']
    if (sources['csi300_intervals']!=file_hash(SRC/'instruments/csi300.txt') or
        sources['qlib_trading_calendar']!=file_hash(SRC/'calendars/day.txt') or
        sources['v1_frozen_factor_panel']!=file_hash(V1/'frozen_factor_panel.pkl.gz')):
        raise ValueError('Qlib membership/calendar/V1 panel changed since V11 audit')
    if manifest['models_preserved']!=list(MODEL_IDS):
        raise ValueError('V11 model ordering changed')
    if v9['source_A_train_scaler_hash']!=a7['a_scaler_sha256'] or v9['source_B_train_scaler_hash']!=pr7['B_scaler_hash']:
        raise ValueError('Train-only preprocessing no longer matches V9')
    return v9,p9,p10,verdict,scaler,label_scale,registry


def lock_ready():
    if not all(p.is_file() for p in (PREF,PROT,AFEAT,ROWS)):
        raise FileNotFoundError('V12 preflight incomplete; run --stage audit')
    meta=read_json(PREF);prot=read_json(PROT)
    if prot['protocol_sha256']!=digest_json({k:v for k,v in prot.items() if k!='protocol_sha256'}):
        raise ValueError('V12 protocol hash mismatch')
    if (meta['protocol_sha256']!=prot['protocol_sha256'] or
        meta['A_cache_sha256']!=file_hash(AFEAT) or
        meta['cohort_rows_sha256']!=file_hash(ROWS)):
        raise ValueError('V12 files changed after preflight')
    return meta,prot


def audit():
    print('[1/7] Verify V9–V11 provenance, four checkpoint hashes and research restrictions...',flush=True)
    v9,p9,p10,verdict,scaler,target,registry=verify_history()
    preexisting=(PREF,PROT,AFEAT,ROWS,*RESULTS,REVIEW)
    if any(x.exists() for x in preexisting):
        if all(x.is_file() for x in (PREF,PROT,AFEAT,ROWS)) and not any(x.exists() for x in RESULTS):
            lock_ready()
            print('V12_PREFLIGHT_ALREADY_FROZEN: existing audit preserved; use --stage evaluate.',flush=True)
            return
        if all(x.is_file() for x in (PREF,PROT,AFEAT,ROWS,*RESULTS,REVIEW)):
            print('V12_ALREADY_EVALUATED: results preserved; do not rerun.',flush=True)
            return
        raise RuntimeError('V12 partial outputs exist. Inspect files; no silent overwrite of final period.')
    print('[2/7] Build frozen B30 eligible FINAL anchors with t+6 cutoff...',flush=True)
    index=pd.read_pickle(V2/'sample_index.pkl.gz',compression='gzip')
    b30=index_for(index,'final_holdout',30)
    check_final_period(b30)
    print(f'  B30 final anchors={len(b30):,}, days={b30.datetime.nunique()}, '
          f'first={b30.datetime.min().date()}, last={b30.datetime.max().date()}',flush=True)

    print('[3/7] Fetch only A8 DAILY FEATURES 2024-10 through 2026-11 (NO target evaluation)...',flush=True)
    import qlib
    from qlib.constant import REG_CN
    from qlib.data import D
    qlib.init(provider_uri=str(SRC),region=REG_CN)
    query=D.features(D.instruments(market='csi300'),[f.expression for f in A_FEATURES],
                     start_time='2024-10-01',end_time=str(END.date()),freq='day',disk_cache=0)
    if query.shape[1]!=len(A_NAMES) or query.empty:
        raise ValueError('Qlib returned invalid A-feature data')
    query.columns=list(A_NAMES)
    query=query.reset_index()
    if not {'instrument','datetime'}.issubset(query.columns):
        raise ValueError('Unexpected Qlib A feature keys')
    query['instrument']=query.instrument.astype(str)
    query['datetime']=pd.to_datetime(query.datetime)
    if query[['instrument','datetime']].duplicated().any():
        raise ValueError('Duplicate Qlib A feature stock/dates')
    qdf=query.set_index(['instrument','datetime']).sort_index()
    panel=pd.read_pickle(V1/'frozen_factor_panel.pkl.gz',compression='gzip')
    v2meta=read_json(V2/'v2_sequence_audit.json')
    if len(panel)!=v2meta['n_panel_rows']:
        raise ValueError('V1 panel shape differs from frozen V2')
    panel['datetime']=pd.to_datetime(panel.datetime)
    context=panel.datetime.between('2024-10-01',str(END.date())).to_numpy(bool)
    positions=np.flatnonzero(context)
    keys=pd.MultiIndex.from_frame(panel.iloc[positions][['instrument','datetime']])
    raw=np.full((len(panel),8),np.nan,dtype=np.float32)
    raw[positions]=qdf.reindex(keys).to_numpy(dtype=np.float32)
    scaled=raw_a_to_scaled(raw,scaler)
    print(f'  A Qlib rows={len(query):,}; frozen panel rows={len(panel):,}',flush=True)

    print('[4/7] Cross-check A/B feature formulas and A scaler against V9 2024 overlap...',flush=True)
    overlap=audit_overlap(raw,panel,b30.panel_row.to_numpy(np.int64),FEATURE_NAMES)
    old=np.load(V9/'a_features_2022_2024_scaled.npy',mmap_mode='r')
    if old.shape!=scaled.shape:
        raise ValueError('V9 A cache shape mismatch')
    past=np.flatnonzero(panel.datetime.between('2024-11-01','2024-12-31').to_numpy(bool))
    past=past[np.isfinite(raw[past]).all(axis=1)]
    if len(past)<500:
        raise ValueError('Insufficient 2024 V9/V12 scaler consistency overlap')
    err=float(np.max(np.abs(scaled[past]-old[past])))
    if err>2e-5:
        raise ValueError(f'V9/V12 A-scaler or Qlib-data drift: {err}')
    common=cohort_by_a(b30,np.isfinite(raw).all(axis=1))
    check_final_period(common)
    if len(common)<100000:raise ValueError('Too many A-window missing samples')
    rows=common.panel_row.to_numpy(dtype=np.int64)
    offsets=np.arange(29,-1,-1,dtype=np.int64)
    for k in range(0,len(rows),4096):
        if not np.isfinite(scaled[rows[k:k+4096,None]-offsets]).all():
            raise ValueError('Nonfinite final A window')
    eligibility=future_eligibility(panel,b30.panel_row.to_numpy(np.int64),rows)
    print(f'  A/B shared final anchors={len(common):,}, A exclusions={len(b30)-len(common):,}; '
          f'overlap max={err:.2g}',flush=True)

    print('[5/7] Audit FINAL historical membership against local file, no model ranking...',flush=True)
    intervals,meta=read_membership_intervals(SRC/'instruments/csi300.txt',end=str(END.date()))
    member,_,_=membership_info(common[['instrument','datetime']],intervals)
    if not member.all():
        raise ValueError(f'Final signals outside local effective CSI300 membership: {int((~member).sum())}')
    strange=intervals[intervals.instrument.isin(OPAQUE_SOURCE_IDENTIFIERS)]
    active=strange[(strange.start<=END)&(strange.end>=START)]
    if not active.empty:
        raise ValueError('Unresolved nonstandard CSI300 source identifiers overlap FINAL period')
    # Compare local Qlib provider with its own file on 5 predetermined final dates.
    ds=pd.DatetimeIndex(common.datetime.unique()).sort_values()
    selected=[ds[0],ds[len(ds)//4],ds[len(ds)//2],ds[3*len(ds)//4],ds[-1]]
    inst=D.instruments(market='csi300')
    snaps=[]
    for date in selected:
        d=pd.Timestamp(date)
        expected=set(intervals.loc[(intervals.start<=d)&(intervals.end>=d),'instrument'])
        retrieved=set(map(str.upper,D.list_instruments(instruments=inst,
            start_time=str(d.date()),end_time=str(d.date()),as_list=True)))
        if expected!=retrieved:
            raise ValueError('Qlib source/provider historical constituent inconsistency')
        snaps.append({'date':str(d.date()),'n_local':len(expected),
                      'n_provider':len(retrieved),'symmetric_difference':0})

    print('[6/7] Freeze sample and preprocessing hashes before any final prediction metrics...',flush=True)
    OUT.mkdir(parents=True,exist_ok=True)
    np.save(AFEAT,scaled)
    np.save(ROWS,rows)
    proto={
        'stage':'V12 FINAL once-only 2025-2026 (2026 PARTIAL through label cutoff)',
        'source':'A-share LSTM research project V1–V11 ONLY',
        'models_order':list(MODEL_IDS),'model_registry':registry,
        'training':'2016–2020 original frozen checkpoints',
        'validation':'2021–2022 previously used in model design',
        'development':'2023–2024 frozen external-period evaluation and idealized economics',
        'final':'2025–2026-11-02; t+6 must not exceed 2026-11-02',
        'A8_spec':[vars(f) for f in A_FEATURES],
        'B16_feature_order':list(FEATURE_NAMES),
        'A_scaler_hash':v9['source_A_train_scaler_hash'],
        'B_scaler_hash':v9['source_B_train_scaler_hash'],
        'label_scaler':target,
        'metric_primary':'equal-weight per-day cross-sectional Spearman RankIC, average over 3 seed model scores',
        'metrics_secondary':['Pearson IC','raw MSE','2025 separately','2026 partial-year separately','seed SD'],
        'primary_pairs':[f'{a}_minus_{b}' for a,b in PRIMARY_PAIRS],
        'inference':'paired daily HAC lag 10, exploratory; no across-study multiple-testing correction',
        'economic_protocol':'identical to V10: quintiles, 5d adjusted-open returns, 5 independent staggered sleeves, 0/5/10/20 bp per side',
        'economic_lstm_prediction':'mean of three raw seed model predictions, not RankIC of averaged models',
        'years_and_annualization':'2026 is truncated YTD; annualized phase metrics are diagnostic only',
        'no_fit_tune_retrain_or_reselect':True,'no_output_overwrite':True,
        'historical_stock_pool_PIT_unverified':True,
        'future_label_observability_selection_unverified':True,
        'real_opening_auction_fills_unverified':True,
        'adjusted_open_prices_are_not_tradable_quotes':True,
        'unadjusted_lot_tax_slippage_and_limit_lock_unmodelled':True,
        'Q1_short_side_is_nontradable_spread_diagnostic':True,
        'equal_weight_universe_is_not_official_index':True,
        'no_holdout_informed_revision':True,
        'v9_protocol_sha256':p9['protocol_sha256'],
        'v10_protocol_sha256':p10['protocol_sha256'],
        'frozen_rows_hash':anchor_sha256(common),
        'date_range':{'earliest':str(ds.min().date()),'latest':str(ds.max().date()),
                      'n_days':len(ds),'n_rows':len(common)},
    }
    proto['protocol_sha256']=digest_json(proto)
    save_json(PROT,proto)
    spec={
        'status':'PRE-EVALUATION LOCKED; no 2025-2026 RankIC / P&L calculated',
        'protocol_sha256':proto['protocol_sha256'],
        'source_model_registry':registry,
        'source_v9_protocol':p9['protocol_sha256'],
        'source_v10_protocol':p10['protocol_sha256'],
        'source_v11_verdict':verdict['status'],
        'source_qlib_instruments_sha256':file_hash(SRC/'instruments/csi300.txt'),
        'source_qlib_calendar_sha256':file_hash(SRC/'calendars/day.txt'),
        'cohort_fingerprint':anchor_sha256(common),
        'final_date_range':proto['date_range'],
        'B_eligible30_n':len(b30),'A_B_common_n':len(common),'A_exclusions':len(b30)-len(common),
        'A_overlap_2024_max_error':err,'A_B_overlap_checks':overlap,
        '2025_2026_future_availability_diagnostic':eligibility,
        'provider_membership_snapshots':snaps,
        'source_anomalies_preserved':meta['nonstandard_source_identifiers'],
        'A_query_start':'2024-10-01','A_query_end':str(END.date()),
        'A_cache_sha256':file_hash(AFEAT),'cohort_rows_sha256':file_hash(ROWS),
        'label_file_identity':label_file_identity(V2/'labels.npy'),
        'B_feature_cache_sha256':file_hash(V2/'features_scaled.npy'),
        'V2_index_file_sha256':file_hash(V2/'sample_index.pkl.gz'),
        'statistical_knowledge':'NONE in this V12 audit. Ex-ante constraints include target finite flags from frozen V1/V2.',
        'limitations_inherited_from_V11':{
            'PIT_independently_certified':False,'actual_trade_executions_certified':False,
            'future_label_selection_proven_unbiased':False},
    }
    save_json(PREF,spec)
    print('[7/7] Final preflight locked; no model score or portfolio return computed.',flush=True)
    print('V12_PREFLIGHT_COMPLETE:',OUT,flush=True)


def frozen_final():
    v9,p9,p10,verdict,scaler,target,reg=verify_history()
    meta,prot=lock_ready()
    if meta['source_model_registry']!=reg or prot['model_registry']!=reg:
        raise ValueError('Final model checkpoints changed')
    if (meta['source_qlib_instruments_sha256']!=file_hash(SRC/'instruments/csi300.txt') or
        meta['source_qlib_calendar_sha256']!=file_hash(SRC/'calendars/day.txt') or
        meta['label_file_identity']!=label_file_identity(V2/'labels.npy') or
        meta['B_feature_cache_sha256']!=file_hash(V2/'features_scaled.npy') or
        meta['V2_index_file_sha256']!=file_hash(V2/'sample_index.pkl.gz')):
        raise ValueError('Source Qlib files or frozen target file changed after audit')
    rows=np.load(ROWS)
    if rows.ndim!=1 or len(rows)!=meta['A_B_common_n'] or not np.array_equal(rows,np.sort(rows)):
        raise ValueError('Final frozen panel row ordering changed')
    idx=pd.read_pickle(V2/'sample_index.pkl.gz',compression='gzip')
    b=index_for(idx,'final_holdout',30)
    sub=b.set_index('panel_row').loc[rows].reset_index()
    check_final_period(sub)
    if anchor_sha256(sub)!=meta['cohort_fingerprint'] or anchor_sha256(sub)!=prot['frozen_rows_hash']:
        raise ValueError('Final stock/date membership or order differs from preflight')
    a=np.load(AFEAT,mmap_mode='r')
    bscaled=np.load(V2/'features_scaled.npy',mmap_mode='r')
    if a.shape!=(861570,8) or bscaled.shape!=(861570,16):
        raise ValueError('Final feature matrices differ from V1–V9 shape')
    return meta,prot,sub,a,bscaled,target


def final_eval(device):
    print('[1/8] Confirm locked V12 preflight, original model hashes and final target boundaries...',flush=True)
    if all(p.is_file() for p in (*RESULTS,REVIEW)):
        print('V12_FINAL_ALREADY_EVALUATED: complete outputs preserved; no rerun.',flush=True)
        return
    if any(p.exists() for p in (*RESULTS,REVIEW)):
        raise RuntimeError('PARTIAL FINAL EVALUATION EXISTS. Inspect/recover results; no silent overwrite.')
    meta,prot,sub,aa,bb,label_scaler=frozen_final()
    rows=sub.panel_row.to_numpy(np.int64)
    # No final outcome was indexed or scored during V12 audit; first read HERE.
    raw_y=np.asarray(np.load(V2/'labels.npy',mmap_mode='r')[rows],dtype=np.float64)
    if not np.isfinite(raw_y).all():raise ValueError('Nonfinite final label on frozen cohort')
    dates=sub.datetime.to_numpy()
    print(f'  FINAL n={len(rows):,}, days={sub.datetime.nunique()}, '
          f'last signal={sub.datetime.max().date()}, device={device}',flush=True)
    print('[2/8] Frozen B1 and A30 Ridge inference; no fitting...',flush=True)
    import joblib
    b1=joblib.load(V4/'ridge_1_model.joblib')
    a30=joblib.load(V8/'checkpoints/ridge_A30_flat.pt')
    preds_scaled={
        'B1_Ridge':[make_predictions(b1,bb,rows,kind='ridge',window=1)],
        'A30_Ridge':[make_predictions(a30,aa,rows,kind='ridge',window=30)],
    }
    print('[3/8] Frozen A30 and B30 LSTM × 3 seeds, original weights unchanged...',flush=True)
    import torch
    for name,prefix,ch,matrix in [('A30_LSTM','a',8,aa),('B30_LSTM','b',16,bb)]:
        pp=[]
        for seed in SEEDS:
            path=V7/f'checkpoints/{prefix}_w30_s{seed}.pt'
            try:
                state=torch.load(path,map_location='cpu',weights_only=True)
            except TypeError:
                state=torch.load(path,map_location='cpu')
            check_lstm_checkpoint(state,ch,seed)
            model=model_for(ch)
            model.load_state_dict(state['state_dict'],strict=True)
            model.eval()
            pp.append(make_predictions(model,matrix,rows,kind='lstm',window=30,device=device))
            del state,model
            print(f'   {name}: predicted frozen seed {seed}',flush=True)
        preds_scaled[name]=pp
    print('[4/8] Final daily RankIC, IC, MSE, year splits and paired HAC contrasts...',flush=True)
    prior=pd.read_csv(V9/'v9_model_summary.csv').set_index('model')
    if set(prior.index)!=set(MODEL_IDS):raise ValueError('Frozen V9 model comparison group differs')
    summaries=[];seeds=[];years=[];daily={};ensemble={}
    for name in MODEL_IDS:
        preds=[pred_to_raw(p,label_scaler) for p in preds_scaled[name]]
        report,seed_detail,df,yy=model_results(name,dates,raw_y,preds,
                                                SEEDS if len(preds)>1 else ('fixed',))
        report['development_rankic_2023_2024']=float(prior.loc[name,'rankic_seed_mean'])
        report['final_minus_development']=report['rankic_seed_mean']-report['development_rankic_2023_2024']
        summaries.append(report);seeds.extend(seed_detail);years.extend(yy);daily[name]=df
        ensemble[name]=np.mean(np.stack(preds,axis=0),axis=0)
        print(f'   {name}: FINAL RankIC={report["rankic_seed_mean"]:+.6f} '
              f'2025={report["rankic_2025"]:+.6f} '
              f'2026-YTD={report["rankic_2026"]:+.6f}',flush=True)
    pairs={f'{a}_minus_{b}':compare_daily(daily[a],daily[b]) for a,b in PRIMARY_PAIRS}
    pred=pd.DataFrame({'instrument':sub.instrument.to_numpy(),
                       'datetime':sub.datetime.to_numpy(), 'label_raw':raw_y})
    for name in MODEL_IDS:pred[name+'_mean_raw_pred']=ensemble[name]
    pred=validate_final_predictions(pred,expected_rows=len(rows),expected_days=sub.datetime.nunique())
    print('[5/8] Retrieve final-period adjusted market quotes, audit t+1/t+6 labels...',flush=True)
    import qlib
    from qlib.constant import REG_CN
    from qlib.data import D
    qlib.init(provider_uri=str(SRC),region=REG_CN)
    codes=sorted(pred.instrument.unique())
    quote=D.features(codes,['$open','$close','$volume','$factor'],
                     start_time=str(START.date()),end_time=str(END.date()),freq='day',disk_cache=0)
    if quote.empty or quote.shape[1]!=4:
        raise ValueError('Empty/changed final Qlib OHLC/volume/factor quote format')
    quote.columns=['open_adj','close_adj','volume_adj','factor']
    quote=quote.reset_index()
    quote.instrument=quote.instrument.astype(str)
    quote.datetime=pd.to_datetime(quote.datetime)
    if quote[['instrument','datetime']].duplicated().any():
        raise ValueError('Duplicate final quote bars')
    cal=pd.DatetimeIndex(D.calendar(start_time=str(START.date()),end_time=str(END.date()),freq='day'))
    if len(cal)<350:raise ValueError('Incomplete final trading calendar')
    assert_calendar_contiguous(pred.datetime.unique(),cal)
    exec_audit=quote_execution_audit(pred,quote,cal)
    if (exec_audit['label_quote_comparable_count']!=len(pred) or
            exec_audit['missing_or_invalid_entry_open'] or
            exec_audit['missing_or_invalid_exit_open']):
        raise ValueError('Final labels cannot be matched to all entry/exit opens')
    trade=align_trade_fields(pred,quote,cal)
    flags=risk_flags(trade)
    risk={'period':'2025-2026_PARTIAL','n_signals':len(pred),
          'all_samples':risk_counts(flags,np.ones(len(pred),bool)),
          'Q5_by_model':q5_risk_by_model(pred,flags),
          'open_quote_consistency':exec_audit,
          '2026_incomplete_year':True,
          'price_adjustment':'Qlib adjusted $open; not exchange executable prints',
          'independent_price_limit_ST_halt_flags_available':False,
          'real_auction_fill_certified':False,'actual_historical_fees_certified':False,
          'future_exit_quote_is_not_an_exante_eligibility_filter':True,
          'same_V11_data_limitations_apply':True}
    print('[6/8] V10-matched quintiles, daily 5d theoretical excess, 5 staggered phase sleeves...',flush=True)
    qdaily,baskets=daily_quintiles(pred)
    qsummary=final_quintile_summaries(qdaily)
    phases,cycles=run_phase_cycles(qdaily,baskets,cal)
    if (len(qdaily)!=len(MODEL_IDS)*pred.datetime.nunique() or
            len(phases)!=len(MODEL_IDS)*3*5*len(COST_SCENARIOS)):
        raise ValueError('Final economic output cardinality mismatch')

    print('[7/8] Commit all final one-shot results; preserve historical model files...',flush=True)
    OUT.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(summaries).to_csv(OUT/RESULT_NAMES[0],index=False,encoding='utf-8-sig')
    pd.DataFrame(years).to_csv(OUT/RESULT_NAMES[1],index=False,encoding='utf-8-sig')
    pd.DataFrame(seeds).to_csv(OUT/RESULT_NAMES[2],index=False,encoding='utf-8-sig')
    save_json(OUT/RESULT_NAMES[3],{
        'protocol_sha256':prot['protocol_sha256'],'n_unique_days':int(pred.datetime.nunique()),
        'primary_predeclared_comparisons':pairs,
        'inference_warning':'Date-wise HAC 10-lag exploratory; model selection preceded final evaluation. Prior model-development choices are not evidence of universal superiority.'})
    qsummary.to_csv(OUT/RESULT_NAMES[4],index=False,encoding='utf-8-sig')
    qdaily.to_csv(OUT/RESULT_NAMES[5],index=False,encoding='utf-8-sig')
    phases.to_csv(OUT/RESULT_NAMES[6],index=False,encoding='utf-8-sig')
    cycles.to_csv(OUT/RESULT_NAMES[7],index=False,encoding='utf-8-sig')
    save_json(OUT/RESULT_NAMES[8],risk)
    pred.to_csv(OUT/RESULT_NAMES[10],index=False,compression='gzip')
    quote.to_pickle(QUOTE,compression='gzip')
    np.save(CALENDAR,cal.to_numpy(dtype='datetime64[ns]'))
    manifest={
        'status':'V12 FINAL once-only evaluation COMPLETE; no rerunning to reselect or tune',
        'preflight_protocol_sha256':prot['protocol_sha256'],
        'n_models':len(MODEL_IDS),'final_rows':len(pred),'final_days':int(pred.datetime.nunique()),
        'last_signal_date':str(pred.datetime.max().date()),
        '2026_is_partial':True,
        'models_untouched':list(MODEL_IDS),
        'returns':'hypothetical adjusted-open 5-trading-day holding results; NOT real executable PnL',
        'cost_scenarios':'symmetric 0, 5, 10, 20 bp per traded side; NOT actual A-share fee schedule',
        'five_sleeves_overlap':'do not treat the 5 phases as independent samples',
        'source_quote_sha256':file_hash(QUOTE),
        'source_calendar_sha256':file_hash(CALENDAR),
        'saved_outputs_sha256':{n:file_hash(OUT/n) for n in RESULT_NAMES[:-2]+(RESULT_NAMES[-1],)},
        'V11_limitations_retained':['Historical CSI300 constituent publication/PIT unverified',
           'Future label availability causes selection conditioning',
           'Adjusted opens not executable prices; ST/halts/limit lock/auction fills unverified',
           'No actual lot/tax/fees/market impact modeling'],
        'no_retuning_after_final':True,
    }
    save_json(OUT/RESULT_NAMES[9],manifest)
    # Small upload bundle: full daily predictions and cycle files remain local.
    with zipfile.ZipFile(REVIEW,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=7) as z:
        for f in SMALL:
            filename=f.name if isinstance(f,Path) else f
            path=f if isinstance(f,Path) else OUT/f
            z.write(path,arcname=filename)
    print('[8/8] FINAL completed and immutable; no subsequent tuning or refitting.',flush=True)
    print('V12_FINAL_EVALUATION_COMPLETE:',OUT,flush=True)
    print('Upload output/v12_final/v12_review_bundle.zip and run_v12.log',flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',required=True,choices=['audit','evaluate'])
    p.add_argument('--device',choices=['auto','cpu','mps'],default='auto')
    a=p.parse_args()
    if a.stage=='audit':audit()
    else:final_eval(torch_device(a.device))


if __name__=='__main__':main()
