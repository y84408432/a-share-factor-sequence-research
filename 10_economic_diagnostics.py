#!/usr/bin/env python3
"""V10: locked 2023-24 economic diagnostics using V9 frozen predictions.

--stage audit: quote/label execution-proxy audit and immutable protocol.
--stage evaluate: 5-day matched quintile and 5 staggered sleeves (no refits).
This is NOT a certified executable A-share backtest.
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
from lstm_research.pilot import anchor_sha256, save_json
from lstm_research.sequences import index_for
from lstm_research.v9_development import file_hash, model_registry
from lstm_research.v10_economics import (
    MODEL_IDS,PRED_COLS,COST_SCENARIOS,validate_predictions,daily_quintiles,
    quintile_summaries,run_phase_cycles,quote_execution_audit,
)

V1=ROOT/'output/v1_data'
V2=ROOT/'output/v2_sequences'
V9=ROOT/'output/v9_development'
OUT=ROOT/'output/v10_economics'
PRED=V9/'v9_development_predictions.csv.gz'
QUOTE=OUT/'quotes_2023_2024.pkl.gz'
CALENDAR=OUT/'calendar_2023_2024.npy'
META=OUT/'v10_preflight.json'
PROTOCOL=OUT/'v10_protocol.json'
EXPECTED_ROWS=141415
EXPECTED_DAYS=478
OUTPUTS=(OUT/'v10_quintile_summary.csv',OUT/'v10_quintile_daily.csv',
         OUT/'v10_phase_summary.csv',OUT/'v10_phase_cycles.csv',
         OUT/'v10_execution_audit.json',OUT/'v10_result_manifest.json')


def json_read(fp):return json.loads(fp.read_text(encoding='utf-8'))


def json_hash(obj):
    s=json.dumps(obj,sort_keys=True,ensure_ascii=False,separators=(',',':'))
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def frozen_v9():
    required=(PRED,V9/'v9_preflight.json',V9/'v9_frozen_protocol.json',
              V9/'development_common_rows.npy',V9/'v9_model_summary.csv',
              V2/'sample_index.pkl.gz',V2/'labels.npy')
    for fp in required:
        if not fp.is_file():raise FileNotFoundError(f'Missing frozen V9 source: {fp}')
    meta=json_read(V9/'v9_preflight.json')
    prot=json_read(V9/'v9_frozen_protocol.json')
    if (meta['model_ids']!=list(MODEL_IDS) or
            prot['models']!=meta['model_registry'] or
            meta['model_registry']!=model_registry(ROOT)):
        raise ValueError('V9 frozen model registry has changed')
    if prot['protocol_sha256']!=json_hash({k:v for k,v in prot.items() if k!='protocol_sha256'}):
        raise ValueError('Frozen V9 protocol hash mismatch')
    if (meta['A_B_common_n']!=EXPECTED_ROWS or
            meta['2023_2024_dates']['n']!=EXPECTED_DAYS):
        raise ValueError('V9 expected sample cohort changed')
    rows=np.load(V9/'development_common_rows.npy')
    if len(rows)!=EXPECTED_ROWS or file_hash(V9/'development_common_rows.npy')!=meta['cohort_rows_file_sha256']:
        raise ValueError('V9 frozen row index changed')
    index=pd.read_pickle(V2/'sample_index.pkl.gz',compression='gzip')
    dev=index_for(index,'development',30).set_index('panel_row').loc[rows].reset_index()
    if anchor_sha256(dev)!=meta['2023_2024_anchor_sha256']:
        raise ValueError('V9 cohort fingerprint changed')
    pred=validate_predictions(pd.read_csv(PRED),expected_rows=EXPECTED_ROWS,expected_days=EXPECTED_DAYS)
    ref=dev[['instrument','datetime']].copy()
    ref['datetime']=pd.to_datetime(ref['datetime'])
    # V9 predictions are stock/date aligned with the exact frozen anchors.
    observed=pred[['instrument','datetime']].sort_values(['instrument','datetime']).reset_index(drop=True)
    intended=ref.sort_values(['instrument','datetime']).reset_index(drop=True)
    if not observed.equals(intended):raise ValueError('V9 prediction sample dates/instruments differ from frozen preflight')
    labels=np.load(V2/'labels.npy',mmap_mode='r')[rows]
    aligned=ref.assign(label_raw=labels).sort_values(['instrument','datetime']).reset_index(drop=True)
    matched=pred[['instrument','datetime','label_raw']].sort_values(['instrument','datetime']).reset_index(drop=True)
    if not np.allclose(aligned.label_raw.to_numpy(),matched.label_raw.to_numpy(),rtol=0,atol=1e-7):
        raise ValueError('V9 prediction labels do not match frozen V2 labels')
    return pred,meta,prot


def audit():
    print('[1/5] Verify V9 predictions, model checkpoints, and exact cohort...',flush=True)
    frame,base,protocol=frozen_v9()
    if any(p.exists() for p in (META,PROTOCOL,QUOTE,CALENDAR)):
        raise RuntimeError('V10 preflight already exists or partly exists; do not silently overwrite')
    print(f'  frozen 2023-2024 cohort: {len(frame):,} stock-dates, {frame.datetime.nunique()} trading days',flush=True)
    print('[2/5] Retrieve historical adjusted OHLC, factor, volume for development instruments...',flush=True)
    import qlib
    from qlib.constant import REG_CN
    from qlib.data import D
    source=Path.home()/'.qlib/qlib_data/cn_data'
    if not (source/'instruments/csi300.txt').is_file():
        raise FileNotFoundError('Local Qlib cn_data missing')
    qlib.init(provider_uri=str(source),region=REG_CN)
    codes=sorted(frame.instrument.unique())
    fields=['$open','$close','$volume','$factor']
    raw=D.features(codes,fields,start_time='2023-01-01',end_time='2024-12-31',freq='day',disk_cache=0)
    if raw.shape[1]!=4:raise ValueError('Market quote fields count changed')
    raw.columns=['open_adj','close_adj','volume_adj','factor']
    raw=raw.reset_index()
    raw['instrument']=raw.instrument.astype(str)
    raw['datetime']=pd.to_datetime(raw.datetime)
    if raw[['instrument','datetime']].duplicated().any():raise ValueError('Duplicate quote rows')
    if raw.empty:raise ValueError('Empty Qlib quote panel')
    calendar=pd.DatetimeIndex(D.calendar(start_time='2023-01-01',end_time='2024-12-31',freq='day'))
    print(f'  quotes={len(raw):,}, covered instruments={raw.instrument.nunique()}, market days={len(calendar)}',flush=True)
    print('[3/5] Check t+1/t+6 execution-bar coverage and label consistency...',flush=True)
    quote_audit=quote_execution_audit(frame,raw,calendar)
    print('  matching opens:',quote_audit['label_quote_comparable_count'],
          '; max abs label mismatch:',quote_audit['label_quote_max_abs_difference'],flush=True)
    print('[4/5] Freeze quote snapshot, cost assumptions and economic evaluation protocol...',flush=True)
    OUT.mkdir(parents=True,exist_ok=True)
    raw.to_pickle(QUOTE,compression='gzip')
    np.save(CALENDAR,calendar.to_numpy(dtype='datetime64[ns]'))
    config={
      'stage':'V10 exploratory 2023-2024 economic diagnostics ONLY',
      'models':list(MODEL_IDS),
      'source_v9_protocol_sha256':protocol['protocol_sha256'],
      'source_v9_cohort_sha256':base['2023_2024_anchor_sha256'],
      'source_v9_model_registry':base['model_registry'],
      'prediction_file_hash':file_hash(PRED),
      'price_file_hash':file_hash(QUOTE),
      'calendar_file_hash':file_hash(CALENDAR),
      'n_signals':len(frame),'n_days':int(frame.datetime.nunique()),
      'signal_clock':'after close(t)',
      'buy_price':'Qlib adjusted $open(t+1), HYPOTHETICAL only',
      'sell_price':'Qlib adjusted $open(t+6), HYPOTHETICAL only',
      'prediction_horizon_trading_day_intervals':5,
      'cross_section_buckets':5,
      'buckets':'equal-name counts, ascending predicted return, stable instrument tie-break',
      'staggered_sleeves':5,
      'sleeve_phase':'global Qlib trading calendar index modulo 5, rebalance every five trading sessions',
      'sleeve_groups':['Q5','Q1','universe'],
      'sleeve_weights':'equal weight on every new t+1 open, post-holding-price-drift turnover',
      'cost_scenarios':[{'name':name,'buy_rate':buy,'sell_rate':sell} for name,buy,sell in COST_SCENARIOS],
      'cost_assumptions_not_historical_fees':True,
      'primary_diagnostics':['daily mean 5d Q5-Q1 spread','daily mean 5d Q5-universe spread',
                             'phase gross/net growth','turnover'],
      'horizon_overlap_inference':'date-wise Newey-West HAC lag 10; exploratory',
      'lstm_predictions':'V9 mean of three raw model predictions (ensemble), NOT average per-seed RankIC',
      'no_model_refit_or_new_selection':True,
      'no_2025_2026_read':True,
      'limitations':[
          'Not a filled-order backtest: opening auction execution, price limits, suspension exits, lot size, impact, taxes and liquidity unmodeled.',
          'Qlib prices are split-adjusted; cannot treat quoted adjusted opens as actual trade prices.',
          'No hindsight filtering based on future t+6 quote availability.',
          'Q1 short basket / top-bottom spread is non-tradable A-share short-side diagnostic.',
          'The equal-weight eligible-universe comparison is NOT the official CSI300 index total return.',
          'Five staggered sleeves overlap in calendar time; do not count them as five independent samples.',
          'Multiple-model comparisons exploratory after prior validation/development research.',
      ],
    }
    config['protocol_sha256']=json_hash(config)
    save_json(PROTOCOL,config)
    save_json(META,{
        'status':'V10_PREFLIGHT_ONLY; no strategy signal results generated',
        'protocol_sha256':config['protocol_sha256'],
        'source_prediction_hash':config['prediction_file_hash'],
        'quote_snapshot_hash':config['price_file_hash'],
        'calendar_hash':config['calendar_file_hash'],
        'quote_audit':quote_audit,
        'source_v9_cohort_sha256':base['2023_2024_anchor_sha256'],
    })
    print('[5/5] Preflight complete; no signal ranking or strategy return computed yet.',flush=True)
    print('V10_PREFLIGHT_COMPLETE:',OUT,flush=True)


def evaluate():
    print('[1/6] Verify immutable V10 audit, frozen V9 files, and price hashes...',flush=True)
    if all(p.is_file() for p in OUTPUTS):
        print('V10_ALREADY_EVALUATED: preserved outputs; no rerun.',flush=True)
        return
    if any(p.exists() for p in OUTPUTS):
        raise RuntimeError('Partial V10 outputs exist; inspect before proceeding; no overwrites')
    if not all(p.is_file() for p in (META,PROTOCOL,QUOTE,CALENDAR)):
        raise FileNotFoundError('Run --stage audit first')
    frame,base,base_protocol=frozen_v9()
    meta=json_read(META);protocol=json_read(PROTOCOL)
    if (meta['protocol_sha256']!=protocol['protocol_sha256'] or
            json_hash({k:v for k,v in protocol.items() if k!='protocol_sha256'})!=protocol['protocol_sha256'] or
            protocol['source_v9_cohort_sha256']!=base['2023_2024_anchor_sha256'] or
            protocol['source_v9_protocol_sha256']!=base_protocol['protocol_sha256'] or
            protocol['source_v9_model_registry']!=base['model_registry'] or
            protocol['prediction_file_hash']!=file_hash(PRED) or
            protocol['price_file_hash']!=file_hash(QUOTE) or
            protocol['calendar_file_hash']!=file_hash(CALENDAR)):
        raise ValueError('V10 preflight or prediction/quote provenance changed')
    q=pd.read_pickle(QUOTE,compression='gzip')
    calendar=pd.DatetimeIndex(np.load(CALENDAR))
    print('[2/6] Re-audit adjusted-open quotes and verify 5-day label alignment...',flush=True)
    execution=quote_execution_audit(frame,q,calendar)
    if execution!=meta['quote_audit']:
        raise ValueError('Quote execution integrity changed between stages')
    print('[3/6] Construct equal-count daily prediction quintiles for four fixed models...',flush=True)
    daily,baskets=daily_quintiles(frame)
    if len(daily)!=EXPECTED_DAYS*len(MODEL_IDS):
        raise ValueError('Unexpected daily quintile length')
    quintiles=quintile_summaries(daily)
    print('[4/6] Build FIVE independent-stagger schedules (non-independent market outcomes)...',flush=True)
    phases,cycles=run_phase_cycles(daily,baskets,calendar)
    if len(phases)!=len(MODEL_IDS)*3*5*len(COST_SCENARIOS):
        raise ValueError('Unexpected phase scenario count')
    print('[5/6] Write only diagnostic economic outputs; no trade fills assumed...',flush=True)
    quintiles.to_csv(OUTPUTS[0],index=False,encoding='utf-8-sig')
    daily.to_csv(OUTPUTS[1],index=False,encoding='utf-8-sig')
    phases.to_csv(OUTPUTS[2],index=False,encoding='utf-8-sig')
    cycles.to_csv(OUTPUTS[3],index=False,encoding='utf-8-sig')
    save_json(OUTPUTS[4],execution)
    save_json(OUTPUTS[5],{
        'status':'V10 exploratory HYPOTHETICAL adjusted-open economic diagnostic; NOT certified executable backtest',
        'protocol_sha256':protocol['protocol_sha256'],
        'n_models':len(MODEL_IDS),'n_signal_dates':EXPECTED_DAYS,
        'n_quintile_summary_rows':len(quintiles),'n_phase_summary_rows':len(phases),
        'scenario_sensitivity':'Gross and symmetrical per-traded-side friction; not actual China A-share historical fees',
        'final_holdout_2025_2026':'NOT accessed',
        'files_sha256':{p.name:file_hash(p) for p in OUTPUTS[:-1]},
        'interpretation_warning':'Five-day overlapping daily signals require HAC; five staggered phases share market outcomes. No inferred causal profitability.'})
    print('[6/6] Finished; all four models preserved, 2025–2026 locked.',flush=True)
    summary=quintiles[(quintiles.year=='2023-2024')&quintiles.measure.isin(['Q5','universe','top_minus_bottom','top_minus_universe'])]
    print(summary[['model','measure','mean','ci95_lo','ci95_hi']].to_string(index=False),flush=True)
    print('V10_ECONOMIC_DIAGNOSTICS_COMPLETE:',OUT,flush=True)
    print('Upload: v10_quintile_summary.csv, v10_phase_summary.csv, v10_execution_audit.json, v10_result_manifest.json, run_v10.log',flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',required=True,choices=['audit','evaluate'])
    args=p.parse_args()
    if args.stage=='audit':audit()
    else:evaluate()


if __name__=='__main__':main()
