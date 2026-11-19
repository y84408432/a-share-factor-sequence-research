#!/usr/bin/env python3
"""V11: one-pass 2023–2024 research-validity audit of frozen V1–V10.

No training, no new predictions, no re-ranked outcome performance, no 2025–26.
The script reports *mechanical* data consistency and limits of inference; it
cannot independently certify as-of-date CSI300 history or executed trades.
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
from lstm_research.v9_development import file_hash,model_registry
from lstm_research.v10_economics import MODEL_IDS,validate_predictions
from lstm_research.v11_validity import (
    read_membership_intervals,membership_info,daily_pool_range,
    eligibility_profile,align_trade_fields,risk_flags,risk_counts,q5_risk_by_model,
)
from lstm_research.pilot import save_json

SRC=Path.home()/'.qlib/qlib_data/cn_data'
V1=ROOT/'output/v1_data'
V2=ROOT/'output/v2_sequences'
V9=ROOT/'output/v9_development'
V10=ROOT/'output/v10_economics'
OUT=ROOT/'output/v11_validity'
PRED=V9/'v9_development_predictions.csv.gz'
QUOTE=V10/'quotes_2023_2024.pkl.gz'
CALENDAR=V10/'calendar_2023_2024.npy'
V11_OUTPUTS=(OUT/'v11_membership_audit.json',OUT/'v11_eligibility_audit.json',
             OUT/'v11_execution_risk_audit.json',OUT/'v11_verdict.json',
             OUT/'v11_source_manifest.json')


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_frozen():
    needed=[SRC/'instruments/csi300.txt', SRC/'calendars/day.txt',
            V1/'frozen_factor_panel.pkl.gz',V2/'sample_index.pkl.gz',
            V9/'v9_preflight.json',V9/'v9_frozen_protocol.json',PRED,
            V10/'v10_preflight.json',V10/'v10_protocol.json',
            V10/'v10_result_manifest.json',QUOTE,CALENDAR]
    for f in needed:
        if not f.is_file():raise FileNotFoundError(f'Missing frozen source: {f}')
    v9pre=read_json(V9/'v9_preflight.json')
    v9prot=read_json(V9/'v9_frozen_protocol.json')
    v10pre=read_json(V10/'v10_preflight.json')
    v10prot=read_json(V10/'v10_protocol.json')
    v10res=read_json(V10/'v10_result_manifest.json')
    if (v9pre['model_registry']!=model_registry(ROOT) or
            v10prot['source_v9_model_registry']!=v9pre['model_registry']):
        raise ValueError('Frozen V9 model checkpoint hashes changed')
    if (v10pre['protocol_sha256']!=v10prot['protocol_sha256'] or
            v10res['protocol_sha256']!=v10prot['protocol_sha256'] or
            v10prot['source_v9_protocol_sha256']!=v9prot['protocol_sha256'] or
            v10prot['source_v9_cohort_sha256']!=v9pre['2023_2024_anchor_sha256'] or
            v10prot['prediction_file_hash']!=file_hash(PRED) or
            v10prot['price_file_hash']!=file_hash(QUOTE) or
            v10prot['calendar_file_hash']!=file_hash(CALENDAR)):
        raise ValueError('Frozen V9/V10 protocol or source data changed')
    files=v10res.get('files_sha256',{})
    for fn,sha in files.items():
        p=V10/fn
        if not p.is_file() or file_hash(p)!=sha:
            raise ValueError(f'Frozen V10 result source changed: {p}')
    if v10pre['quote_snapshot_hash']!=file_hash(QUOTE):
        raise ValueError('Frozen quote audit mismatch')
    pred=validate_predictions(pd.read_csv(PRED),expected_rows=141415,expected_days=478)
    if set(pred.columns).isdisjoint({m+'_mean_raw_pred' for m in MODEL_IDS}):
        raise ValueError('No frozen model prediction columns')
    q=pd.read_pickle(QUOTE,compression='gzip')
    cal=pd.DatetimeIndex(np.load(CALENDAR))
    if q.empty or len(cal)<480:raise ValueError('V10 quote/calendar unexpected')
    return pred,q,cal,v9pre,v10prot


def provider_snapshots(intervals,days):
    """Optional check Qlib API vs local file on fixed 2023/2024 signal dates."""
    selected=[days[0],days[len(days)//4],days[len(days)//2],
              days[(3*len(days))//4],days[-1]]
    reports=[]
    import qlib
    from qlib.constant import REG_CN
    from qlib.data import D
    qlib.init(provider_uri=str(SRC),region=REG_CN)
    inst=D.instruments(market='csi300')
    for date in selected:
        d=pd.Timestamp(date)
        expected=set(intervals.loc[(intervals.start<=d)&(intervals.end>=d),'instrument'])
        fetched=set(map(str.upper,D.list_instruments(
            instruments=inst,start_time=str(d.date()),end_time=str(d.date()),as_list=True)))
        reports.append({'date':str(d.date()),'in_local_intervals':len(expected),
                        'via_qlib_provider':len(fetched),
                        'symmetric_difference':len(expected^fetched)})
    return reports


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=['audit'],default='audit')
    args=p.parse_args()
    if all(fp.exists() for fp in V11_OUTPUTS):
        print('V11_ALREADY_AUDITED: existing outputs preserved; no overwrite.',flush=True)
        return
    if any(fp.exists() for fp in V11_OUTPUTS):
        raise RuntimeError('Some V11 files already exist. Review before rerun; will not overwrite.')
    print('[1/6] Confirm V9/V10 frozen artifacts and no 2025-2026 evaluation...',flush=True)
    pred,q,cal,v9pre,v10prot=verify_frozen()
    print(f'  {len(pred):,} frozen stock-date signals, {pred.datetime.nunique()} trading days',flush=True)

    print('[2/6] Parse local CSI300 historical stock membership intervals...',flush=True)
    intervals,interval_meta=read_membership_intervals(SRC/'instruments/csi300.txt')
    source_anomalies = interval_meta['nonstandard_source_identifiers']
    for anomaly in source_anomalies:
        print('  SOURCE ANOMALY PRESERVED:', anomaly['source_identifier'],
              '| source rows:', anomaly['raw_interval_rows_preserved'],
              '| earliest:', anomaly['earliest_start'],
              '| latest:', anomaly['latest_end'],
              '| active in 2023-2024:', anomaly['active_in_2023_2024'],
              flush=True)
    allcal=pd.DatetimeIndex(pd.to_datetime(
        pd.read_csv(SRC/'calendars/day.txt',header=None).iloc[:,0],errors='raise'))
    if len(allcal)<2000 or not np.all(allcal.to_numpy()[1:]>allcal.to_numpy()[:-1]):
        raise ValueError('Invalid local Qlib trading calendar')
    included,age,starts=membership_info(pred[['instrument','datetime']],intervals,allcal)
    pools=daily_pool_range(intervals,pd.DatetimeIndex(pred.datetime.unique()).sort_values())
    snap=provider_snapshots(intervals,pools.datetime.tolist())
    mismatches=sum(r['symmetric_difference'] for r in snap)
    member_report={
        'analysis_period':'2023-2024 only',
        'source_file':'~/.qlib/qlib_data/cn_data/instruments/csi300.txt',
        'source_sha256':file_hash(SRC/'instruments/csi300.txt'),
        'source_intervals':interval_meta,
        'unresolved_nonstandard_codes_active_in_evaluation': [
            a['source_identifier'] for a in source_anomalies if a['active_in_2023_2024']],
        'frozen_samples':len(pred),
        'selected_signal_rows_outside_file_membership':int((~included).sum()),
        'selected_signal_rows_in_membership':int(included.sum()),
        'selected_member_interval_age_lt_30_trading_sessions':int(((age>=0)&(age<29)).sum()),
        'selected_member_interval_age_lt_60_trading_sessions':int(((age>=0)&(age<59)).sum()),
        'selected_member_interval_age_sessions_min':int(age[included].min()) if included.any() else None,
        'frozen_signal_dates':len(pools),
        'active_constituent_count_min':int(pools.n_active.min()),
        'active_constituent_count_median':float(pools.n_active.median()),
        'active_constituent_count_max':int(pools.n_active.max()),
        'active_constituent_days_outside_250_to_350':int((~pools.n_active.between(250,350)).sum()),
        'qlib_provider_file_snapshots':snap,
        'provider_file_symmetric_difference_total':int(mismatches),
        'provider_asof_provenance':'UNVERIFIED: local effective intervals and Qlib API share a dataset. Source revision history, announcement timing and historical as-of publication must be checked independently.',
        'conclusion_type':'mechanical consistency only; NOT an independent PIT certification',
    }
    print(f"  missing-membership={member_report['selected_signal_rows_outside_file_membership']} ; "
          f'provider snapshot differences={mismatches}',flush=True)

    print('[3/6] Build ex-ante B30 candidate cohort BEFORE checking future label...',flush=True)
    panel=pd.read_pickle(V1/'frozen_factor_panel.pkl.gz',compression='gzip')
    panel=panel.loc[(panel.split=='development')&
                    pd.to_datetime(panel.datetime).between('2023-01-01','2024-12-31')].copy()
    eligible, candidates, label_lost=eligibility_profile(panel,pred)
    cand_mem,_,_=membership_info(candidates[['instrument','datetime']],intervals)
    eligible['exante_candidates_outside_constituent_file']=int((~cand_mem).sum())
    eligible['frozen_cohort_is_future_label_conditioned']=True
    eligible['eligibility_status']=(
        'Future-label eligibility explicitly audited, but completely excluded stocks/bars '
        'cannot be detected from the membership-filtered V1 panel alone.')
    print(f"  ex-ante B30 candidates={eligible['exante_b30_candidates']:,}, "
          f"future-label exclusions={eligible['excluded_due_to_future_label_nonfinite']:,}",flush=True)

    print('[4/6] Audit existing V10 opening-bar executability proxies, without refiltering...',flush=True)
    detail=align_trade_fields(pred,q,cal)
    flags=risk_flags(detail)
    all_count=risk_counts(flags,np.ones(len(pred),dtype=bool))
    by_model=q5_risk_by_model(pred,flags)
    fac0=detail.entry_factor.to_numpy(dtype=float)
    fac1=detail.exit_factor.to_numpy(dtype=float)
    good=np.isfinite(fac0)&np.isfinite(fac1)&(fac0>0)&(fac1>0)
    factor_changes=np.zeros(len(pred),dtype=bool)
    factor_changes[good]=np.abs(fac1[good]/fac0[good]-1)>=0.005
    symbols=sorted(pred.instrument.astype(str).str.lower().unique())
    features=SRC/'features'
    up=sum((features/s/'up_limit.day.bin').is_file() for s in symbols)
    down=sum((features/s/'down_limit.day.bin').is_file() for s in symbols)
    risk_report={
        'period':'2023-2024','signals':len(pred),
        'frozen_sample_all':all_count,
        'frozen_Q5_by_model':by_model,
        'factor_entry_exit_nonpositive_or_nan':int((~good).sum()),
        'factor_change_ge_0_5pct_between_entry_exit_count':int(factor_changes.sum()),
        'instruments_with_up_limit_bin_present':up,
        'instruments_with_down_limit_bin_present':down,
        'instruments_checked_for_limit_field':len(symbols),
        'large_adjusted_open_gaps_are_only_proxies':True,
        'no_actual_limit_up_down_flags_verified':True,
        'volume_positive_does_not_prove_auction_execution':True,
        'unadjusted_price_reconstruction_not_certified':True,
        'rejected_trades_or_forced_holding_costs_not_simulated':True,
        'no_hindsight_filter_or_recomputed_return':True,
        'interpretation':'Q5 signals are counted, not excluded. Future t+6 volume and exit quote are audit-only; no ex-ante tradability rule may depend on them. Auction fill, limit-lock, ST status, minimum lot, market impact and fee schedule require independent data.'
    }
    print('  entry gap >=9.8%:',all_count['entry_gap_up_9_8pct']['count'],
          '; entry gap <=-9.8%:',all_count['entry_gap_down_9_8pct']['count'],flush=True)

    print('[5/6] Explicit evidence boundaries and independent-validity verdict...',flush=True)
    severe=[]
    if member_report['unresolved_nonstandard_codes_active_in_evaluation']:
        severe.append('Unresolved source instrument identifier overlaps 2023-2024 evaluation period')
    if member_report['selected_signal_rows_outside_file_membership']:
        severe.append('Frozen signal dates outside effective CSI300 membership ranges')
    if mismatches: severe.append('Qlib stock-pool API differs from local membership intervals')
    if eligible['exante_candidates_outside_constituent_file']:
        severe.append('Ex-ante candidate stocks outside effective CSI300 membership ranges')
    if all_count['entry_quote_invalid']['count'] or all_count['exit_quote_invalid']['count']:
        severe.append('Missing/nonpositive execution quote or volume among selected frozen cohort')
    verdict={
        'status':'REVIEW_REQUIRED' if severe else 'MECHANICAL_CHECKS_PASSED_WITH_UNVERIFIED_ASSUMPTIONS',
        'hard_inconsistencies':severe,
        'source_nonstandard_identifiers_preserved_unmapped': source_anomalies,
        'historical_source_identifier_provenance_certified': False,
        'independent_historical_CSI300_PIT_certified':False,
        'no_future_eligibility_selection_certified':False,
        'real_opening_auction_fills_certified':False,
        'historical_China_A_share_trading_costs_certified':False,
        'suitable_to_call_real_executable_backtest':False,
        'suitable_to_call_hypothetical_signal_economics':len(severe)==0,
        'strict_policy':'Do not select/refit models or revise V9/V10 outcomes using these audit findings. Report limitations even if mechanical checks pass.',
        'future_work_if_genuine_execution_claim_required':[
          'Independently source date-effective CSI300 constituent history with publication dates',
          'Rebuild ex-ante active universe and separately quantify t+6 label/censoring loss',
          'Acquire raw unadjusted OHLC, exact daily price-limit and ST/halt/auction data',
          'Introduce reject/delay/forced exit simulation without pre-filtering on t+6 status',
          'Model actual broker fees, stamps, lots, capacity and slippage with dated source',
        ],
        'final_period_2025_2026_performance':'NOT evaluated',
    }
    print('  verdict:',verdict['status'],flush=True)

    print('[6/6] Write immutable short audit files and source hashes...',flush=True)
    OUT.mkdir(parents=True,exist_ok=True)
    for path,body in zip(V11_OUTPUTS[:4],(member_report,eligible,risk_report,verdict)):
        save_json(path,body)
    src={
      'stage':'V11 locked 2023-2024 data-validity audit ONLY',
      'source_files_sha256':{
        'v1_frozen_factor_panel':file_hash(V1/'frozen_factor_panel.pkl.gz'),
        'v9_predictions':file_hash(PRED),
        'v10_quote_snapshot':file_hash(QUOTE),
        'v10_calendar':file_hash(CALENDAR),
        'csi300_intervals':file_hash(SRC/'instruments/csi300.txt'),
        'qlib_trading_calendar':file_hash(SRC/'calendars/day.txt'),
      },
      'source_protocols':{'v9_preflight_cohort_sha256':v9pre['2023_2024_anchor_sha256'],
                          'v10_protocol_sha256':v10prot['protocol_sha256']},
      'outputs_sha256':{f.name:file_hash(f) for f in V11_OUTPUTS[:4]},
      'models_preserved':list(MODEL_IDS),
      'market_period':'2023-2024',
      'final_2025_2026_outcomes':'NOT evaluated',
      'note':'No model fitting, inference, re-ranking, return computation, or changed V9/V10 results.',
    }
    save_json(V11_OUTPUTS[4],src)
    print('V11_VALIDITY_AUDIT_COMPLETE:',OUT,flush=True)
    print('Share: v11_membership_audit.json, v11_eligibility_audit.json, '
          'v11_execution_risk_audit.json, v11_verdict.json, v11_source_manifest.json, run_v11.log',flush=True)


if __name__=='__main__':main()
