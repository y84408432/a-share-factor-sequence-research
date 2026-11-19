#!/usr/bin/env python3
"""V5: user's LSTM window hypothesis, strictly TRAIN 2016-20 / VALID 2021-22.

4 sequence lengths (1,5,10,30) * 3 predeclared seeds, 16 frozen features,
SAME eligible30 train/valid stock-date anchors. Does not read development or
holdout labels/predictions. No ranking-driven tuning or architecture changes.
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

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from lstm_research.feature_spec import FEATURE_NAMES
from lstm_research.sequences import index_for
from lstm_research.pilot import (
    anchor_sha256, evaluate_validation, fit_train_label_scaler, pred_to_raw,
    save_json, target_for_train, target_for_valid,
)
from lstm_research.lstm_window_v5 import (
    WINDOWS, SEEDS, HIDDEN, EPOCHS, PATIENCE, BATCH_SIZE, LR, WEIGHT_DECAY,
    MIN_DELTA, GRAD_CLIP, materialize_cohort, train_one, check_paired_daily,
    across_seeds_daily, summarize_window, torch_device,
)

V2 = ROOT / 'output' / 'v2_sequences'
V3 = ROOT / 'output' / 'v3_pilot'
OUT = ROOT / 'output' / 'v5_lstm_windows'


def sha_json(data: dict) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False,
                       separators=(',', ':')).encode('utf-8')).hexdigest()


def verify_source(v2: dict, v3: dict, tr: pd.DataFrame, va: pd.DataFrame,
                  features: np.ndarray, labels: np.ndarray):
    if v2.get('features') != list(FEATURE_NAMES) or v2.get('n_features') != 16 or v2.get('default_window') != 30:
        raise ValueError('V2 frozen feature schema mismatch')
    if v2.get('n_panel_rows') != 861570:
        raise ValueError('V2 panel shape changed since V1/V2')
    if features.shape != (861570, 16) or len(labels) != len(features):
        raise ValueError('Stored V2 arrays have unexpected shapes')
    if len(tr) != 327555 or len(va) != 139220:
        raise ValueError('30d common cohort differs from V2 audit')
    if set(tr.split) != {'train'} or set(va.split) != {'valid'}:
        raise ValueError('Unexpected split in sequence study')
    if tr.datetime.min() < pd.Timestamp('2016-01-01') or tr.datetime.max() > pd.Timestamp('2020-12-31'):
        raise ValueError('Train dates outside preregistered period')
    if va.datetime.min() < pd.Timestamp('2021-01-01') or va.datetime.max() > pd.Timestamp('2022-12-31'):
        raise ValueError('Valid dates outside preregistered period')
    t_hash, v_hash = anchor_sha256(tr), anchor_sha256(va)
    if v3.get('train_anchor_sha256') != t_hash or v3.get('valid_anchor_sha256') != v_hash:
        raise ValueError('V3 baseline and V5 anchor cohorts differ')
    if v3.get('features') != list(FEATURE_NAMES):
        raise ValueError('V3 baseline and V5 factor names differ')
    return t_hash, v_hash


def write_protocol(data: dict):
    path = OUT / 'v5_protocol.json'
    if path.exists():
        former = json.loads(path.read_text(encoding='utf-8'))
        if former != data:
            raise ValueError('V5 protocol changed after experiment start. Do not silently replace; inspect prior results.')
    else:
        save_json(path, data)


def metric_record(window: int, seed: int, stats: dict, model_info: dict) -> dict:
    ri = stats['rankic_daily_equal_weight']
    return {'window': window, 'seed': seed, 'n_samples': stats['n_samples'],
            'n_days': ri['n_days'], 'rankic': ri['mean'], 'rankic_hac_lo': ri['ci95'][0],
            'rankic_hac_hi': ri['ci95'][1], 'ic': stats['ic_daily_equal_weight']['mean'],
            'mse_raw': stats['mse_raw'], 'best_epoch': model_info['best_epoch_by_valid_mse'],
            'param_count': model_info['parameter_count'], 'device': model_info['device']}


def main():
    parser = argparse.ArgumentParser(description='V5 LSTM 1/5/10/30d ×3 seeds, VALID only')
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps'], default='auto')
    parser.add_argument('--overwrite', action='store_true',
                        help='rerun ALL V5 fits under the exact same preregistered protocol')
    args = parser.parse_args()
    resolved_device = torch_device(args.device)

    print('[1/6] Loading frozen V2 arrays and V3 cohort fingerprint; only TRAIN/VALID target rows...', flush=True)
    v2 = json.loads((V2 / 'v2_sequence_audit.json').read_text(encoding='utf-8'))
    v3 = json.loads((V3 / 'pilot_protocol.json').read_text(encoding='utf-8'))
    features = np.load(V2 / 'features_scaled.npy', mmap_mode='r')
    targets = np.load(V2 / 'labels.npy', mmap_mode='r')
    indices = pd.read_pickle(V2 / 'sample_index.pkl.gz', compression='gzip')
    train = index_for(indices, 'train', 30)
    valid = index_for(indices, 'valid', 30)
    thash, vhash = verify_source(v2, v3, train, valid, features, targets)
    tr = train.panel_row.to_numpy(dtype=np.int64)
    va = valid.panel_row.to_numpy(dtype=np.int64)
    tr_raw = np.asarray(targets[tr], dtype=np.float64)
    va_raw = np.asarray(targets[va], dtype=np.float64)
    if not (np.isfinite(tr_raw).all() and np.isfinite(va_raw).all()):
        raise ValueError('Eligible anchor has a missing label')

    train_scaler = fit_train_label_scaler(tr_raw)
    saved_scaler = json.loads((V3 / 'label_scaler_train_only.json').read_text(encoding='utf-8'))
    for k in ('low', 'high', 'mean', 'std'):
        if not np.isclose(saved_scaler[k], train_scaler[k], rtol=1e-12, atol=1e-12):
            raise ValueError('V3 label scaler does not match train-only V5 scaler')
    ytr = target_for_train(tr_raw, train_scaler)
    yva = target_for_valid(va_raw, train_scaler)
    dates = valid.datetime.to_numpy()

    protocol = {
        'stage': 'V5 preregistered diagnostic: LSTM history-window and 3-seed stability, VALID only',
        'research_question': 'Do extra rolling-factor observations improve LSTM RankIC beyond latest-period rolling information?',
        'user_hypothesis': 'Long rolling factors already compress much history; a 30d LSTM window may be redundant.',
        'frozen_windows': list(WINDOWS), 'frozen_seeds': list(SEEDS),
        'frozen_features': list(FEATURE_NAMES),
        'shared_cohort': 'V2 eligible30 for ALL windows, including LSTM-1',
        'train_anchors_sha256': thash, 'valid_anchors_sha256': vhash,
        'train_n': len(tr), 'valid_n': len(va), 'expected_valid_days': 479,
        'training_years': '2016-2020', 'validation_years': '2021-2022',
        'development_2023_24': 'NOT ACCESSED', 'final_holdout_2025_26': 'NOT ACCESSED',
        'v2_preprocess_sha256': v2['scaler_sha256'],
        'v3_protocol_sha256': hashlib.sha256((V3/'pilot_protocol.json').read_bytes()).hexdigest(),
        'target_scaler': train_scaler,
        'architecture': {'type': 'LSTM', 'input_size': 16, 'hidden_size': HIDDEN,
                         'layers': 1, 'head': [32, 16, 1],
                         'parameter_count_invariant_over_windows': True},
        'optimizer': {'kind': 'AdamW', 'lr': LR, 'weight_decay': WEIGHT_DECAY,
                      'batch_size': BATCH_SIZE, 'max_epochs': EPOCHS,
                      'patience': PATIENCE, 'min_delta': MIN_DELTA,
                      'gradient_clip_norm': GRAD_CLIP},
        'checkpoint': 'minimum VALID scaled-target MSE; VALID RankIC never chooses epoch',
        'primary_metric': 'mean per-date cross-sectional RankIC',
        'secondary_metrics': ['Pearson IC', 'raw-return MSE', 'per-year RankIC', 'seed dispersion'],
        'paired_comparisons': ['5-1', '10-1', '30-1', '30-5', '30-10', '10-5'],
        'uncertainty': 'HAC Newey-West fixed lag 10 on date-paired 3-seed averaged daily RankIC differences; descriptive',
        'limitations': [
            'LSTM-1 has no multi-step recurrent temporal modeling; it is an architectural short-history control.',
            'The latest-day factors already include rolling price/volume history; LSTM-1 does NOT mean history-free information.',
            'The number of trainable parameters is constant over windows, but optimization and effective capacity can still differ.',
            'Validation MSE chooses checkpoints; comparisons of VALID RankIC are exploratory, not independent final-test performance.',
            'Three seeds are robustness checks, not three independent market samples; HAC uses calendar days, not 3*N pseudo-independent dates.',
            'Price adjustment/tradability and original CSI300 PIT provenance not yet fully audited.',
        ],
        'environment': {'python': sys.version.split()[0], 'arch': platform.machine()},
    }
    protocol_hash = sha_json(protocol)
    protocol['protocol_sha256'] = protocol_hash
    OUT.mkdir(parents=True, exist_ok=True)
    write_protocol(protocol)
    print(f'  shared samples: train={len(tr):,}, valid={len(va):,}; V3 anchors verified', flush=True)
    print(f'  selected device: {resolved_device}; results must use the same device', flush=True)
    print('  outcomes 2023-2026 not accessed; 4 windows × 3 seeds = 12 LSTM fits', flush=True)

    all_summary = []
    all_daily = []
    daily_seed_map = {}
    print('[2/6] Run window-by-window on the same anchored samples (input lookback only changes)...', flush=True)
    for w in WINDOWS:
        pending = []
        for seed in SEEDS:
            name = f'lstm_w{w:02d}_s{seed}'
            metric_file = OUT / f'{name}_metrics.json'
            day_file = OUT / f'{name}_daily_rankic.csv'
            weight_file = OUT / 'checkpoints' / f'{name}.pt'
            present = [p.is_file() for p in (metric_file, day_file, weight_file)]
            if all(present) and not args.overwrite:
                old = json.loads(metric_file.read_text(encoding='utf-8'))
                if (old.get('protocol_sha256') != protocol_hash or old.get('valid_anchor_sha256') != vhash or
                    old.get('train_anchor_sha256') != thash or old.get('seed') != seed or old.get('window') != w):
                    raise ValueError(f'Old output {name} conflicts with V5 protocol')
                if old.get('model_training', {}).get('device') != resolved_device:
                    raise ValueError(f'{name} trained on another device; do not mix devices. Use --overwrite to rerun ALL V5 fits.')
                print(f'  {name}: prior successful fit found, reuse', flush=True)
            else:
                pending.append(seed)
        xtr, xva = None, None
        if pending:
            print(f'  Building cached 16-factor sequences for window={w} (one window at a time)...', flush=True)
            xtr = materialize_cohort(features, tr, w)
            xva = materialize_cohort(features, va, w)
        for seed in pending:
            name = f'lstm_w{w:02d}_s{seed}'
            metric_file = OUT / f'{name}_metrics.json'
            day_file = OUT / f'{name}_daily_rankic.csv'
            weight_file = OUT / 'checkpoints' / f'{name}.pt'
            print(f'   Fitting {name}...', flush=True)
            pred_scaled, detail = train_one(xtr, ytr, xva, yva, window=w, seed=seed,
                                            device_request=args.device, checkpoint=weight_file)
            prediction = pred_to_raw(pred_scaled, train_scaler)
            stats, daily = evaluate_validation(dates, va_raw, prediction)
            if stats['rankic_daily_equal_weight']['n_days'] != 479:
                raise ValueError(f'Unexpected valid day count for {name}')
            stats.update({'model': name, 'window': w, 'seed': seed,
                          'valid_anchor_sha256': vhash, 'train_anchor_sha256': thash,
                          'protocol_sha256': protocol_hash, 'model_training': detail})
            # Completion marker is metrics JSON, written LAST.
            daily.to_csv(day_file, index=False, encoding='utf-8-sig')
            save_json(metric_file, stats)
            print(f'   {name}: RankIC={stats["rankic_daily_equal_weight"]["mean"]:+.6f}, '
                  f'IC={stats["ic_daily_equal_weight"]["mean"]:+.6f}, '
                  f'best_epoch={detail["best_epoch_by_valid_mse"]}', flush=True)
        del xtr, xva

    print('[3/6] Checking all runs, exact seed and date alignment...', flush=True)
    for w in WINDOWS:
        for seed in SEEDS:
            name = f'lstm_w{w:02d}_s{seed}'
            m = json.loads((OUT / f'{name}_metrics.json').read_text(encoding='utf-8'))
            if m['protocol_sha256'] != protocol_hash or m['valid_anchor_sha256'] != vhash:
                raise ValueError('Output protocol mismatch')
            d = pd.read_csv(OUT / f'{name}_daily_rankic.csv', parse_dates=['date'])
            if len(d) != 479 or d.date.duplicated().any():
                raise ValueError('Unexpected or duplicate validation days')
            all_summary.append(metric_record(w, seed, m, m['model_training']))
            all_daily.append(d.assign(window=w, seed=seed))
            daily_seed_map[w, seed] = d
    # Paired checks across EVERY window and seed: exactly same days and stock counts.
    exemplar = daily_seed_map[WINDOWS[0], SEEDS[0]]
    for d in daily_seed_map.values():
        check_paired_daily(d, exemplar)

    print('[4/6] Compute across-seed summaries and per-day HAC paired differences...', flush=True)
    window_meta, daily_mean = {}, {}
    for w in WINDOWS:
        ms = [json.loads((OUT / f'lstm_w{w:02d}_s{s}_metrics.json').read_text(encoding='utf-8')) for s in SEEDS]
        ds = [daily_seed_map[w, s] for s in SEEDS]
        window_meta[str(w)] = summarize_window(ms, ds)
        daily_mean[w] = across_seeds_daily(ds)
    comparisons = {}
    for a, b in ((5, 1), (10, 1), (30, 1), (30, 5), (30, 10), (10, 5)):
        pooled = check_paired_daily(daily_mean[a], daily_mean[b])
        per_seed = {str(s): check_paired_daily(daily_seed_map[a, s], daily_seed_map[b, s]) for s in SEEDS}
        comparisons[f'lstm_{a}_minus_lstm_{b}'] = {
            'daily_mean_over_three_seeds_hac': pooled,
            'paired_same_seed_results': per_seed,
            'interpretation': 'Difference in mean daily RankIC for same date/stock anchors, not an independent holdout test',
        }

    print('[5/6] Write summary files, model traces and diagnostic daily results...', flush=True)
    pd.DataFrame(all_summary).sort_values(['window', 'seed']).to_csv(
        OUT / 'v5_seed_model_summary.csv', index=False, encoding='utf-8-sig')
    table = []
    for w in WINDOWS:
        s = window_meta[str(w)]
        row = {'window': w, 'seed_count': s['n_seeds'], 'valid_days': s['n_valid_days'],
               'rankic_mean_3seed': s['rankic_seed_mean'], 'rankic_sd_between_seeds': s['rankic_seed_std'],
               'rankic_min_seed': s['rankic_seed_min'], 'rankic_max_seed': s['rankic_seed_max'],
               'rankic_mean_daily_hac_low': s['rankic_daily_seed_mean_hac']['ci95'][0],
               'rankic_mean_daily_hac_high': s['rankic_daily_seed_mean_hac']['ci95'][1],
               'ic_mean_3seed': s['ic_seed_mean'], 'mse_raw_mean_3seed': s['raw_mse_seed_mean']}
        row.update({f'rankic_{year}': val for year, val in s['yearly_seed_mean_rankic'].items()})
        table.append(row)
    pd.DataFrame(table).to_csv(OUT/'v5_window_summary.csv', index=False, encoding='utf-8-sig')
    save_json(OUT/'v5_paired_rankic.json', {
        'protocol_sha256': protocol_hash,
        'daily_rankic_summary': window_meta,
        'pairwise': comparisons,
        'warning': 'Validation MSE checkpoint-selected; comparisons are development diagnostics, not blind-test inference',
    })
    pd.concat(all_daily, ignore_index=True).to_csv(OUT/'v5_daily_rankic_all_seeds.csv', index=False, encoding='utf-8-sig')
    trace_rows = []
    for w in WINDOWS:
        for s in SEEDS:
            stats = json.loads((OUT/f'lstm_w{w:02d}_s{s}_metrics.json').read_text(encoding='utf-8'))
            for row in stats['model_training']['trace']:
                trace_rows.append({'window': w, 'seed': s, **row})
    pd.DataFrame(trace_rows).to_csv(OUT/'v5_training_curves.csv', index=False, encoding='utf-8-sig')
    print(pd.read_csv(OUT/'v5_window_summary.csv').to_string(index=False), flush=True)
    print('  daily paired contrasts (3-seed average):', flush=True)
    for name, cmp in comparisons.items():
        c = cmp['daily_mean_over_three_seeds_hac']
        print(f'  {name}: {c["mean"]:+.6f}; HAC95={c["ci95"]}', flush=True)
    print('[6/6] Validation-only finished; NO later-period outcomes accessed.', flush=True)
    print('V5_LSTM_WINDOW_EXPERIMENT_COMPLETE:', OUT, flush=True)
    print('Share v5_window_summary.csv, v5_seed_model_summary.csv, v5_paired_rankic.json, run_v5.log', flush=True)


if __name__ == '__main__':
    main()
