#!/usr/bin/env python3
"""V6 user-led hypothesis: do historical positions/order matter to V5 LSTMs?

C: trained V5 checkpoint, alter VALID past order only: sensitivity test.
D: train NEW LSTM using independently shuffled past per sample, shuffle VALID
   with the same randomization *distribution*, compare to V5 chronological
   training. D isolates the role of ordered historical positions imperfectly:
   other network inductive biases and optimization still change.

Frozen windows: 10 and 30; seeds: 3; 3 C permutation repeats per seed.
Same V2 30-day eligible train/valid cohort, 16 factors, same V5 architecture,
loss, optimizer, target scaler, stopping and evaluation; future years unread.
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
from lstm_research.pilot import (anchor_sha256, fit_train_label_scaler,
        target_for_train, target_for_valid, nw_mean_interval, save_json)
from lstm_research.lstm_window_v5 import (materialize_cohort, train_one, torch_device)
from lstm_research.order_v6 import (WINDOWS, SEEDS, C_REPEATS, shuffle_past,
        shuffle_seed, predict_checkpoint, average_daily, compare_daily, cv_eval)

V2 = ROOT / 'output' / 'v2_sequences'
V3 = ROOT / 'output' / 'v3_pilot'
V5 = ROOT / 'output' / 'v5_lstm_windows'
OUT = ROOT / 'output' / 'v6_order_study'


def load_and_verify():
    v2 = json.loads((V2 / 'v2_sequence_audit.json').read_text(encoding='utf-8'))
    v3 = json.loads((V3 / 'pilot_protocol.json').read_text(encoding='utf-8'))
    v5 = json.loads((V5 / 'v5_protocol.json').read_text(encoding='utf-8'))
    if (v2['features'] != list(FEATURE_NAMES) or v2['n_features'] != 16 or
            v2['default_window'] != 30 or v2['n_panel_rows'] != 861570):
        raise ValueError('V2 frozen schema changed')
    if list(v5.get('frozen_windows', [])) != [1, 5, 10, 30] or list(v5.get('frozen_seeds', [])) != list(SEEDS):
        raise ValueError('V5 experiment identity changed')
    if v5.get('frozen_features') != list(FEATURE_NAMES) or v5.get('v2_preprocess_sha256') != v2['scaler_sha256']:
        raise ValueError('V5 dataset fingerprint mismatch')
    feat = np.load(V2 / 'features_scaled.npy', mmap_mode='r')
    yy = np.load(V2 / 'labels.npy', mmap_mode='r')
    if feat.shape != (861570, 16) or len(yy) != len(feat):
        raise ValueError('V2 stored data shape changed')
    indices = pd.read_pickle(V2 / 'sample_index.pkl.gz', compression='gzip')
    tr = index_for(indices, 'train', 30)
    va = index_for(indices, 'valid', 30)
    if len(tr) != 327555 or len(va) != 139220:
        raise ValueError('V5 common cohort size mismatch')
    if (tr.datetime.min() < pd.Timestamp('2016-01-01') or
            tr.datetime.max() > pd.Timestamp('2020-12-31') or
            va.datetime.min() < pd.Timestamp('2021-01-01') or
            va.datetime.max() > pd.Timestamp('2022-12-31')):
        raise ValueError('Train/valid period mismatch')
    th, vh = anchor_sha256(tr), anchor_sha256(va)
    if v3['train_anchor_sha256'] != th or v3['valid_anchor_sha256'] != vh:
        raise ValueError('V3 anchor fingerprints differ')
    if v5['train_anchors_sha256'] != th or v5['valid_anchors_sha256'] != vh:
        raise ValueError('V5 anchor fingerprints differ')
    train_rows = tr.panel_row.to_numpy(dtype=np.int64)
    valid_rows = va.panel_row.to_numpy(dtype=np.int64)
    ytr_raw = np.asarray(yy[train_rows], dtype=np.float64)
    yva_raw = np.asarray(yy[valid_rows], dtype=np.float64)
    label_scaler = fit_train_label_scaler(ytr_raw)
    old_label_scaler = json.loads((V3 / 'label_scaler_train_only.json').read_text(encoding='utf-8'))
    for k in ('low', 'high', 'mean', 'std'):
        if not np.isclose(label_scaler[k], old_label_scaler[k], rtol=1e-12, atol=1e-12):
            raise ValueError('V3/V6 training label scaler differs')
    return dict(v2=v2, v5=v5, feat=feat, train=tr, valid=va,
                tr_rows=train_rows, va_rows=valid_rows, tr_raw=ytr_raw,
                va_raw=yva_raw, ytr=target_for_train(ytr_raw, label_scaler),
                yva=target_for_valid(yva_raw, label_scaler),
                scaler=label_scaler, dates=va.datetime.to_numpy(),
                train_hash=th, valid_hash=vh)


def source_daily(w: int, seed: int, source: dict) -> pd.DataFrame:
    name = f'lstm_w{w:02d}_s{seed}'
    metap = V5 / f'{name}_metrics.json'
    path = V5 / f'{name}_daily_rankic.csv'
    checkpoint = V5 / 'checkpoints' / f'{name}.pt'
    if not all(p.is_file() for p in (metap, path, checkpoint)):
        raise FileNotFoundError(f'V5 unfinished model {name}; run V5 first')
    m = json.loads(metap.read_text(encoding='utf-8'))
    if (m['seed'] != seed or m['window'] != w or m['valid_anchor_sha256'] != source['valid_hash'] or
            m['train_anchor_sha256'] != source['train_hash'] or
            m['protocol_sha256'] != source['v5']['protocol_sha256']):
        raise ValueError(f'V5 metrics fingerprint mismatch: {name}')
    frame = pd.read_csv(path, parse_dates=['date'])
    if len(frame) != 479 or frame.date.duplicated().any():
        raise ValueError(f'V5 daily file invalid: {name}')
    return frame


def make_protocol(source: dict, device: str):
    p = {'stage': 'V6 preregistered historical-order perturbation C/D, validation only',
         'user_hypothesis': 'Rolling factors summarize prior prices; repeated historical snapshots may be redundant',
         'windows': list(WINDOWS), 'seeds': list(SEEDS), 'c_permutation_repeats': list(C_REPEATS),
         'train_n': 327555, 'valid_n': 139220, 'valid_days': 479,
         'train_hash': source['train_hash'], 'valid_hash': source['valid_hash'],
         'v5_protocol_sha256': source['v5']['protocol_sha256'],
         'v2_scaler_sha256': source['v2']['scaler_sha256'],
         'features': list(FEATURE_NAMES), 'device': device,
         'C': 'V5 fitted checkpoint; shuffle past positions per VALID sample; latest day unchanged; no retraining; 3 repeats',
         'D': 'New LSTM from scratch; each TRAIN and VALID sample gets independent random past permutation, fixed for each sample; same distribution both stages, same hyperparameters as V5',
         'randomization': 'whole 16-feature day-vector moved together; no cross-stock mixing; no new or future observations',
         'target_scaler': source['scaler'],
         'training_parameters_from': 'lstm_research.lstm_window_v5.train_one',
         'metrics': 'daily cross-sectional RankIC, IC, MSE; daily paired HAC fixed 10 lags; per-seed and 3-seed means',
         'restrictions': ['C detects model sensitivity and distribution shift; not causal temporal predictive value.',
                          'D randomizes older order during both fit and validation; a gap is not uniquely attributable to temporal alpha.',
                          'D has additional sample-specific input randomness, potentially affecting optimization.',
                          'A non-significant gap cannot establish no temporal learning.',
                          'V5 model checkpoint selected using validation MSE, so this is NOT a blind evaluation.',
                          '2023-2024 and 2025-2026 outcomes never read.',
                          'Only completed 3-seed means are reported; seeds are not independent market samples.',
                          'PIT, adjust-factor and tradability concerns remain for economic backtests.'],
         'python': sys.version.split()[0], 'architecture': platform.machine()}
    raw = json.dumps(p, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    p['protocol_sha256'] = hashlib.sha256(raw.encode('utf-8')).hexdigest()
    return p


def write_protocol_once(protocol: dict):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'v6_protocol.json'
    if path.is_file():
        old = json.loads(path.read_text(encoding='utf-8'))
        if old != protocol:
            raise ValueError('Existing V6 protocol differs. Do NOT silently mix device or experimental definitions.')
    else:
        save_json(path, protocol)


def c_run(window: int, xva: np.ndarray, src: dict, protocol: dict, requested_device: str):
    print(f'[C] window={window}: V5 checkpoint sensitivity, 3 seeds × 3 permutations', flush=True)
    rows = []
    for seed in SEEDS:
        original = source_daily(window, seed, src)
        checkpoint = V5 / 'checkpoints' / f'lstm_w{window:02d}_s{seed}.pt'
        base_scaled = predict_checkpoint(xva, checkpoint, window=window,
                                         seed=seed, device_request=requested_device)
        _, baseline_recomputed = cv_eval(src['dates'], src['va_raw'], base_scaled, src['scaler'])
        compare_daily(baseline_recomputed, original)
        deviation = float(np.abs(baseline_recomputed.rankic - original.rankic).mean())
        if deviation > 0.002:
            raise ValueError(f'V5 checkpoint predictions do not reproduce logged RankIC, mean diff={deviation:.6f}')
        for repeat in C_REPEATS:
            name = f'C_w{window:02d}_s{seed}_r{repeat}'
            fp = OUT / (name + '_metrics.json')
            dp = OUT / (name + '_daily.csv')
            if fp.is_file() and dp.is_file():
                record = json.loads(fp.read_text(encoding='utf-8'))
                if record.get('protocol_sha256') != protocol['protocol_sha256']:
                    raise ValueError(f'Conflicting cached C result: {name}')
                print(f'   {name}: reused', flush=True)
            else:
                xsh = shuffle_past(xva, shuffle_seed(seed, window=window, experiment='C', repeat=repeat))
                # Proof latest factor vector unchanged and time-step multiset intact is in tests.
                pred = predict_checkpoint(xsh, checkpoint, window=window,
                                          seed=seed, device_request=requested_device)
                stats, daily = cv_eval(src['dates'], src['va_raw'], pred, src['scaler'])
                delta = compare_daily(daily, original)
                stats.update({'model': name, 'type': 'C', 'window':window, 'seed':seed,
                              'repeat':repeat, 'protocol_sha256':protocol['protocol_sha256'],
                              'mean_absolute_scaled_prediction_change':float(np.mean(np.abs(pred-base_scaled))),
                              'prediction_correlation': float(np.corrcoef(pred, base_scaled)[0, 1]),
                              'paired_rankic_minus_original':delta})
                daily.to_csv(dp, index=False, encoding='utf-8-sig')
                save_json(fp, stats)
                print(f'   {name}: RankIC={stats["rankic_daily_equal_weight"]["mean"]:+.6f}; '
                      f'delta={delta["mean"]:+.6f}, avg abs pred delta={stats["mean_absolute_scaled_prediction_change"]:.5f}', flush=True)
            record = json.loads(fp.read_text(encoding='utf-8'))
            rows.append({'experiment':'C', 'window':window, 'seed':seed, 'repeat':repeat,
                         'rankic':record['rankic_daily_equal_weight']['mean'],
                         'ic':record['ic_daily_equal_weight']['mean'],
                         'mse_raw':record['mse_raw'],
                         'mean_absolute_scaled_prediction_change':record['mean_absolute_scaled_prediction_change'],
                         'prediction_correlation':record['prediction_correlation']})
    return rows


def d_run(window: int, xtr: np.ndarray, xva: np.ndarray, src: dict,
          protocol: dict, requested_device: str):
    print(f'[D] window={window}: TRAIN + VALID both samplewise shuffled; 3 seeds', flush=True)
    rows = []
    for seed in SEEDS:
        name = f'D_w{window:02d}_s{seed}'
        fp = OUT / (name+'_metrics.json')
        dp = OUT / (name+'_daily.csv')
        checkpoint = OUT / 'checkpoints' / (name+'.pt')
        if all(p.is_file() for p in (fp, dp, checkpoint)):
            m = json.loads(fp.read_text(encoding='utf-8'))
            if m.get('protocol_sha256') != protocol['protocol_sha256']:
                raise ValueError(f'Conflicting cached D result: {name}')
            print(f'   {name}: reused', flush=True)
        else:
            xtr_sh = shuffle_past(xtr, shuffle_seed(seed, window=window, experiment='D', split='train'))
            xva_sh = shuffle_past(xva, shuffle_seed(seed, window=window, experiment='D', split='valid'))
            pred, training = train_one(xtr_sh, src['ytr'], xva_sh, src['yva'],
                                       window=window, seed=seed, device_request=requested_device,
                                       checkpoint=checkpoint)
            stats, daily = cv_eval(src['dates'], src['va_raw'], pred, src['scaler'])
            orig = source_daily(window, seed, src)
            delta = compare_daily(daily, orig)
            stats.update({'model':name, 'type':'D', 'window':window, 'seed':seed,
                          'protocol_sha256':protocol['protocol_sha256'],
                          'model_training':training, 'paired_rankic_minus_original':delta})
            daily.to_csv(dp, index=False, encoding='utf-8-sig')
            save_json(fp, stats)
            print(f'   {name}: RankIC={stats["rankic_daily_equal_weight"]["mean"]:+.6f}; '
                  f'delta={delta["mean"]:+.6f}, best_epoch={training["best_epoch_by_valid_mse"]}', flush=True)
            del xtr_sh, xva_sh
        m = json.loads(fp.read_text(encoding='utf-8'))
        rows.append({'experiment':'D', 'window':window, 'seed':seed, 'repeat':0,
                     'rankic':m['rankic_daily_equal_weight']['mean'],
                     'ic':m['ic_daily_equal_weight']['mean'], 'mse_raw':m['mse_raw'],
                     'best_epoch':m['model_training']['best_epoch_by_valid_mse']})
    return rows


def write_report(rows: list[dict], source: dict, protocol: dict):
    if not rows:
        return
    # The official summary includes only completed comparisons; no silent missing replicates.
    record = pd.DataFrame(rows)
    all_entries = []
    paired = {}
    for w in WINDOWS:
        for exp in ('C', 'D'):
            expected = len(SEEDS) * (len(C_REPEATS) if exp=='C' else 1)
            sub = record.loc[(record.window==w) & (record.experiment==exp)]
            if len(sub) != expected:
                continue
            a = average_daily([pd.read_csv(V5/f'lstm_w{w:02d}_s{s}_daily_rankic.csv', parse_dates=['date']) for s in SEEDS])
            ps = ([(s, r) for s in SEEDS for r in C_REPEATS] if exp=='C' else [(s, 0) for s in SEEDS])
            bs = [pd.read_csv(OUT / (f'{exp}_w{w:02d}_s{s}' + (f'_r{r}' if exp=='C' else '') + '_daily.csv'),
                             parse_dates=['date']) for s, r in ps]
            b = average_daily(bs)
            delta = compare_daily(b, a)
            name = f'{exp}_w{w:02d}_minus_original'
            paired[name] = {'daily_mean_over_seed_and_repetitions_hac':delta,
                            'original_rankic_mean':float(a.rankic.mean()),
                            'modified_rankic_mean':float(b.rankic.mean()),
                            'n_seeds':len(SEEDS),'permutation_repeats':len(C_REPEATS) if exp=='C' else 1}
            all_entries.append({'experiment':exp,'window':w, 'n_seeds':3,
                                'repeats_per_seed':len(C_REPEATS) if exp=='C' else 1,
                                'rankic_original':float(a.rankic.mean()),
                                'rankic_modified':float(b.rankic.mean()),
                                'rankic_delta_modified_minus_original':delta['mean'],
                                'rankic_delta_ci95_lo':delta['ci95'][0],
                                'rankic_delta_ci95_hi':delta['ci95'][1],
                                'ic_original':float(a.ic.mean()),
                                'ic_modified':float(b.ic.mean())})
    record.sort_values(['experiment','window','seed','repeat']).to_csv(
        OUT/'v6_individual_results.csv', index=False, encoding='utf-8-sig')
    if all_entries:
        pd.DataFrame(all_entries).sort_values(['experiment','window']).to_csv(
            OUT/'v6_summary.csv', index=False, encoding='utf-8-sig')
        save_json(OUT/'v6_paired_rankic.json', {'protocol_sha256':protocol['protocol_sha256'],
                'comparisons':paired,'caveat':'VALID-only diagnostics; date-wise HAC 10 lags; not independent holdout inference'})
        print(pd.DataFrame(all_entries).to_string(index=False), flush=True)
    return len(all_entries)


def main():
    ap = argparse.ArgumentParser(description='V6 C/D order perturbation experiment, VALID only')
    ap.add_argument('--experiment', choices=['C','D','both'], default='both')
    ap.add_argument('--device', choices=['auto','mps','cpu'], default='auto')
    args = ap.parse_args()
    source = load_and_verify()
    device = torch_device(args.device)
    protocol = make_protocol(source, device)
    write_protocol_once(protocol)
    print('V6 frozen sources verified; same V2 30-day anchors, only 2021-22 validation outcomes', flush=True)
    print('Selected device:', device, '; windows:', WINDOWS, '; seeds:', SEEDS, flush=True)
    rows = []
    for w in WINDOWS:
        print(f'Preparing original {w}-day arrays...', flush=True)
        xva = materialize_cohort(source['feat'], source['va_rows'], w)
        if args.experiment in ('C','both'):
            rows.extend(c_run(w, xva, source, protocol, args.device))
        if args.experiment in ('D','both'):
            xtr = materialize_cohort(source['feat'], source['tr_rows'], w)
            rows.extend(d_run(w, xtr, xva, source, protocol, args.device))
            del xtr
        del xva
    # In case C, D were run separately, reload validated complete artifacts
    # and rebuild summary from all completed files, no need to retrain.
    all_saved = []
    for w in WINDOWS:
        for s in SEEDS:
            for r in C_REPEATS:
                p=OUT/f'C_w{w:02d}_s{s}_r{r}_metrics.json'
                if p.is_file():
                    m=json.loads(p.read_text(encoding='utf-8'))
                    if m['protocol_sha256'] != protocol['protocol_sha256']:
                        raise ValueError('V6 cached C protocol mismatch')
                    all_saved.append({'experiment':'C','window':w,'seed':s,'repeat':r,
                                      'rankic':m['rankic_daily_equal_weight']['mean'],
                                      'ic':m['ic_daily_equal_weight']['mean'], 'mse_raw':m['mse_raw'],
                                      'mean_absolute_scaled_prediction_change':m['mean_absolute_scaled_prediction_change'],
                                      'prediction_correlation':m['prediction_correlation']})
            p=OUT/f'D_w{w:02d}_s{s}_metrics.json'
            if p.is_file():
                m=json.loads(p.read_text(encoding='utf-8'))
                if m['protocol_sha256'] != protocol['protocol_sha256']:
                    raise ValueError('V6 cached D protocol mismatch')
                all_saved.append({'experiment':'D','window':w,'seed':s,'repeat':0,
                                  'rankic':m['rankic_daily_equal_weight']['mean'],
                                  'ic':m['ic_daily_equal_weight']['mean'],'mse_raw':m['mse_raw'],
                                  'best_epoch':m['model_training']['best_epoch_by_valid_mse']})
    completed = write_report(all_saved, source, protocol)
    print(f'V6 completed comparisons: {completed}/4; 2023-2026 outcome data not accessed', flush=True)
    if args.experiment == 'both' and completed != 4:
        raise RuntimeError('Expected four completed C/D, window10/window30 comparisons')
    print('V6_ORDER_EXPERIMENT_COMPLETE:', OUT, flush=True)
    print('Upload: v6_summary.csv, v6_individual_results.csv, v6_paired_rankic.json, run_v6.log', flush=True)

if __name__ == '__main__':
    main()
