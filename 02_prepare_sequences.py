#!/usr/bin/env python3
"""V2: fit TRAIN-only preprocessing, persist lazy 10/30/60d sequence datasets.

Do not fit or evaluate models; never use final_holdout outcomes for decisions.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from lstm_research.feature_spec import FACTORS, FEATURE_NAMES, LABEL_EXPR, SPLITS
from lstm_research.sequences import (
    WINDOWS, build_sample_index, fit_train_only_scaler, index_for,
    scale_feature_column, validate_panel, validate_window_boundaries,
)

SOURCE = ROOT / 'output' / 'v1_data'
OUT = ROOT / 'output' / 'v2_sequences'


def ensure_source_contract(v1_manifest: dict) -> None:
    factor_contract = json.dumps([vars(x) for x in FACTORS], sort_keys=True, ensure_ascii=False)
    expected = hashlib.sha256(factor_contract.encode('utf-8')).hexdigest()
    if v1_manifest.get('feature_sha256') != expected:
        raise ValueError('V1 feature formulas differ from frozen 16-factor specification')
    if v1_manifest.get('features') != list(FEATURE_NAMES):
        raise ValueError('V1 feature ordering changed')
    if v1_manifest.get('label_formula') != LABEL_EXPR:
        raise ValueError('V1 label formula changed')
    if v1_manifest.get('split_spec') != [dict(name=n, start=s, end=e) for n,s,e in SPLITS]:
        raise ValueError('V1 split boundaries changed')


def main() -> None:
    manifest_file = SOURCE / 'v1_data_audit.json'
    panel_file = SOURCE / 'frozen_factor_panel.pkl.gz'
    if not manifest_file.exists() or not panel_file.exists():
        raise FileNotFoundError('V1 output missing; output/v1_data must contain manifest and frozen_factor_panel.pkl.gz')
    v1 = json.loads(manifest_file.read_text(encoding='utf-8'))
    ensure_source_contract(v1)
    OUT.mkdir(parents=True, exist_ok=True)

    print('[1/6] Reading and verifying V1 frozen panel...', flush=True)
    panel = validate_panel(pd.read_pickle(panel_file, compression='gzip'))
    if len(panel) != v1['panel_rows']:
        raise ValueError('V1 stored panel row count differs from audit')
    print(f'  rows={len(panel):,}, features={len(FEATURE_NAMES)}', flush=True)

    print('[2/6] Fit clipping and robust scaling on TRAIN dates ONLY...', flush=True)
    scaler = fit_train_only_scaler(panel)
    scaler_bytes = json.dumps(scaler, ensure_ascii=False, sort_keys=True)
    scaler_sha = hashlib.sha256(scaler_bytes.encode('utf-8')).hexdigest()
    (OUT / 'scaler_train_only.json').write_text(json.dumps(scaler, ensure_ascii=False, indent=2), encoding='utf-8')
    print('  scaling fitted on 2016–2020, not 2021–2026.', flush=True)

    print('[3/6] Store transformed daily rows and labels WITHOUT materializing windows...', flush=True)
    features = np.lib.format.open_memmap(OUT / 'features_scaled.npy', mode='w+',
                                         dtype=np.float32, shape=(len(panel),len(FEATURE_NAMES)))
    for j, name in enumerate(FEATURE_NAMES):
        raw = pd.to_numeric(panel[name], errors='coerce').to_numpy(dtype=np.float64)
        features[:, j] = scale_feature_column(raw, scaler['parameters'][name])
    features.flush()
    del features
    labels = pd.to_numeric(panel['label'], errors='coerce').to_numpy(dtype=np.float32)
    np.save(OUT / 'labels.npy', labels)

    print('[4/6] Create shared stock/date sample indices for 10/30/60d windows...', flush=True)
    index = build_sample_index(panel)
    index.to_pickle(OUT / 'sample_index.pkl.gz', compression='gzip')
    checkpoints = validate_window_boundaries(panel, index)
    print('  full-window integrity checks passed: every eligible 10/30/60-day window.', flush=True)

    print('[5/6] Reconcile eligible counts with audited V1 contract...', flush=True)
    counts = []
    expected_counts = {r['split']:r for r in v1['sample_eligibility']}
    for split, _start, _end in SPLITS:
        rec = {'split': split}
        for w in WINDOWS:
            n = int(len(index_for(index, split, w)))
            expected_n = expected_counts[split][f'strict_{w}_usable_label_rows']
            if n != expected_n:
                raise AssertionError(f'{split}/{w} V2 count {n} differs from V1 {expected_n}')
            rec[f'eligible_{w}'] = n
        rec['common_60_all_windows'] = rec['eligible_60']
        counts.append(rec)
        print(f'  {split:<14} 10d={rec["eligible_10"]:>7,} 30d={rec["eligible_30"]:>7,} 60d={rec["eligible_60"]:>7,}', flush=True)
    pd.DataFrame(counts).to_csv(OUT / 'sample_counts_by_window.csv', index=False, encoding='utf-8-sig')

    # Illustrative train-only 30d preview, no held-out label values read.
    from lstm_research.sequences import NumpySequenceDataset
    ds = NumpySequenceDataset(OUT, split='train', window=30)
    preview = []
    for i in range(min(5, len(ds))):
        x, y, m = ds[i]
        preview.append({'instrument':m['instrument'], 'anchor_date':str(m['datetime'].date()),
                        'shape':str(tuple(x.shape)), 'RET1_earliest_scaled':float(x[0,0]),
                        'RET1_latest_scaled':float(x[-1,0]), 'train_label':float(y)})
    pd.DataFrame(preview).to_csv(OUT / 'train_preview_30d.csv', index=False, encoding='utf-8-sig')

    print('[6/6] Write reproducibility manifest (no predictive evaluation)...', flush=True)
    manifest = {
        'status': 'V2 completed preprocessing + sample indices ONLY; no model trained',
        'source_v1_features_sha256': v1['feature_sha256'],
        'scaler_sha256': scaler_sha,
        'n_panel_rows': len(panel), 'n_features': len(FEATURE_NAMES),
        'features': list(FEATURE_NAMES), 'windows': list(WINDOWS), 'default_window': 30,
        'clip_quantiles_train_only': [.005, .995],
        'scaler_fit_rows': 'all finite observations for each factor on split=train, 2016-2020',
        'normalization': 'median and IQR after winsorizing TRAIN distribution per feature',
        'missing': 'strict full-window eligibility; 0 stored only as placeholder in invalid daily rows',
        'window_order': 'ascending timestamp, past -> anchor day t, including t',
        'common_cohort_for_window_comparison': 'eligible_60 anchors used for 10/30/60',
        'cross_split_past_context': 'historical features from preceding calendar period allowed; labels purged at t+6',
        'holdout_read_policy': 'NumpySequenceDataset refuses final_holdout unless explicit allow_final_holdout=True',
        'data_file_shapes': {'features_scaled.npy':[len(panel),len(FEATURE_NAMES)], 'labels.npy':[len(panel)]},
        'counts': counts, 'boundary_check_counts': checkpoints,
        'machine': platform.machine(), 'python':sys.version.split()[0],
        'limitations': [
            'Qlib index constituent PIT and pre-membership feature histories require additional investigation.',
            'Zero placeholders for invalid daily rows must never be exposed through eligible samples.',
            'Prices are potentially adjusted; no tradability/transaction cost backtest is certified.',
            '2025–2026 is not pristine, having been partially consulted in earlier work.',
            'No held-out predictive statistics, labels, or model performance were inspected to select this preprocessing.',
        ],
    }
    (OUT/'v2_sequence_audit.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2),encoding='utf-8')
    print(f'\nV2_SEQUENCE_BUILD_COMPLETE: {OUT}', flush=True)
    print('Send: v2_sequence_audit.json, sample_counts_by_window.csv, scaler_train_only.json, build_v2.log', flush=True)
    print('Do not upload large .npy or sample_index.pkl.gz files.', flush=True)


if __name__ == '__main__':
    main()
