"""V11 frozen data validity functions. No inference, training, or portfolio re-selection.

Interpretation: matching date-effective local membership intervals is a *mechanical*
check, NOT independent evidence for as-of-date historical index constituents.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import re
import unicodedata
import numpy as np
import pandas as pd

from .v10_economics import MODEL_IDS, assign_quintile

# Source-specific, nonstandard Qlib identifier observed in CSI300 historical data.
# This is NOT a recognized exchange stock code and must NEVER be recoded
# (in particular, not to SH000018), dropped, or treated as PIT-certified.
# Preserve the exact source key solely for interval reconciliation.
OPAQUE_SOURCE_IDENTIFIERS = frozenset({"SHT00018"})


def normalize_qlib_stock_code(raw: str) -> str | None:
    """Canonicalize only unambiguous A-share exchange-tagged symbols.

    Accepted: Qlib SH600000, exchange-dotted SH.600000, and 600000.SH/
    600000.XSHG (with analogous SZ/BJ codes). Unadorned six-digit codes are
    *not* inferred from leading digits because that can corrupt PIT checks.
    """
    if not isinstance(raw, str):
        return None
    text = unicodedata.normalize('NFKC', raw).replace('\ufeff', '').replace('\u200b', '')
    text = text.strip().upper()
    prefixes = {'SH':'SH', 'SZ':'SZ', 'BJ':'BJ', 'XSHG':'SH',
                'XSHE':'SZ', 'XBEI':'BJ', 'SSE':'SH', 'SZSE':'SZ',
                'BSE':'BJ', 'SS':'SH', 'SHSE':'SH', 'BJSE':'BJ'}
    m = re.fullmatch(r'(SH|SZ|BJ)([0-9]{6})', text)
    if m:
        return m.group(1) + m.group(2)
    m = re.fullmatch(r'(SH|SZ|BJ|XSHG|XSHE|XBEI|SSE|SZSE|BSE|SS|SHSE|BJSE)[._-]([0-9]{6})', text)
    if m:
        return prefixes[m.group(1)] + m.group(2)
    m = re.fullmatch(r'([0-9]{6})[._-](SH|SZ|BJ|XSHG|XSHE|XBEI|SSE|SZSE|BSE|SS|SHSE|BJSE)', text)
    if m:
        return prefixes[m.group(2)] + m.group(1)
    return None


def read_membership_intervals(path: Path, *, end='2024-12-31'):
    """Read three-column date-effective Qlib constituent intervals.

    Unrecognized names are NEVER silently discarded; include their raw
    representations in an actionable error instead of hiding broken input.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    raw = pd.read_csv(p, sep=r'\s+', header=None, names=['instrument','start','end'],
                      dtype=str, comment='#', engine='python', encoding='utf-8-sig')
    if raw.empty or raw.isna().any().any() or len(raw.columns) != 3:
        raise ValueError('Missing/malformed three-column Qlib constituent intervals')
    x = raw.copy()
    # Header variants sometimes accompany manually exported instrument files.
    # Record such a header explicitly rather than silently accepting it as data.
    first = [unicodedata.normalize('NFKC', str(v)).strip().lower() for v in x.iloc[0]]
    known_header = (first[0] in {'instrument','symbol','code','ticker'} and
                    first[1] in {'start','start_date','start_datetime','start_time','begin'} and
                    first[2] in {'end','end_date','end_datetime','end_time','finish'})
    header_count = int(known_header)
    if known_header:
        x = x.iloc[1:].copy()
    if x.empty:
        raise ValueError('No date-effective CSI300 membership rows after header')
    originals = x['instrument'].copy()
    source_tokens = originals.map(lambda v: unicodedata.normalize('NFKC', str(v))
                                  .replace('\ufeff', '').replace('\u200b', '')
                                  .strip().upper())
    normalized = originals.map(normalize_qlib_stock_code)
    opaque = source_tokens.isin(OPAQUE_SOURCE_IDENTIFIERS)
    invalid = normalized.isna() & ~opaque
    if invalid.any():
        examples = list(dict.fromkeys(repr(v) for v in originals[invalid].head(12).tolist()))
        # Show precise values; do NOT skip, reinterpret, or coerce unknown codes.
        raise ValueError(
            'Unexpected Qlib instrument code: '
            f'{int(invalid.sum())}/{len(x)} rows cannot be safely normalized. '
            f'Examples={examples}. Source={p}. '
            'Expected SH600000 / SZ000001 / BJ430047 or an explicit '
            'exchange suffix such as 600000.SH. Do not delete unknown rows; '
            'share these examples for a source-specific fix.'
        )
    # Keep source-specific anomalies byte-for-byte in their canonical upper-case
    # Qlib token form. Never infer the underlying stock or remove the rows.
    normalized.loc[opaque] = source_tokens.loc[opaque]
    x['instrument'] = normalized.to_numpy()
    try:
        for name in ('start','end'):
            x[name] = pd.to_datetime(x[name], errors='raise').dt.normalize()
    except (TypeError, ValueError) as e:
        raise ValueError(f'Invalid CSI300 membership dates in {p}: {e}') from e
    if (x.start>x.end).any():
        raise ValueError('Constituent start is after end')
    # Preserve origin and exact date spans before any audit-horizon clipping.
    source_anomalies = []
    for token in sorted(x.loc[opaque, 'instrument'].unique()):
        sub = x.loc[x.instrument == token]
        eval_start = pd.Timestamp('2023-01-01')
        eval_end = pd.Timestamp('2024-12-31')
        in_eval = (sub.start <= eval_end) & (sub.end >= eval_start)
        source_anomalies.append({
            'source_identifier': token,
            'raw_interval_rows_preserved': int(len(sub)),
            'source_date_intervals': [
                {'start': str(start.date()), 'end': str(end.date())}
                for start, end in sub[['start','end']].itertuples(index=False, name=None)
            ],
            'earliest_start': str(sub.start.min().date()),
            'latest_end': str(sub.end.max().date()),
            'rows_touching_2023_2024': int(in_eval.sum()),
            'active_in_2023_2024': bool(in_eval.any()),
            'meaning': 'Unresolved source symbol; no inferred A-share security code',
        })
    orig_n=len(x)
    x=x[x.start<=pd.Timestamp(end)].copy()
    x['end']=x['end'].clip(upper=pd.Timestamp(end))
    x=x.sort_values(['instrument','start','end']).reset_index(drop=True)
    overlaps=0; coalesced=[]
    for name, group in x.groupby('instrument',sort=True):
        start=None; last=None
        for s,e in group[['start','end']].itertuples(index=False,name=None):
            if start is None:
                start,last=s,e
            elif s<=last:
                overlaps+=1
                last=max(last,e)
            else:
                coalesced.append((name,start,last))
                start,last=s,e
        if start is not None:coalesced.append((name,start,last))
    converted=int((originals.map(lambda v:unicodedata.normalize('NFKC',str(v)).strip().upper())!=normalized).sum())
    return pd.DataFrame(coalesced,columns=['instrument','start','end']), {
        'raw_interval_count':int(orig_n), 'header_rows_recognized':header_count,
        'nonstandard_source_identifiers': source_anomalies,
        'nonstandard_source_rows_preserved': int(opaque.sum()),
        'nonstandard_source_mapping_certified': False,
        'noncanonical_codes_normalized':converted,
        'interval_count_through_2024':int(len(x)),
        'coalesced_interval_count':int(len(coalesced)),
        'overlap_records':int(overlaps),
        'stock_count_through_2024':int(x.instrument.nunique())}

def membership_info(stock_dates: pd.DataFrame, intervals: pd.DataFrame,
                    calendar: pd.DatetimeIndex|None=None):
    """Stock/date effective membership; interval starts for age audit.

    Handles multiple entry/exit spans per stock without backfilled membership.
    """
    required={'instrument','datetime'}
    if not required.issubset(stock_dates.columns):raise ValueError('Missing stock/date')
    if not {'instrument','start','end'}.issubset(intervals.columns):raise ValueError('Missing intervals')
    n=len(stock_dates)
    hit=np.zeros(n,dtype=bool)
    interval_start=np.full(n,np.datetime64('NaT'),dtype='datetime64[ns]')
    ids=stock_dates.instrument.astype(str).str.upper().to_numpy()
    times=pd.to_datetime(stock_dates.datetime).to_numpy(dtype='datetime64[ns]')
    lookup={name:(g.start.to_numpy(dtype='datetime64[ns]'),g.end.to_numpy(dtype='datetime64[ns]'))
            for name,g in intervals.groupby('instrument',sort=False)}
    for name in np.unique(ids):
        rows=np.flatnonzero(ids==name)
        if name not in lookup: continue
        starts,ends=lookup[name]
        if len(starts)>1 and not np.all(starts[1:]>ends[:-1]):
            raise ValueError('Membership intervals not coalesced')
        j=np.searchsorted(starts,times[rows],side='right')-1
        valid=j>=0
        if np.any(valid):
            local=np.flatnonzero(valid); rr=rows[local]; jj=j[local]
            ok=times[rr]<=ends[jj]
            hit[rr[ok]]=True
            interval_start[rr[ok]]=starts[jj[ok]]
    age=np.full(n,-1,dtype=np.int64)
    if calendar is not None:
        cal=pd.DatetimeIndex(calendar).to_numpy(dtype='datetime64[ns]')
        if not np.all(cal[1:]>cal[:-1]):raise ValueError('Calendar must be increasing')
        k=np.flatnonzero(hit)
        current=np.searchsorted(cal,times[k],side='left')
        if np.any(current>=len(cal)) or not np.all(cal[current]==times[k]):
            raise ValueError('Stock-date missing from official calendar')
        starts=np.searchsorted(cal,interval_start[k],side='left')
        age[k]=current-starts
    return hit,age,interval_start


def daily_pool_range(intervals:pd.DataFrame,days:pd.DatetimeIndex):
    days=pd.DatetimeIndex(days)
    starts=intervals.start.to_numpy(dtype='datetime64[ns]')
    ends=intervals.end.to_numpy(dtype='datetime64[ns]')
    dayarr=days.to_numpy(dtype='datetime64[ns]')
    # Count inclusive date ranges via sorted start/end boundaries.
    ss=np.sort(starts); ee=np.sort(ends)
    active=np.searchsorted(ss,dayarr,side='right')-np.searchsorted(ee,dayarr,side='left')
    return pd.DataFrame({'datetime':days,'n_active':active.astype(int)})


def eligibility_profile(panel:pd.DataFrame, selected:pd.DataFrame):
    """Compare signal-time B30 full-window anchors vs label-finite V9 cohort.

    Panel is V1 frozen membership-filtered data. It cannot reveal stocks that
    disappeared *before* feature extraction, nor external provider PIT provenance.
    """
    req={'instrument','datetime','split','strict_window_30','within_split_label','label_finite'}
    if req-set(panel.columns):raise ValueError(f'V1 panel missing {req-set(panel.columns)}')
    x=panel.loc[(panel.split=='development') &
                pd.to_datetime(panel.datetime).between('2023-01-01','2024-12-31')].copy()
    x['datetime']=pd.to_datetime(x.datetime)
    exante=x.loc[x.strict_window_30.astype(bool)&x.within_split_label.astype(bool)].copy()
    eligible=exante.loc[exante.label_finite.astype(bool)].copy()
    if exante[['instrument','datetime']].duplicated().any():raise ValueError('Duplicate V1 signal candidates')
    lhs=pd.MultiIndex.from_frame(eligible[['instrument','datetime']])
    rhs=pd.MultiIndex.from_frame(selected[['instrument','datetime']].assign(datetime=lambda d:pd.to_datetime(d.datetime)))
    if len(lhs)!=len(rhs) or not lhs.sort_values().equals(rhs.sort_values()):
        raise ValueError('Pre-label V1 candidate and frozen V9 cohort mismatch')
    lost=exante.loc[~exante.label_finite.astype(bool)].copy()
    years=[]
    for year in (2023,2024):
        cy=exante[exante.datetime.dt.year==year]
        ly=lost[lost.datetime.dt.year==year]
        py=x[x.datetime.dt.year==year]
        years.append({'year':year,'membership_panel_rows':int(len(py)),
                      'exante_b30_candidates':int(len(cy)),
                      'retained_label_finite':int(len(cy)-len(ly)),
                      'excluded_only_due_future_label_nonfinite':int(len(ly)),
                      'future_label_excluded_fraction':float(len(ly)/len(cy)) if len(cy) else None,
                      'panel_rows_missing_b30_or_known_label_horizon':int(len(py)-len(cy))})
    return {'panel_membership_stock_dates_2023_2024':int(len(x)),
            'exante_b30_candidates':int(len(exante)),
            'selected_frozen_samples':int(len(selected)),
            'excluded_due_to_future_label_nonfinite':int(len(lost)),
            'future_label_excluded_fraction':float(len(lost)/len(exante)) if len(exante) else None,
            'pre_label_eligible_and_frozen_sample_exact_match':True,
            'by_year':years,
            'interpretation':(
              'The label-finite requirement is determined using data through t+6. '
              'A nonzero excluded count demonstrates future-availability conditioning; '
              'zero does NOT certify an unbiased dataset, since Qlib dynamic extraction '
              'may already omit instruments or date-bars. No returned samples are filtered.'
            )},exante,lost


def align_trade_fields(pred:pd.DataFrame,quotes:pd.DataFrame,cal:pd.DatetimeIndex):
    """Map current signal t to official t+1/t+6/t+5 bars. No trade simulation."""
    required={'instrument','datetime','label_raw'}
    if required-set(pred.columns):raise ValueError('Missing frozen prediction keys')
    cal=pd.DatetimeIndex(cal)
    if not np.all(cal.to_numpy()[1:]>cal.to_numpy()[:-1]):raise ValueError('Bad calendar order')
    dates=pd.to_datetime(pred.datetime).to_numpy(dtype='datetime64[ns]')
    pos=cal.get_indexer(dates)
    if np.any(pos<0) or np.any(pos+6>=len(cal)):
        raise ValueError('Missing signal/exit day')
    codes=pred.instrument.astype(str).to_numpy()
    q=quotes.copy();q['datetime']=pd.to_datetime(q.datetime)
    if q[['instrument','datetime']].duplicated().any():raise ValueError('Duplicate quotes')
    q=q.set_index(['instrument','datetime'])
    def at(offset):
        idx=pd.MultiIndex.from_arrays([codes,cal.to_numpy()[pos+offset]],
                                      names=['instrument','datetime'])
        return q.reindex(idx).reset_index(drop=True)
    sig=at(0);entry=at(1);pre_exit=at(5);exit_=at(6)
    n=len(pred)
    def fl(col,frame):return frame[col].to_numpy(dtype=np.float64)
    entry_open=fl('open_adj',entry); exit_open=fl('open_adj',exit_)
    sig_close=fl('close_adj',sig); pre_exit_close=fl('close_adj',pre_exit)
    r=np.full(n,np.nan,dtype=np.float64)
    ok=np.isfinite(entry_open)&(entry_open>0)&np.isfinite(exit_open)&(exit_open>0)
    r[ok]=exit_open[ok]/entry_open[ok]-1
    label=pred.label_raw.to_numpy(dtype=np.float64)
    mismatch=ok&(~np.isclose(r,label,atol=5e-5,rtol=0))
    if np.any(mismatch):raise ValueError('V11 frozen quote/label mismatch')
    entry_gap=np.full(n,np.nan);exit_gap=np.full(n,np.nan)
    ko=np.isfinite(sig_close)&(sig_close>0)&np.isfinite(entry_open)&(entry_open>0)
    ke=np.isfinite(pre_exit_close)&(pre_exit_close>0)&np.isfinite(exit_open)&(exit_open>0)
    entry_gap[ko]=entry_open[ko]/sig_close[ko]-1
    exit_gap[ke]=exit_open[ke]/pre_exit_close[ke]-1
    return pd.DataFrame({
        'instrument':codes,'datetime':pd.to_datetime(dates),
        'entry_open_finite_positive':np.isfinite(entry_open)&(entry_open>0),
        'exit_open_finite_positive':np.isfinite(exit_open)&(exit_open>0),
        'entry_volume_finite_positive':np.isfinite(fl('volume_adj',entry))&(fl('volume_adj',entry)>0),
        'exit_volume_finite_positive':np.isfinite(fl('volume_adj',exit_))&(fl('volume_adj',exit_)>0),
        'entry_gap':entry_gap, 'exit_gap':exit_gap,
        'entry_factor':fl('factor',entry),'exit_factor':fl('factor',exit_),
    })


def risk_flags(frame):
    e=frame.entry_gap.to_numpy(dtype=float)
    x=frame.exit_gap.to_numpy(dtype=float)
    return {
        'entry_gap_up_9_8pct':np.isfinite(e)&(e>=.098),
        'entry_gap_down_9_8pct':np.isfinite(e)&(e<=-.098),
        'entry_gap_abs_19_8pct':np.isfinite(e)&(np.abs(e)>=.198),
        'exit_gap_abs_9_8pct':np.isfinite(x)&(np.abs(x)>=.098),
        'entry_quote_invalid':~(frame.entry_open_finite_positive.to_numpy(dtype=bool)&
                                frame.entry_volume_finite_positive.to_numpy(dtype=bool)),
        'exit_quote_invalid':~(frame.exit_open_finite_positive.to_numpy(dtype=bool)&
                               frame.exit_volume_finite_positive.to_numpy(dtype=bool)),
    }


def risk_counts(flags,subset):
    n=int(np.sum(subset))
    return {'n':n,**{k:{'count':int(np.sum(v&subset)),
                       'rate':float(np.sum(v&subset)/n) if n else None} for k,v in flags.items()}}


def q5_risk_by_model(pred,flags):
    group_rows=defaultdict(list)
    for date, idx in pred.groupby('datetime',sort=True).indices.items():
        indices=np.asarray(idx,dtype=int)
        codes=pred.instrument.iloc[indices].to_numpy(dtype=str)
        for model in MODEL_IDS:
            scores=pred.iloc[indices][model+'_mean_raw_pred'].to_numpy(dtype=float)
            groups=assign_quintile(scores,codes)
            group_rows[model].append(indices[groups==5])
    n=len(pred)
    return {name:risk_counts(flags,np.isin(np.arange(n),np.concatenate(chunks)))
            for name,chunks in group_rows.items()}
