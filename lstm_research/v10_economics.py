"""V10 deterministic economic diagnostics of V9 frozen predictions.

No model selection, fitting or data access in this module. Portfolio figures
are idealized 5-day holding-period scenarios, NOT exchange-executable P&L.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MODEL_IDS = ('B1_Ridge', 'A30_Ridge', 'A30_LSTM', 'B30_LSTM')
PRED_COLS = tuple(k + '_mean_raw_pred' for k in MODEL_IDS)
QUINTILES = 5
COST_SCENARIOS = (
    ('gross_0bp', 0.0, 0.0),
    ('sym_5bp', 0.0005, 0.0005),
    ('sym_10bp', 0.001, 0.001),
    ('sym_20bp', 0.002, 0.002),
)


def validate_predictions(frame: pd.DataFrame, *, expected_rows=None, expected_days=None):
    required={'instrument','datetime','label_raw',*PRED_COLS}
    if required - set(frame.columns):
        raise ValueError(f'V9 predictions missing {required-set(frame.columns)}')
    if frame[['instrument','datetime']].isna().any().any():
        raise ValueError('Missing instrument/date')
    if frame[['instrument','datetime']].duplicated().any():
        raise ValueError('Duplicated signal stock-date')
    frame=frame.copy()
    frame['datetime']=pd.to_datetime(frame['datetime'])
    if frame.datetime.min().year<2023 or frame.datetime.max().year>2024:
        raise ValueError('V10 must use 2023-2024 development only')
    if expected_rows is not None and set(frame.datetime.dt.year.unique())!={2023,2024}:
        raise ValueError('Full V10 requires both calendar years')
    if not np.isfinite(frame[['label_raw',*PRED_COLS]].to_numpy(dtype=np.float64)).all():
        raise ValueError('Nonfinite model prediction or five-day label')
    if expected_rows is not None and len(frame)!=expected_rows:
        raise ValueError(f'V9 sample size changed: {len(frame)}')
    if expected_days is not None and frame.datetime.nunique()!=expected_days:
        raise ValueError('V9 trading-date count changed')
    counts=frame.groupby('datetime',sort=False).size()
    if (counts<(100 if expected_rows is not None else 5)).any():
        raise ValueError('Too few stocks in a development day')
    return frame.sort_values(['datetime','instrument'],kind='mergesort').reset_index(drop=True)


def assign_quintile(scores: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Assign 1=lowest through 5=highest, deterministic tie-break by ID."""
    scores=np.asarray(scores,dtype=np.float64)
    ids=np.asarray(ids,dtype=str)
    n=len(scores)
    if n<5 or len(ids)!=n or not np.isfinite(scores).all() or len(set(ids))!=n:
        raise ValueError('Invalid cross-sectional score/id')
    order=np.lexsort((ids,scores))
    groups=np.empty(n,dtype=np.int8)
    groups[order]=1+np.minimum(QUINTILES-1,(np.arange(n)*QUINTILES//n))
    return groups


def daily_quintiles(frame: pd.DataFrame):
    """For every date/model: complete 5-day quintile returns and stock baskets."""
    daily=[]
    baskets={}
    for date,df in frame.groupby('datetime',sort=True):
        y=df['label_raw'].to_numpy(dtype=np.float64)
        codes=df.instrument.to_numpy(dtype=str)
        for name in MODEL_IDS:
            scores=df[name+'_mean_raw_pred'].to_numpy(dtype=np.float64)
            q=assign_quintile(scores,codes)
            vals={i:float(np.mean(y[q==i])) for i in range(1,6)}
            count={i:int(np.sum(q==i)) for i in range(1,6)}
            daily.append({'model':name,'signal_date':date,'n_stocks':len(df),
                          **{f'Q{i}':vals[i] for i in range(1,6)},
                          **{f'Q{i}_n':count[i] for i in range(1,6)},
                          'universe':float(np.mean(y)),
                          'top_minus_bottom':vals[5]-vals[1],
                          'top_minus_universe':vals[5]-float(np.mean(y))})
            key=(name,pd.Timestamp(date))
            baskets[key]={
                'Q5':dict(zip(codes[q==5],y[q==5])),
                'Q1':dict(zip(codes[q==1],y[q==1])),
                'universe':dict(zip(codes,y)),
            }
    return pd.DataFrame(daily),baskets


def lag_hac(values: np.ndarray, lags=10):
    vals=np.asarray(values,dtype=np.float64)
    if vals.ndim!=1 or not np.isfinite(vals).all() or len(vals)<15:
        raise ValueError('HAC requires at least 15 finite days')
    n=len(vals); mean=float(vals.mean()); e=vals-mean
    var=float(e@e/n)
    for k in range(1,min(lags,n-1)+1):
        var+=2*(1-k/(min(lags,n-1)+1))*float(e[k:]@e[:-k]/n)
    se=float(np.sqrt(max(var,0.)/n))
    return {'mean':mean,'se_hac':se,'ci95_lo':mean-1.96*se,'ci95_hi':mean+1.96*se,'n_days':n}


def quintile_summaries(daily):
    rows=[]
    for name, df in daily.groupby('model',sort=False):
        df=df.sort_values('signal_date')
        for year,dd in [('2023-2024',df),('2023',df.loc[df.signal_date.dt.year==2023]),
                        ('2024',df.loc[df.signal_date.dt.year==2024])]:
            for metric in ('Q1','Q2','Q3','Q4','Q5','universe','top_minus_bottom','top_minus_universe'):
                stat=lag_hac(dd[metric].to_numpy(dtype=np.float64))
                rows.append({'model':name,'year':year,'measure':metric,'units':'5-day adjusted-open return',**stat})
    return pd.DataFrame(rows)


def assert_calendar_contiguous(signal_dates, calendar):
    dates=pd.DatetimeIndex(sorted(pd.to_datetime(signal_dates).unique()))
    cal=pd.DatetimeIndex(calendar)
    loc=cal.get_indexer(dates)
    if (loc<0).any() or not np.all(np.diff(loc)==1):
        raise ValueError('Development signal dates have gaps in the global trading calendar')
    if loc[-1]+6>=len(cal):
        raise ValueError('Calendar missing t+6 exit day')
    return dict(zip(dates,loc.tolist()))


def share_turnover(prev_holdings, new_returns):
    """Drift-aware buy/sell notional fractions when rebalancing every 5d.

    prev_holdings maps symbol -> last cycle 5d simple return; all were bought equal-weight.
    new_returns maps symbol -> next-cycle 5d simple return. The use of the
    previous 5d realized return is legitimate at the next rebalance open.
    """
    if not new_returns or not np.isfinite(np.fromiter(new_returns.values(),dtype=float)).all():
        raise ValueError('Invalid new basket')
    target={s:1/len(new_returns) for s in new_returns}
    if prev_holdings is None:
        old={}
    else:
        rr=np.fromiter(prev_holdings.values(),dtype=np.float64)
        if np.any(rr<=-1) or not np.isfinite(rr).all():
            raise ValueError('Invalid prior-period adjusted return')
        denom=float(np.sum(1+rr))
        old={s:(1+r)/denom for s,r in prev_holdings.items()}
    keys=set(old)|set(target)
    buy=float(sum(max(target.get(s,0.)-old.get(s,0.),0.) for s in keys))
    sell=float(sum(max(old.get(s,0.)-target.get(s,0.),0.) for s in keys))
    return buy,sell


def run_phase_cycles(daily, baskets, calendar):
    """Five independent 5-trading-day staggered sleeves for each long-only basket.

    Idealized every-five-day opening price execution, equal-weight selection,
    full reset and drift-aware turnover. Fixed per-side friction scenarios;
    not an order/lot/suspension-aware exchange simulation.
    """
    unique=pd.DatetimeIndex(daily.signal_date.unique()).sort_values()
    positions=assert_calendar_contiguous(unique,calendar)
    entries=[]; summaries=[]
    for name in MODEL_IDS:
        for group in ('Q5','Q1','universe'):
            for phase in range(5):
                subset=[d for d in unique if positions[d]%5==phase]
                if not subset:raise ValueError('Empty staggered cohort')
                if any(positions[b]-positions[a]!=5 for a,b in zip(subset[:-1],subset[1:])):
                    raise ValueError('Nonconsecutive 5-day rebalance cycles')
                for label,buy_cost,sell_cost in COST_SCENARIOS:
                    wealth=1.0
                    max_wealth=1.0; maxdd=0.; prior=None
                    rets=[]; turnovers=[]; trades=[]
                    for seq,date in enumerate(subset):
                        hold=baskets[(name,pd.Timestamp(date))][group]
                        buy,sell=share_turnover(prior,hold)
                        gross=float(np.mean(list(hold.values())))
                        cost_factor=1-buy_cost*buy-sell_cost*sell
                        if cost_factor<=0:raise ValueError('Cost factor invalid')
                        net=(1+gross)*cost_factor-1
                        if seq==len(subset)-1:
                            # Liquidate at the final t+6 open, then close the diagnostic sleeve.
                            net=(1+net)*(1-sell_cost)-1
                        if net<=-1:raise ValueError('Impossible net wealth')
                        wealth*=1+net
                        max_wealth=max(max_wealth,wealth)
                        maxdd=min(maxdd,wealth/max_wealth-1)
                        turns=buy+sell
                        rets.append(net); turnovers.append(turns)
                        trades.append({'model':name,'group':group,'phase':phase,'scenario':label,
                                       'signal_date':date,'entry_date':calendar[positions[date]+1],
                                       'exit_date':calendar[positions[date]+6],
                                       'n_holdings':len(hold),'gross_5d':gross,'net_5d':net,
                                       'buy_fraction':buy,'sell_fraction':sell,
                                       'two_way_turnover':turns,'wealth':wealth})
                        prior=hold
                    r=np.asarray(rets,dtype=np.float64)
                    mean=float(np.mean(r)); std=float(np.std(r,ddof=1))
                    years=5*len(subset)/252.
                    summaries.append({'model':name,'group':group,'phase':phase,'scenario':label,
                                      'n_cycles':len(subset),'n_trading_days_approx':5*len(subset),
                                      'cumulative_return':wealth-1,
                                      'annualized_return':wealth**(1/years)-1,
                                      'annualized_vol_5d':std*np.sqrt(252/5),
                                      'sharpe_proxy':mean/std*np.sqrt(252/5) if std>0 else np.nan,
                                      'max_drawdown_on_5d_marks':maxdd,
                                      'mean_two_way_turnover':float(np.mean(turnovers)),
                                      'mean_buy_frac':float(np.mean([x['buy_fraction'] for x in trades])),
                                      'mean_sell_frac':float(np.mean([x['sell_fraction'] for x in trades]))})
                    entries.extend(trades)
    return pd.DataFrame(summaries),pd.DataFrame(entries)


def quote_execution_audit(frame, quotes, calendar, *, tol=5e-5):
    """Match each signal to its t+1 and t+6 bar, without future-informed filtering.

    quote data are Qlib adjusted OHLC and volume. Volume, missing bars, large
    gap proxies are diagnostics only. Flagged observations remain in returns.
    """
    cal=pd.DatetimeIndex(calendar)
    unique=pd.DatetimeIndex(frame.datetime.unique()).sort_values()
    pos=assert_calendar_contiguous(unique,cal)
    dates=frame.datetime.to_numpy(dtype='datetime64[ns]')
    idx=np.array([pos[pd.Timestamp(x)] for x in dates],dtype=np.int64)
    entry=cal.to_numpy()[idx+1]; exit_=cal.to_numpy()[idx+6]
    codes=frame.instrument.to_numpy(dtype=str)
    q=quotes.set_index(['instrument','datetime']).sort_index()
    for field in ('open_adj','close_adj','volume_adj','factor'):
        if field not in q.columns:raise ValueError('Missing raw quote field '+field)
    start_index=pd.MultiIndex.from_arrays([codes,entry],names=['instrument','datetime'])
    end_index=pd.MultiIndex.from_arrays([codes,exit_],names=['instrument','datetime'])
    start=q.reindex(start_index);end=q.reindex(end_index)
    p0=start.open_adj.to_numpy(dtype=np.float64)
    p1=end.open_adj.to_numpy(dtype=np.float64)
    y=frame.label_raw.to_numpy(dtype=np.float64)
    bar_ok=(np.isfinite(p0)&np.isfinite(p1)&(p0>0)&(p1>0))
    v0=start.volume_adj.to_numpy(dtype=np.float64)
    v1=end.volume_adj.to_numpy(dtype=np.float64)
    volume_ok=np.isfinite(v0)&np.isfinite(v1)&(v0>0)&(v1>0)
    gap_abs=np.full(len(frame),np.nan,dtype=np.float64)
    prev_close_date=cal.to_numpy()[idx]
    prev_index=pd.MultiIndex.from_arrays([codes,prev_close_date],names=['instrument','datetime'])
    prev=q.reindex(prev_index)
    prev_close=prev.close_adj.to_numpy(dtype=np.float64)
    okprev=(np.isfinite(prev_close)&(prev_close>0)&np.isfinite(p0)&(p0>0))
    gap_abs[okprev]=np.abs(p0[okprev]/prev_close[okprev]-1)
    delta=np.full(len(frame),np.nan,dtype=np.float64)
    delta[bar_ok]=p1[bar_ok]/p0[bar_ok]-1-y[bar_ok]
    good=bar_ok & np.isfinite(delta)
    mismatch=good&(np.abs(delta)>tol)
    # Do not use quote-based subset as a backtest filter; observing sell-day
    # availability ex post would be lookahead if applied when buying.
    report={
        'n_signals':int(len(frame)),
        'entry_and_exit_open_nonmissing_positive':int(bar_ok.sum()),
        'entry_and_exit_volume_positive':int(volume_ok.sum()),
        'missing_or_invalid_entry_open':int((~np.isfinite(p0)|(p0<=0)).sum()),
        'missing_or_invalid_exit_open':int((~np.isfinite(p1)|(p1<=0)).sum()),
        'entry_volume_nonpositive_or_missing':int((~np.isfinite(v0)|(v0<=0)).sum()),
        'exit_volume_nonpositive_or_missing':int((~np.isfinite(v1)|(v1<=0)).sum()),
        'label_quote_comparable_count':int(good.sum()),
        'label_quote_mismatch_over_tolerance':int(mismatch.sum()),
        'label_quote_max_abs_difference':float(np.max(np.abs(delta[good]))) if good.any() else None,
        'entry_gap_proxy_ge_9_8pct_count':int(np.sum(gap_abs>=0.098)),
        'entry_gap_proxy_ge_19_8pct_count':int(np.sum(gap_abs>=0.198)),
        'entry_gap_proxy_definition':'abs(Qlib adjusted open(t+1) / adjusted close(t) - 1), NOT an exchange limit flag',
        'adjusted_price_note':'Qlib $open/$close are adjusted; $factor allows indicative unadjusted prices only.',
    }
    fa=start.factor.to_numpy(dtype=np.float64);fb=end.factor.to_numpy(dtype=np.float64)
    report['entry_or_exit_nonpositive_factor']=int((~np.isfinite(fa)|(fa<=0)|~np.isfinite(fb)|(fb<=0)).sum())
    if report['label_quote_comparable_count']==0 or report['label_quote_mismatch_over_tolerance']:
        raise ValueError(f'Quote/label consistency failed: {report}')
    return report
