"""V1 frozen feature definitions. Expressions reference only information <= close(t).

Source: user's Qlib Alpha158 dump (Qlib 0.9.7), with two explicitly custom
features RET1 and RVOL20. Order is part of the model-input contract.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Factor:
    name: str
    group: str
    zh: str
    expr: str
    source: str


FACTORS = (
    Factor('RET1', '趋势与动量', '1日收盘收益率', '$close/Ref($close,1)-1', '自定义'),
    Factor('ROC5', '趋势与动量', '5日前收盘价/当日收盘价', 'Ref($close, 5)/$close', 'Alpha158'),
    Factor('ROC20', '趋势与动量', '20日前收盘价/当日收盘价', 'Ref($close, 20)/$close', 'Alpha158'),
    Factor('RANK20', '趋势与动量', '20日内收盘价的时间序列相对排名', 'Rank($close, 20)', 'Alpha158'),
    Factor('SUMD20', '趋势与动量', '20日价格增减幅度净占比', '(Sum(Greater($close-Ref($close, 1), 0), 20)-Sum(Greater(Ref($close, 1)-$close, 0), 20))/(Sum(Abs($close-Ref($close, 1)), 20)+1e-12)', 'Alpha158'),
    Factor('RVOL20', '波动与价格路径', '20日单日收益率标准差', 'Std($close/Ref($close,1)-1, 20)', '自定义'),
    Factor('STD20', '波动与价格路径', '20日收盘价水平标准差/当日收盘价', 'Std($close, 20)/$close', 'Alpha158'),
    Factor('RSQR20', '波动与价格路径', '20日收盘价线性趋势R平方', 'Rsquare($close, 20)', 'Alpha158'),
    Factor('VMA10', '成交量变化', '10日平均成交量/当日成交量', 'Mean($volume, 10)/($volume+1e-12)', 'Alpha158'),
    Factor('VSUMD20', '成交量变化', '20日成交量增减幅度净占比', '(Sum(Greater($volume-Ref($volume, 1), 0), 20)-Sum(Greater(Ref($volume, 1)-$volume, 0), 20))/(Sum(Abs($volume-Ref($volume, 1)), 20)+1e-12)', 'Alpha158'),
    Factor('VSTD20', '成交量变化', '20日成交量标准差/当日成交量', 'Std($volume, 20)/($volume+1e-12)', 'Alpha158'),
    Factor('CORD20', '量价关系', '20日价格比与对数成交量比的相关系数', 'Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), 20)', 'Alpha158'),
    Factor('KMID', 'K线及当日价格结构', '当日收盘价相对开盘价收益', '($close-$open)/$open', 'Alpha158'),
    Factor('KLEN', 'K线及当日价格结构', '当日最高最低价之差/开盘价', '($high-$low)/$open', 'Alpha158'),
    Factor('KSFT2', 'K线及当日价格结构', '收盘价在日内高低区间的偏向值', '(2*$close-$high-$low)/($high-$low+1e-12)', 'Alpha158'),
    Factor('VWAP0', 'K线及当日价格结构', '当日成交量加权均价/收盘价', '$vwap/$close', 'Alpha158'),
)
FEATURE_NAMES = tuple(x.name for x in FACTORS)
FEATURE_EXPRESSIONS = tuple(x.expr for x in FACTORS)
FEATURE_GROUPS = tuple(sorted(set(x.group for x in FACTORS)))
LABEL_EXPR = 'Ref($open,-6)/Ref($open,-1)-1'
HORIZON = 5
DEFAULT_WINDOWS = (10, 30, 60)
RAW_AUDIT = (
    ('$open', 'open_adj'),
    ('$high', 'high_adj'),
    ('$low', 'low_adj'),
    ('$close', 'close_adj'),
    ('$volume', 'volume_adj'),
    ('$factor', 'adjustment_factor'),
)
SPLITS = (
    ('train', '2016-01-01', '2020-12-31'),
    ('valid', '2021-01-01', '2022-12-31'),
    ('development', '2023-01-01', '2024-12-31'),
    ('final_holdout', '2025-01-01', '2026-11-02'),
)


def assert_contract() -> None:
    assert len(FACTORS) == len(set(FEATURE_NAMES)) == 16
    assert len([f for f in FACTORS if f.source == '自定义']) == 2
    assert set(FEATURE_NAMES) == {
        'RET1', 'ROC5', 'ROC20', 'RANK20', 'SUMD20',
        'RVOL20', 'STD20', 'RSQR20', 'VMA10', 'VSUMD20',
        'VSTD20', 'CORD20', 'KMID', 'KLEN', 'KSFT2', 'VWAP0',
    }
    assert 'CORR20' not in FEATURE_NAMES
    for f in FACTORS:
        assert 'Ref(' not in f.expr or 'Ref($' in f.expr  # formulas frozen above, inspect individually
        assert ', -' not in f.expr and ',-' not in f.expr, (f.name, f.expr)
