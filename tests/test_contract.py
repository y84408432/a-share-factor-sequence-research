"""Pure math check, no Qlib required."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from importlib.machinery import SourceFileLoader
module = SourceFileLoader('audit', str(Path(__file__).resolve().parents[1] / '00_audit.py')).load_module()


def test_future_open_label():
    opens = [100, 101, 102, 103, 104, 105, 106]
    # signal is after day 0 close, buy on day 1 open, sell on day 6 open
    assert abs(module.label_from_opens(opens, anchor=0, horizon=5) - (106/101-1)) < 1e-14


def test_insufficient_future_history():
    try:
        module.label_from_opens([100, 101, 102], anchor=0, horizon=5)
    except IndexError:
        return
    raise AssertionError('must not form labels beyond available data')


if __name__ == '__main__':
    test_future_open_label(); test_insufficient_future_history(); print('PURE_CONTRACT_TESTS_OK')
