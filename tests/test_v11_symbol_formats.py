"""Regression for real-world Qlib constituent symbol representations."""
import tempfile
import unittest
from pathlib import Path
from lstm_research.v11_validity import read_membership_intervals, normalize_qlib_stock_code


class V11RealMarketSymbolTests(unittest.TestCase):
    def test_known_unambiguous_qlib_format_variants(self):
        cases={
            'SH600000':'SH600000','sh600000':'SH600000',
            '\ufeffSH600000':'SH600000','SH.600000':'SH600000',
            '600000.SH':'SH600000','600000.SS':'SH600000',
            '600000.XSHG':'SH600000', 'XSHG.600000':'SH600000',
            'SZ000001':'SZ000001','000001.SZ':'SZ000001',
            'SZSE.000001':'SZ000001', '000001.XSHE':'SZ000001',
            'BJ430047':'BJ430047','430047.BJ':'BJ430047',
            '430047.XBEI':'BJ430047'
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(normalize_qlib_stock_code(raw),expected)

    def test_header_variants_and_symbol_aliases_fully_read(self):
        data=('instrument\tstart_datetime\tend_datetime\n'
              '600000.SH\t2022-01-01\t2024-12-31\n'
              'SH600000\t2023-01-01\t2024-12-31\n'
              'SZ.000001\t2023-01-01\t2024-12-31\n')
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'csi300.txt';path.write_text(data)
            ranges,meta=read_membership_intervals(path)
            self.assertEqual(meta['header_rows_recognized'],1)
            self.assertEqual(meta['noncanonical_codes_normalized'],2)
            self.assertEqual(meta['raw_interval_count'],3)
            self.assertEqual(meta['coalesced_interval_count'],2)
            self.assertEqual(set(ranges.instrument),{'SH600000','SZ000001'})

    def test_unknown_symbols_never_silently_dropped_or_guessed(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'csi300.txt'
            path.write_text('SH600000\t2023-01-01\t2023-12-31\n'
                            '600002\t2023-01-01\t2023-12-31\n'
                            'STRANGE_CODE\t2023-01-01\t2023-12-31\n')
            with self.assertRaisesRegex(ValueError,'600002') as ctx:
                read_membership_intervals(path)
            self.assertIn('STRANGE_CODE',str(ctx.exception))
            self.assertIn('2/3 rows',str(ctx.exception))

    def test_empty_and_bad_dates_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'csi300.txt'
            path.write_text('instrument start_date end_date\n')
            with self.assertRaisesRegex(ValueError,'No date-effective'):read_membership_intervals(path)
            path.write_text('SH600000 NOT_A_DATE 2024-12-31\n')
            with self.assertRaisesRegex(ValueError,'Invalid CSI300 membership dates'):
                read_membership_intervals(path)

if __name__=='__main__':unittest.main()
