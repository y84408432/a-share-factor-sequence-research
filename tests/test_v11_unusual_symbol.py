"""Regression for the observed 7 SHT00018 historical intervals in user Qlib data."""
from __future__ import annotations
import importlib.util
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from lstm_research.v11_validity import (
    read_membership_intervals, normalize_qlib_stock_code, membership_info,
    daily_pool_range,
)


class V11HistoricalOpaqueIdentifierTests(unittest.TestCase):
    def make_source(self, lines):
        td=tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        f=Path(td.name)/'csi300.txt'
        f.write_text('\n'.join(lines)+'\n',encoding='utf-8')
        return f

    def test_observed_seven_rows_preserved_unmapped_and_older_dates(self):
        lines=['SH600000 2015-01-01 2024-12-31']
        lines += [f'SHT00018 {2006+i}-01-01 {2006+i}-12-31' for i in range(7)]
        intervals,meta=read_membership_intervals(self.make_source(lines))
        self.assertEqual(meta['raw_interval_count'],8)
        self.assertEqual(meta['nonstandard_source_rows_preserved'],7)
        self.assertEqual(meta['nonstandard_source_identifiers'][0]['source_identifier'],'SHT00018')
        self.assertFalse(meta['nonstandard_source_identifiers'][0]['active_in_2023_2024'])
        self.assertEqual(meta['nonstandard_source_identifiers'][0]['raw_interval_rows_preserved'],7)
        self.assertEqual(len(meta['nonstandard_source_identifiers'][0]['source_date_intervals']),7)
        self.assertEqual(int((intervals.instrument=='SHT00018').sum()),7)
        self.assertIsNone(normalize_qlib_stock_code('SHT00018'))
        self.assertNotIn('SH000018',set(intervals.instrument))
        result=membership_info(pd.DataFrame({
            'instrument':['SHT00018','SHT00018','SH600000'],
            'datetime':pd.to_datetime(['2010-06-01','2023-03-01','2023-03-01'])}),intervals)
        self.assertEqual(result[0].tolist(),[True,False,True])

    def test_active_nonstandard_source_is_reported_for_review_not_silently_dropped(self):
        intervals,meta=read_membership_intervals(self.make_source([
            'SHT00018 2023-01-01 2023-12-31',
            'SH600000 2023-01-01 2024-12-31']))
        self.assertTrue(meta['nonstandard_source_identifiers'][0]['active_in_2023_2024'])
        self.assertEqual(meta['nonstandard_source_identifiers'][0]['rows_touching_2023_2024'],1)
        self.assertEqual(daily_pool_range(intervals,pd.to_datetime(['2023-06-01'])).n_active.iloc[0],2)

    def test_unknown_other_symbols_still_rejected(self):
        with self.assertRaisesRegex(ValueError,'OTHER_T00018'):
            read_membership_intervals(self.make_source([
                'SHT00018 2010-01-01 2010-12-31',
                'OTHER_T00018 2010-01-01 2010-12-31']))

    def test_main_wires_anomaly_warning_and_review_guard(self):
        script=(Path(__file__).resolve().parent.parent/'11_data_validity_audit.py').read_text()
        self.assertIn("source_anomalies = interval_meta['nonstandard_source_identifiers']",script)
        self.assertIn("if member_report['unresolved_nonstandard_codes_active_in_evaluation']:",script)

if __name__=='__main__':
    unittest.main()
