"""V8 real-entrypoint contract regression test; NEVER reads production market data."""
from __future__ import annotations

import importlib.util
import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'08_test_a30_temporal_value.py'


def load_v8():
    spec=importlib.util.spec_from_file_location('_v8_verified_entrypoint',SCRIPT)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestV8Entrypoint(unittest.TestCase):
    def test_v7_runner_api_is_callable(self):
        mod=load_v8()
        for name in ('load_sources','ensure_a_cache','shared_runtime',
                     'materialize','canonical_sha'):
            with self.subTest(method=name):
                self.assertTrue(callable(getattr(mod.v7,name,None)),
                                f'V7 research IO interface missing: {name}')

    def test_main_reaches_experiment_steps_without_real_market_data(self):
        """Catch missing V7 entrypoint API before touching the 327k-row panel."""
        mod=load_v8()
        rows={'train':np.zeros(327555,dtype=np.int32),
              'valid':np.zeros(139220,dtype=np.int32)}
        train_x=np.broadcast_to(np.zeros((1,30,8),np.float32),(327555,30,8))
        valid_x=np.broadcast_to(np.zeros((1,30,8),np.float32),(139220,30,8))
        label_scaler={'low':-1,'high':1,'mean':0,'std':1}
        return_values=(
            {'protocol_sha256':'v7mock'}, rows,
            np.zeros(327555,np.float32), np.zeros(139220,np.float32),
            label_scaler, np.zeros(139220,dtype='datetime64[D]'),
            {}, 'original_daily_stub'
        )
        with patch.object(mod,'torch_device',return_value='cpu'), \
             patch.object(mod.v7,'load_sources',return_value='source_stub') as source, \
             patch.object(mod.v7,'ensure_a_cache',return_value={'mock':True}) as cache, \
             patch.object(mod,'validate_v7_results',return_value=return_values), \
             patch.object(mod,'write_protocol',return_value={'protocol_sha256':'mock'}), \
             patch.object(mod.np,'load',return_value='array_stub'), \
             patch.object(mod.v7,'materialize',side_effect=(train_x,valid_x)) as materialize, \
             patch.object(mod,'target_for_train',return_value=np.zeros(327555,np.float32)), \
             patch.object(mod,'target_for_valid',return_value=np.zeros(139220,np.float32)), \
             patch.object(mod,'do_order') as order, \
             patch.object(mod,'do_baselines') as baseline, \
             patch.object(mod,'finish_summary',return_value=(9,True,True)), \
             patch.object(sys,'argv',['08_test_a30_temporal_value.py','--experiment','all']):
            with redirect_stdout(io.StringIO()) as buffer:
                mod.main()
            source.assert_called_once_with()
            cache.assert_called_once_with('source_stub')
            self.assertEqual(materialize.call_count,2)
            order.assert_called_once()
            baseline.assert_called_once()
            self.assertIn('V8_A30_CONTROLS_COMPLETE',buffer.getvalue())


if __name__=='__main__':unittest.main()
