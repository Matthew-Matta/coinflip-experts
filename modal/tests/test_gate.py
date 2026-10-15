import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('modal')
spec = importlib.util.spec_from_file_location('olmoe_gate', Path(__file__).parents[1] / 'gate_olmoe.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def test_driver_and_command():
    assert gate.driver_major('Driver Version: 580.12.01') == 580
    with pytest.raises(RuntimeError):
        gate.driver_major('no GPU')
    cmd = gate.server_command('/weights/test')
    assert cmd[cmd.index('--dtype')+1] == 'bfloat16'
    assert '--disable-radix-cache' in cmd
    assert '--enable-return-routed-experts' in cmd
    assert cmd[cmd.index('--chunked-prefill-size')+1] == '-1'


def test_prefill_route_validation():
    values = np.tile(np.arange(8), (3, 16, 1)).astype(np.int32)
    decode = lambda _: values.reshape(-1)
    assert gate.decode_routes({}, 3, decode).shape == (3, 16, 8)
    with pytest.raises(RuntimeError, match='prefill route rows'):
        gate.decode_routes({}, 2, decode)
    with pytest.raises(RuntimeError, match='duplicate'):
        gate.decode_routes({}, 3, lambda _: np.zeros((3,16,8), dtype=np.int32))
    with pytest.raises(RuntimeError, match='malformed'):
        gate.decode_routes({}, 3, lambda _: np.arange(12))


def test_failure_log(tmp_path):
    log = tmp_path / 'server.log'
    log.write_text('normal startup\nRuntimeError: no kernel image\n')
    assert 'no kernel image' in gate.log_errors(log)
