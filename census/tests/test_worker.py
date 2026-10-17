import json

import pandas as pd
import pytest

from census.worker import save_census


def test_successful_measurements_have_pass_provenance(tmp_path, monkeypatch):
    from coinflip import provenance
    monkeypatch.setattr(provenance,'collect',lambda config: {'config':config})
    summary=dict(status='FAIL',mode='census',phase='prefill')
    records=[dict(prompt_id='p0',config='bs8',units=16,flips=2)]

    save_census(records,tmp_path,summary)

    sidecar=json.loads((tmp_path/'census.provenance.json').read_text())
    assert sidecar['config']==dict(status='PASS',mode='census',phase='prefill')
    assert pd.read_parquet(tmp_path/'census.parquet').to_dict('records')==records
    # execute() marks completion only after every output write succeeds.
    assert summary['status']=='FAIL'


def test_write_failure_propagates_without_marking_run_pass(tmp_path, monkeypatch):
    from coinflip import provenance
    def fail(*args,**kwargs):
        raise OSError('disk full')
    monkeypatch.setattr(provenance,'save_run',fail)
    summary=dict(status='FAIL',mode='census')

    with pytest.raises(OSError,match='disk full'):
        save_census([],tmp_path,summary)

    assert summary['status']=='FAIL'
