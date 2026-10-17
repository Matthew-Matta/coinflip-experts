"""The public real-v2 thresholds must stay identical to the sealed evaluation."""
import hashlib
import json
from pathlib import Path


def test_published_real_v2_tolerances_match_frozen_sha256():
    root=Path(__file__).resolve().parents[2]/'findings/routediff-eval-real-v2'
    expected='1944f35dfec6c5e3ff108f482cc47c8effb7a20e8c1d820d2d5c92d4cfd886ff'
    frozen=json.loads((root/'frozen.json').read_text())
    assert frozen['tolerances_sha256']==expected
    assert hashlib.sha256((root/'tolerances.json').read_bytes()).hexdigest()==expected
