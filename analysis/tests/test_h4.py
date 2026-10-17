import numpy as np
import pytest
from analysis.h4 import bootstrap_flips


def test_bootstrap_resamples_whole_unequal_prompts():
    # One 3-token prompt always flips, the 1-token prompt never flips.
    r=bootstrap_flips([1,1,1,0],['a','a','a','b'],seed=42)
    assert r['flip_rate']==.75
    assert r['ci_low']==0 and r['ci_high']==1
    assert r['prompts']==2
    assert r==bootstrap_flips([1,1,1,0],['a','a','a','b'],seed=42)


def test_zero_event_interval_is_degenerate():
    r=bootstrap_flips(np.zeros(4),['a','a','b','b'])
    assert r['ci_low']==r['ci_high']==0
    with pytest.raises(ValueError):bootstrap_flips([1],[])
