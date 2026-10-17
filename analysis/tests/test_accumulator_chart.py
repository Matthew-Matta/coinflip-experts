import json
import pandas as pd
import pytest
from analysis.accumulator_chart import load_complete


def test_partial_sweep_is_never_plotted(tmp_path):
    (tmp_path/'rows.jsonl').write_text('{}\n')
    with pytest.raises(FileNotFoundError,match='not complete'):load_complete(tmp_path)


def test_completed_grid_requires_both_splits(tmp_path):
    config=dict(layers=[0],knobs={'router':'router_accum'},mantissas=[7])
    (tmp_path/'summary.json').write_text('{}')
    (tmp_path/'accumulator_width.provenance.json').write_text(json.dumps(dict(config=config)))
    rows=[dict(layer=0,knob='router',variant='e8m7',split='all',tokens=2,flips_vs_oracle=1)]
    pd.DataFrame(rows).to_parquet(tmp_path/'accumulator_width.parquet')
    with pytest.raises(ValueError,match='evaluation'):load_complete(tmp_path)
    rows.append(dict(rows[0],split='evaluation'))
    pd.DataFrame(rows).to_parquet(tmp_path/'accumulator_width.parquet')
    frame,_=load_complete(tmp_path)
    assert frame.flip_rate.tolist()==[.5]
