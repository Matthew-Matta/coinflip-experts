import json
from pathlib import Path
import subprocess
import pytest
from routediff.synthetic import make_export
from evaluation.run import run


def test_tiny_blinded_pipeline(tmp_path):
    binary = Path('port/build/moe_run')
    if not binary.exists():
        pytest.fail('Build port/build/moe_run before the evaluation integration tests')
    source = make_export(tmp_path/'export', hidden=8, inter=4, experts=4, top_k=2, tokens=24)
    num = tmp_path/'num.json'
    num.write_text(json.dumps(dict(preset='hf_fp32')))
    reference = tmp_path/'reference'
    subprocess.run([str(binary), '--export', str(source), '--numerics', str(num), '--dump', str(reference)], check=True)
    result = run([source], [reference], tmp_path/'evaluation', seed=42, repeats=1, resamples=100, margin=2.0)
    settings = json.loads((tmp_path/'evaluation/settings.json').read_text())
    tolerances = json.loads((tmp_path/'evaluation/tolerances.json').read_text())
    assert settings['margin'] == 2.0
    assert tolerances['_meta']['margin'] == 2.0
    assert result['total_runs'] == 14  # 13 bug types + 1 held-out prompt
    assert result['errors'] == 0
    assert result['benign_runs'] == 1
    assert result['effective_bug_runs'] + result['no_effect_runs'] == 13
    assert (tmp_path/'evaluation/frozen.json').exists()
    assert (tmp_path/'evaluation/sealed_reports.json').exists()
    for log in (tmp_path/'evaluation/cases').glob('*/localiser.log'):
        assert 'bug_type' not in log.read_text()
        assert 'evaluation-private' not in log.read_text()


@pytest.mark.parametrize('margin', [0.0, -1.0, float('nan'), float('inf')])
def test_invalid_margin_creates_no_output(tmp_path, margin):
    out = tmp_path/'invalid'
    with pytest.raises(ValueError, match='margin'):
        run([], [], out, margin=margin)
    assert not out.exists()
