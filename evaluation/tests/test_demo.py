from pathlib import Path
from evaluation.demo import run


def test_offline_demo_localises_and_restores_known_precision_mismatch(tmp_path):
    result=run(tmp_path/'demo',Path('port/build/moe_run'))
    assert result['localized_op']=='router_logits'
    assert result['restored'] is True
    assert Path(result['report']).is_file()
