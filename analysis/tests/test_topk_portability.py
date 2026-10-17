from pathlib import Path
from analysis.topk_portability import probe


def test_all_tie_rows_are_reported_without_requiring_platform_equality():
    result=probe(Path('reference/tests/fixtures/bf16_topk_ties.json'))
    assert result['total_rows']==38
    assert sum(r['layer']==0 for r in result['rows'])==13
    assert sum(r['layer']==7 for r in result['rows'])==25
    assert len({(r['layer'],r['token']) for r in result['rows']})==38
    for row in result['rows']:
        assert len(set(row['stored_probs_chosen_set']))==8
        assert row['stored_probs_same_set']==(row['stored_probs_chosen_set']==row['macos_chosen_set'])
