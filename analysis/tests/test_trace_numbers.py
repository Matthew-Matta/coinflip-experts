from decimal import Decimal

from analysis.trace_numbers import Repository, audit, evidence, matches, tokens


def test_identifiers_and_link_targets_do_not_become_measurements():
    line='7.25% [evidence](findings/layer0/file32.json) fp32 Qwen3-30B-A3B'
    assert [item['raw'] for item in tokens(line)]==['7.25']
    assert [item['raw'] for item in tokens('250x-10^6x, 2.6-2.9x')]==['250','10^6','2.6','2.9']


def test_rounding_respects_reported_precision():
    target=dict(value=Decimal('7.25'),raw='7.25',approx=False)
    assert matches(target,Decimal('7.254'))
    assert not matches(target,Decimal('7.256'))
    assert not matches(dict(value=Decimal('0'),raw='0',approx=False),Decimal('.001'))


def test_arithmetic_stays_in_record_and_citations_stay_on_line(tmp_path):
    folder=tmp_path/'findings/example';folder.mkdir(parents=True)
    source=folder/'counts.json'
    source.write_text('{"flips":2,"tokens":4}')
    doc=tmp_path/'README.md'
    doc.write_text('50% findings/example/counts.json\n50%\n')
    result=audit(Repository(tmp_path),['README.md'])
    assert [row['status'] for row in result['rows']]==['matched','unmatched']
    assert ' / ' in result['rows'][0]['proof']
    source.write_text('[{"flips":2},{"tokens":4}]')
    result=audit(Repository(tmp_path),['README.md'])
    assert result['rows'][0]['status']=='unmatched'
    values,groups=evidence('{"flag":true,"flips":2,"tokens":4}','counts.json')
    assert len(values)==2 and len(groups)==1
