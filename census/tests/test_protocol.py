import pytest
from census.protocol import batches,parse_prefills


def test_scheduler_evidence_multiple_batches_and_cached_tokens():
    text='Prefill batch, #new-seq: 8, #new-token: 4000, #cached-token: 0, abc\nPrefill batch [14], #new-seq: 2, #new-token: 20, #cached-token: 4,'
    assert parse_prefills(text)==[dict(sequences=8,new_tokens=4000,cached_tokens=0),dict(sequences=2,new_tokens=20,cached_tokens=4)]
    assert parse_prefills('requested_batch_size=32')==[]


def test_seeded_batch_composition_covers_every_prompt():
    a=batches(list(range(200)),32)
    assert a==batches(list(range(200)),32)
    assert sorted(i for b in a for i in b)==list(range(200))
    assert len(a[-1])==8


def test_measurement_requires_committed_hash(tmp_path,monkeypatch):
    import hashlib,json
    from census.protocol import require_committed_tokens
    p=tmp_path/'tokens.json';p.write_text('[]')
    monkeypatch.setattr('subprocess.check_output',lambda *a,**kw:'wronghash file.json')
    with pytest.raises(ValueError,match='committed HEAD'):require_committed_tokens(p)


def test_census_excess_is_paired_and_zero_bootstrap_is_disclosed():
    from analysis.census import summarize
    records=[dict(prompt_id=str(i),config=c,units=16,flips=0,r0_flips=0,any_flip=False)
             for i in range(200) for c in ('r0b','bs8','bs32')]
    result=summarize(records,resamples=100)
    assert result['results']['bs32']['excess_ci']==[0.,0.]
    assert result['h1'].startswith('FALSIFIED')
    assert 'degenerate' in result['limitation']
