"""Frozen source equality, offline contract checks, and local raw-dump regeneration."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

from analysis import site_data as export

ROOT = export.ROOT
DATA = ROOT / 'site/data'


def read(path):
    return json.loads((ROOT / path).read_text())


def validate(value, schema, defs):
    if isinstance(schema, str):
        if '|' in schema:
            for branch in schema.split('|'):
                try:
                    validate(value, branch, defs)
                    return
                except AssertionError:
                    pass
            raise AssertionError(f'no union branch matched {schema}')
        if schema in defs:
            return validate(value, defs[schema], defs)
        types = {'integer': lambda x: type(x) is int, 'number': lambda x: type(x) in (int, float),
                 'boolean': lambda x: type(x) is bool, 'string': lambda x: isinstance(x, str),
                 'object': lambda x: isinstance(x, dict), 'null': lambda x: x is None}
        assert types[schema](value), (schema, value)
    elif isinstance(schema, list):
        assert isinstance(value, list)
        for item in value:
            validate(item, schema[0], defs)
    else:
        assert isinstance(value, dict)
        if set(schema) == {'*'}:
            for item in value.values():
                validate(item, schema['*'], defs)
        else:
            assert set(value) == set(schema), (set(value), set(schema))
            for key in schema:
                validate(value[key], schema[key], defs)


def test_contract_and_budget():
    contract = (ROOT / 'site/DATA_CONTRACT.md').read_text()
    schema = json.loads(re.search(r'```json\n(.*?)\n```', contract, re.S)[1])
    assert {p.name for p in DATA.glob('*.json')} == set(schema['files'])
    assert sum(p.stat().st_size for p in DATA.glob('*.json')) < 300000
    for name, shape in schema['files'].items():
        obj = read('site/data/' + name)
        validate(obj, shape, schema['definitions'])
        assert obj['schema_version'] == 1
        for source in obj['provenance']['sources']:
            assert not Path(source['path']).is_absolute()
            path = ROOT / source['path']
            assert re.fullmatch('[0-9a-f]{64}', source['sha256'])
            if path.exists():
                assert hashlib.sha256(path.read_bytes()).hexdigest() == source['sha256'], path
    pipeline = read('site/data/pipeline.json')
    assert [o['id'] for o in pipeline['ops']] == list(export.OPS)
    assert all(len(edge) == 2 and set(edge) <= set(export.OPS) for edge in pipeline['edges'])
    sc = read('site/data/scenarios.json')
    assert len(sc['token']['probabilities']) == 64
    assert len(set(sc['token']['top8'])) == 8
    assert sc['exact_tie']['probabilities'][0] == sc['exact_tie']['probabilities'][1]
    assert not any(x['flagged'] for x in sc['scenarios'][0]['before']['token']['ops'].values())
    assert sc['scenarios'][1]['after']['batch']['patch_check']['routes_restored']
    assert sc['scenarios'][1]['after']['token'] is None  # unavailable, never invented
    assert all(x['max_scaled_ulp'] in (0, None) for x in sc['scenarios'][2]['after']['token']['ops'].values())


def test_headlines_equal_frozen_sources():
    items = read('site/data/headline.json')['items']
    assert len(items) == 5
    for item in items:
        source = read(item['source'])
        assert item['values'] == {key: export.field(source, key) for key in item['source_fields']}


def test_copied_results():
    obj = read('site/data/census.json')
    assert obj['results'] == read('findings/census-olmoe-prefill/summary.json')['results']
    assert obj['structure'] == read('findings/census-olmoe-prefill/structure.json')['results']
    assert all(len(v['per_layer_rates']) == 16 for v in obj['structure'].values())
    obj = read('site/data/nccl.json')
    source = read('findings/nccl-validation/order-followup/summary.json')
    assert obj['order_summary'] == source['order_summary']
    assert obj['n2'] == [x for x in source['errors'] if x['n'] == 2]
    assert len(obj['n2']) == 8 and all(x['bit_exact'] for x in obj['n2'])
    assert obj['tree_order'] == [2, 3, 1, 0]
    assert obj['ring_blocks'] == read('findings/nccl-validation/order-followup/position-pattern.json')['checks']
    rows = read('site/data/accumulator.json')['rows']
    assert len(rows) == 90
    for acc in ('o_proj', 'rms', 'router', 'expert', 'combine'):
        selected = [r for r in rows if r['accumulator'] == acc]
        assert {r['mantissa_bits'] for r in selected} == set(range(7, 24))
        assert len([r for r in selected if 'kahan' in r['format']]) == 1
        assert all(r['flips_total'] == r['flips_layer0'] + r['flips_layer7'] for r in selected)


def local_sources_available():
    return all((ROOT / source['path']).is_file()
               for name in ('pipeline', 'scenarios')
               for source in read(f'site/data/{name}.json')['provenance']['sources'])


@pytest.mark.skipif(not local_sources_available(), reason='local-only frozen raw dumps not installed')
def test_regeneration_byte_identical(tmp_path):
    export.generate(tmp_path)
    for original in DATA.glob('*.json'):
        assert original.read_bytes() == (tmp_path / original.name).read_bytes()


def test_scenario_reports_and_routes():
    obj = read('site/data/scenarios.json')
    for scenario in obj['scenarios']:
        path = ROOT / scenario['source_report']
        if path.exists():
            assert scenario['before']['batch'] == export.scrub(json.loads(path.read_text()))
        for state in (scenario['before'], scenario['after']):
            if state is None or state['source_runs'] is None:
                continue
            for side, directory in state['source_runs'].items():
                path = ROOT / directory / 'topk_ids.npy'
                if path.exists():
                    ids = np.load(path, mmap_mode='r')[scenario['token_index']]
                    assert state['routes'][side] == sorted(ids.astype(int).tolist())
            assert state['token']['route_sets_equal'] == (state['routes']['reference'] == state['routes']['candidate'])
    tie = obj['exact_tie']
    for platform, name in [('pytorch_macos', 'macos'), ('pytorch_linux', 'linux')]:
        source = read(f'findings/topk-portability/{name}.json')
        row = next(r for r in source['rows'] if r['layer'] == tie['layer'] and r['token'] == tie['index'])
        assert tie['route_sets'][platform] == row['stored_probs_chosen_set']
    source = read('findings/backend-pairs-v1/bf16_ties.json')['0']['examples'][0]
    assert tie['choices']['cpp'] == source['cpp_chose']
    assert tie['choices']['apple_gpu'] == source['mps_chose']
