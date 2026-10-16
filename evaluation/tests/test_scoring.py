from evaluation.run import choose_cases
from evaluation.scoring import counts, score_case, summarize


def test_picker_is_seeded_and_covers_registry():
    registry = [dict(type=f't{i}', injected_op='output', description='test') for i in range(13)]
    a = choose_cases(registry, ['p0','p1'], 42, 2, 0)
    assert a == choose_cases(registry, ['p0','p1'], 42, 2, 0)
    assert len({r['case_id'] for r in a}) == 26
    assert {r['bug_type'] for r in a} == {b['type'] for b in registry}


def test_no_effect_denominator_and_localization():
    rows = []
    for i, (effect, detected, localized) in enumerate(((True,True,True), (True,False,False), (False,True,False))):
        rows.append(dict(case_id=str(i), prompt_id=f'p{i}', layer=0, kind='bug', bug_type='test',
                         effect=effect, detected=detected, correct_localization=localized, error=None))
    rows.append(dict(case_id='b', prompt_id='p0', layer=0, kind='benign', effect=None,
                     detected=False, correct_localization=False, error=None))
    result = summarize(rows, 42, 100)
    assert result['detection_rate'] == .5
    assert result['localization_rate'] == 1
    assert result['no_effect_runs'] == result['no_effect_alarms'] == 1
    assert result['false_positive_rate'] == 0
    assert not result['passes_thresholds']
    assert result == summarize(rows, 42, 100)


def test_flagging_without_patch_is_not_localization():
    report = dict(flagged_ops=['output'], localized_op=None)
    truth = dict(case_id='x', prompt_id='p', layer=0, kind='bug', bug_type='combine', injected_op='output')
    row = score_case(report, truth, True)
    assert row['detected'] and not row['correct_localization']
