import pytest
from analysis.collective_cost import allreduce_cost


def test_ring_two_ranks_and_bf16_bandwidth():
    a=allreduce_cost(1024,2,'fp32','ring',1e-6,1e9)
    b=allreduce_cost(1024,2,'bf16','ring',1e-6,1e9)
    assert a['sequential_steps']==2
    assert a['per_rank_sent_bytes']==4096
    assert a['aggregate_sent_bytes']==8192
    assert b['aggregate_sent_bytes']==a['aggregate_sent_bytes']/2
    assert b['sequential_steps']==a['sequential_steps']
    assert b['predicted_seconds'] != a['predicted_seconds']/2  # latency unchanged


def test_tree_critical_path_and_one_rank():
    a=allreduce_cost(1024,8,'fp32','tree',0,1e9)
    assert a['sequential_steps']==6
    assert a['critical_path_bytes']==6*4096
    assert a['aggregate_sent_bytes']==14*4096
    assert a['per_rank_sent_bytes'] is None
    assert allreduce_cost(1024,1,'bf16','ring',1e-6,1e9)['predicted_seconds']==0
    with pytest.raises(ValueError): allreduce_cost(1,2,'bf16','ring',0,0)
