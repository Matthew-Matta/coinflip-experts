import numpy as np
from analysis.nccl_orders import reduction,explain,size_error


def test_order_search_includes_planted_order_and_discloses_nonuniqueness():
    parts=np.array([[1e8,1],[1,2],[-1e8,3],[1,4]],dtype=np.float32)
    actual=reduction(parts,(0,2,1,3),'sequential','fp32')
    result=explain(parts,actual,actual,'fp32',bins=2)
    assert result['candidate_count']==48
    assert result['any_order_share']==1
    planted=next(x for x in result['candidates'] if x['order']==[0,2,1,3] and x['tree']=='sequential')
    assert planted['share']==1
    assert any(x['share']<1 for x in result['candidates'])
    assert result['size_aware_error']['max_size_scaled_wire_ulp']==0


def test_size_metric_uses_partial_scale_not_cancelled_output():
    parts=np.array([[1],[-1],[1],[-1]],dtype=np.float32)
    result=size_error(np.array([2**-7],dtype=np.float32),np.zeros(1,dtype=np.float32),parts,'bf16')
    assert result['max_relative_to_sum_abs']==2**-9
    assert result['max_size_scaled_wire_ulp']==.25


def test_bf16_orders_match_explicit_torch_additions():
    import torch
    from itertools import permutations
    from analysis.nccl_orders import rounded
    parts=rounded(np.random.default_rng(7).normal(size=(4,17)).astype('f4'),'bf16')
    tensors=[torch.from_numpy(x.copy()).bfloat16() for x in parts]
    for order in permutations(range(4)):
        a,b,c,d=[tensors[i] for i in order]
        for tree,expected in [('sequential',((a+b)+c)+d),('pairwise',(a+b)+(c+d))]:
            assert np.array_equal(reduction(parts,order,tree,'bf16').view('u4'),expected.float().numpy().view('u4'))
