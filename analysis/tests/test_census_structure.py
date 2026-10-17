import numpy as np
from analysis.census_structure import structure


def test_structure_counts_set_replacements_and_layer_union(tmp_path):
    a=np.broadcast_to(np.arange(8,dtype=np.int32),(2,16,8)).copy();b=a.copy()
    b[0,0,-1]=8
    b[0,1,-2:]=[9,10]
    b[1,3]=b[1,3,::-1]  # order-only change is not a set flip
    for label,values in [('r0a',a),('bs8',b),('bs32',b)]:
        (tmp_path/label).mkdir();np.save(tmp_path/label/'000.npy',values)
    r=structure(tmp_path,count=1)['results']['bs8']
    assert r['differing_sets']==2
    assert r['one_expert_swaps']==1
    assert r['tokens_flipped_any_layer']==1
    assert r['tokens_flipped_all_layers']==0
    assert r['per_layer_flips'][:2]==[1,1]
