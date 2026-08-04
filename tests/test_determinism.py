import torch, json
from nanotabicl_chem import ChemPriorConfig,sample_episode

def test_deterministic_cpu():
    cfg=ChemPriorConfig(debug=True,filter_predictive_signal=False)
    a=sample_episode(cfg,seed=22,n_rows=32,n_train=16,n_features=20,debug=True)
    b=sample_episode(cfg,seed=22,n_rows=32,n_train=16,n_features=20,debug=True)
    assert torch.equal(a.x,b.x) and torch.equal(a.y,b.y)
    assert json.dumps(a.debug.world_json,sort_keys=True)==json.dumps(b.debug.world_json,sort_keys=True)
