import torch
from model import NanoTabICLv2
from nanotabicl_chem import ChemPriorConfig,sample_batch

def test_model_forward_compatibility():
    b=sample_batch(2,ChemPriorConfig(filter_predictive_signal=False),seed=17,n_rows=64,n_train=48,n_features=32)
    m=NanoTabICLv2(max_classes=0,out_dim=19,embed_dim=32,col_num_blocks=1,row_num_blocks=1,icl_num_blocks=2,col_nhead=4,row_nhead=4,icl_nhead=4,n_cls_rows=16)
    pred=m(b.x,b.y_train)
    assert pred.shape==(2,16,19) and torch.isfinite(pred).all()
