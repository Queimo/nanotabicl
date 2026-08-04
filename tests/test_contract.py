import torch
from nanotabicl_chem import ChemPriorConfig,sample_episode,sample_batch,rand_molecular_dataset_plain

def test_episode_batch_and_wrapper():
    cfg=ChemPriorConfig(filter_predictive_signal=False,debug=True)
    ep=sample_episode(cfg,seed=17,n_rows=64,n_train=48,n_features=32,debug=True)
    assert ep.x.shape==(64,32) and ep.y.shape==(64,) and ep.debug is not None
    b=sample_batch(2,cfg,seed=17,n_rows=64,n_train=48,n_features=32)
    assert b.x.shape==(2,64,32) and b.y_train.shape==(2,48) and b.y_query.shape==(2,16)
    d=rand_molecular_dataset_plain([0,3],[0],10,seed=1)
    assert set(d)=={'x_0','x_1','y_0'} and d['x_1'].max()<3
