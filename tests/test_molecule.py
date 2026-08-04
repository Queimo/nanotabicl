import torch, numpy as np
from nanotabicl_chem import ChemPriorConfig
from nanotabicl_chem.molecule import sample_molecule,validate_molecule,permute_molecule
from nanotabicl_chem.descriptors import descriptor_basis

def test_graph_and_atom_permutation_invariance():
    cfg=ChemPriorConfig(filter_predictive_signal=False)
    m=sample_molecule(0,cfg,123); assert validate_molecule(m,cfg)
    p=np.random.default_rng(0).permutation(len(m.atom_types)).tolist()
    assert torch.allclose(descriptor_basis(m,cfg), descriptor_basis(permute_molecule(m,p),cfg), atol=1e-6)
