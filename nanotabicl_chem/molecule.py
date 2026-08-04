import numpy as np, torch
from .types import MoleculeSpec
from .rng import rng

def validate_molecule(m, cfg):
    n=len(m.atom_types); assert cfg.min_atoms<=n<=cfg.max_atoms; assert len(m.formal_charges)==n
    seen=set(); parent=list(range(n))
    def find(x):
        while parent[x]!=x: x=parent[x]
        return x
    for i,j,b in m.edges:
        assert 0<=i<n and 0<=j<n and i!=j and 0<=b<cfg.bond_type_count
        key=tuple(sorted((i,j))); assert key not in seen; seen.add(key)
        parent[find(i)]=find(j)
    assert len({find(i) for i in range(n)})==1
    return True

def sample_molecule(mid,cfg,seed):
    r=rng(seed); n=int(np.exp(r.uniform(np.log(cfg.min_atoms),np.log(cfg.max_atoms+1)))); n=max(cfg.min_atoms,min(cfg.max_atoms,n))
    atoms=tuple(int(x) for x in r.integers(0,cfg.atom_type_count,n)); charges=tuple(int(x) for x in r.choice([-2,-1,0,1,2],n,p=[.02,.06,.84,.06,.02]))
    deg=[0]*n; edges=[]; val=r.integers(2,7,cfg.atom_type_count)
    for j in range(1,n):
        i=int(r.integers(0,j)); b=int(r.integers(0,cfg.bond_type_count)); edges.append((min(i,j),max(i,j),b)); deg[i]+=1; deg[j]+=1
    p=float(r.uniform(.02,.12))
    for i in range(n):
        for j in range(i+1,n):
            if any(e[0]==i and e[1]==j for e in edges): continue
            if r.random()<p and deg[i]<val[atoms[i]] and deg[j]<val[atoms[j]]:
                b=int(r.integers(0,cfg.bond_type_count)); edges.append((i,j,b)); deg[i]+=1; deg[j]+=1
    m=MoleculeSpec(mid,atoms,charges,tuple(sorted(edges))); validate_molecule(m,cfg); return m

def permute_molecule(m, perm):
    inv={old:new for new,old in enumerate(perm)}
    atoms=tuple(m.atom_types[i] for i in perm); charges=tuple(m.formal_charges[i] for i in perm)
    edges=tuple(sorted((min(inv[i],inv[j]),max(inv[i],inv[j]),b) for i,j,b in m.edges))
    return MoleculeSpec(m.molecule_id,atoms,charges,edges)
