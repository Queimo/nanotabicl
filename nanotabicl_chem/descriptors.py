import numpy as np, torch, hashlib
from .rng import rng

def descriptor_basis(m, cfg):
    n=len(m.atom_types); deg=np.zeros(n); A=np.zeros((n,n)); order={0:1,1:2,2:3,3:1}
    bos=0
    for i,j,b in m.edges: deg[i]+=1; deg[j]+=1; A[i,j]=A[j,i]=order.get(b,1); bos+=order.get(b,1)
    eig=np.linalg.eigvalsh(A) if n else np.array([0.])
    known=np.array([n,len(m.edges),sum((a+1)*2.5 for a in m.atom_types),sum(m.formal_charges),sum(abs(c) for c in m.formal_charges),np.mean(np.array(m.atom_types)!=0),deg.mean(),deg.var(),len(m.edges)-n+1,np.mean(deg>=3),np.mean(deg==1),bos,np.trace(A@A)/max(n,1),np.trace(A@A@A)/max(n,1),np.max(np.abs(eig)),np.dot(m.formal_charges,deg)/max(n,1)],float)
    known=np.array([np.log1p(max(x,0)) if i in [0,1,2,11,14] else x for i,x in enumerate(known)],float)/5.0
    seed=int.from_bytes(hashlib.blake2b(str((m.atom_types,m.formal_charges,m.edges)).encode(),digest_size=8).digest(),'little')&0x7fffffff
    r=rng(seed); anon=[]
    hist=np.bincount(m.atom_types,minlength=cfg.atom_type_count)/n
    base=np.r_[hist, np.bincount(np.clip(np.array(m.formal_charges)+2,0,4),minlength=5)/n, known]
    W=r.normal(size=(cfg.anonymous_descriptor_dim,base.size))
    anon=np.tanh(W@base/np.sqrt(base.size))
    return torch.tensor(np.r_[known,anon][:cfg.descriptor_basis_dim],dtype=torch.float32)

def library_descriptors(library,cfg): return torch.stack([descriptor_basis(m,cfg) for m in library])
