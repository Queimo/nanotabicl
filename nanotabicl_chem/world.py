import json, hashlib, torch, numpy as np
from dataclasses import asdict
from .rng import rng, split_seed
from .molecule import sample_molecule
from .descriptors import library_descriptors
from .types import LatentRows, Evaluation, TargetSpec, MoleculeSpec

def canonical_json(obj): return json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False)
def canonicalize(ids, fracs, maxc):
    d={}
    for i,f in zip(ids,fracs):
        i=int(i); f=float(f)
        if i>=0 and f>0: d[i]=d.get(i,0)+f
    keys=sorted(d); vals=np.array([d[k] for k in keys],float); vals=vals/vals.sum()
    outi=np.full(maxc,-1,np.int64); outf=np.zeros(maxc,float); outi[:len(keys)]=keys; outf[:len(keys)]=vals
    return outi,outf

class ChemicalWorld:
    def __init__(self,cfg,seed,spec=None):
        self.config=cfg; self.seed=seed
        if spec is None:
            r=rng(seed); L=int(r.integers(cfg.library_size_min,cfg.library_size_max+1))
            self.library=[sample_molecule(i,cfg,split_seed(seed,'molecule_library',i)) for i in range(L)]
            md=cfg.descriptor_basis_dim; rr=rng(split_seed(seed,'mechanisms'))
            fam=list(cfg.property_family_weights); p=np.array(list(cfg.property_family_weights.values()),float); p=p/p.sum()
            self.family=str(rr.choice(fam,p=p)); self.has_pair=bool(rr.random()<cfg.pair_probability); self.has_triple=bool(rr.random()<cfg.triple_probability and cfg.max_interaction_order>=3)
            self.w_pure=torch.tensor(rr.normal(0,.5,md+3),dtype=torch.float32); self.w_unary=torch.tensor(rr.normal(0,.15,md+3),dtype=torch.float32)
            self.w_pair=torch.tensor(rr.normal(0,.08,md*3+3),dtype=torch.float32); self.w_triple=torch.tensor(rr.normal(0,.04,md*5+3),dtype=torch.float32); self.w_lat=torch.tensor(rr.normal(0,.1,4),dtype=torch.float32)
            self.noise_std=float(rr.uniform(cfg.noise_std_min,cfg.noise_std_max))
        else: self._load_spec(spec)
        self.base_descriptors=library_descriptors(self.library,cfg)
        self.spec=self.to_json(include_id=False); self.world_id=hashlib.sha256(canonical_json(self.spec).encode()).hexdigest()[:16]
    def to_json(self,include_id=True):
        d={'spec_version':self.config.spec_version,'descriptor_basis_version':'0.1.0','seed':int(self.seed),'config':self.config.to_dict(),'library':[asdict(m) for m in self.library],'property':{'family':self.family,'link':'identity','has_pair':self.has_pair,'has_triple':self.has_triple},'weights':{k:getattr(self,k).tolist() for k in ['w_pure','w_unary','w_pair','w_triple','w_lat']},'noise_std':self.noise_std,'seeds':{n:split_seed(self.seed,n) for n in ['world','molecule_library','descriptor_views','row_composition','conditions','mechanisms','measurement','schema','counterfactuals']}}
        if include_id: d['world_id']=self.world_id
        return d
    @classmethod
    def from_json(cls,world_json):
        from .config import ChemPriorConfig
        return cls(ChemPriorConfig(**world_json['config']),world_json['seed'],world_json)
    def _load_spec(self,s):
        self.library=[MoleculeSpec(m['molecule_id'],tuple(m['atom_types']),tuple(m['formal_charges']),tuple(tuple(e) for e in m['edges'])) for m in s['library']]
        self.family=s['property']['family']; self.has_pair=s['property']['has_pair']; self.has_triple=s['property']['has_triple']; self.noise_std=s['noise_std']
        for k,v in s['weights'].items(): setattr(self,k,torch.tensor(v,dtype=torch.float32))
    def sample_rows(self,n_rows,*,seed):
        r=rng(seed); ids=[]; fr=[]; cond=[]; L=len(self.library); M=self.config.max_components_per_row
        for _ in range(n_rows):
            m=int(r.integers(1,M+1)); ci=r.choice(L,size=m,replace=False); cf=r.dirichlet(r.uniform(.3,3,size=m)); oi,of=canonicalize(ci,cf,M); ids.append(oi); fr.append(of); cond.append([r.uniform(-2,2),r.normal(),0 if r.random()<.4 else r.normal()])
        return LatentRows(torch.tensor(np.array(ids),dtype=torch.long),torch.tensor(np.array(fr),dtype=torch.float32),torch.tensor(np.array(cond),dtype=torch.float32))
    def evaluate(self,rows,*,measurement_seed,return_contributions=False):
        B=self.base_descriptors; n=rows.component_ids.shape[0]; M=self.config.max_components_per_row
        pure=torch.zeros(n); unary=torch.zeros(n); pair=torch.zeros(n); triple=torch.zeros(n); latent=torch.zeros(n)
        for i in range(n):
            active=[j for j in range(M) if rows.component_ids[i,j]>=0 and rows.compositions[i,j]>0]
            C=rows.conditions[i]
            for a in active:
                b=B[rows.component_ids[i,a]]; z=torch.cat([b,C]); pure[i]+=rows.compositions[i,a]*(z@self.w_pure); unary[i]+=rows.compositions[i,a]*(torch.tanh(z)@self.w_unary)
            if self.has_pair:
                for ia,a in enumerate(active):
                    for bidx in active[ia+1:]:
                        ba=B[rows.component_ids[i,a]]; bb=B[rows.component_ids[i,bidx]]; s=torch.cat([ba+bb, torch.abs(ba-bb), ba*bb, C]); pair[i]+=rows.compositions[i,a]*rows.compositions[i,bidx]*(torch.tanh(s)@self.w_pair)
            if self.has_triple:
                for ia,a in enumerate(active):
                    for ib,bidx in enumerate(active[ia+1:],ia+1):
                        for c in active[ib+1:]:
                            X=torch.stack([B[rows.component_ids[i,a]],B[rows.component_ids[i,bidx]],B[rows.component_ids[i,c]]]); s=torch.cat([X.sum(0),(X**2).sum(0),(X**3).sum(0),X.max(0).values,X.min(0).values,C]); triple[i]+=rows.compositions[i,a]*rows.compositions[i,bidx]*rows.compositions[i,c]*(torch.tanh(s)@self.w_triple)
            latent[i]=torch.tensor([pure[i],unary[i],pair[i],triple[i]])@self.w_lat
        z=pure+unary+pair+triple+latent
        y_clean=torch.exp(torch.clamp(z,-10,10)) if self.family=='log_additive_excess' else (torch.sigmoid(z) if self.family=='bounded_response' else z)
        r=rng(measurement_seed); noise=torch.tensor(r.normal(0,self.noise_std,size=n),dtype=torch.float32); bias=torch.zeros(n); y_obs=y_clean+bias+noise
        contrib={'pure':pure,'unary':unary,'pair':pair,'triple':triple,'latent':latent,'measurement_bias':bias,'noise':noise}
        return Evaluation(y_clean,y_obs,contrib,{'latent_0':latent},TargetSpec(task=self.config.task))
    def intervene(self,rows,intervention,*,measurement_seed):
        from .interventions import SetCondition, SetComposition
        new=LatentRows(rows.component_ids.clone(),rows.compositions.clone(),rows.conditions.clone())
        if isinstance(intervention,SetCondition): new.conditions[list(intervention.row_indices),intervention.condition_index]=intervention.value
        if isinstance(intervention,SetComposition):
            oi,of=canonicalize(intervention.component_ids,intervention.fractions,self.config.max_components_per_row)
            for rix in intervention.row_indices: new.component_ids[rix]=torch.tensor(oi); new.compositions[rix]=torch.tensor(of,dtype=torch.float32)
        return self.evaluate(new,measurement_seed=measurement_seed,return_contributions=True)

def sample_world(config,*,seed): return ChemicalWorld(config,seed)
