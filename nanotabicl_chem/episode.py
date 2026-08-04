import torch, numpy as np
from .config import ChemPriorConfig
from .rng import split_seed, rng
from .world import sample_world
from .types import FeatureSpec, PriorEpisode, EpisodeDebug, PriorBatch

def observe_rows(world,rows,evaluation,n_features,seed):
    r=rng(seed); B=world.base_descriptors; n=rows.component_ids.shape[0]; M=world.config.max_components_per_row; cols=[]; meta=[]
    # conditions first
    for k,name in enumerate(['T','P','E']): cols.append(rows.conditions[:,k]); meta.append(FeatureSpec(len(meta),f'condition.{name}','condition',(name,),{},'condition',(1,)))
    # summaries
    for k in range(min(world.config.pooled_descriptor_dim,B.shape[1])):
        v=torch.zeros(n)
        for i in range(n):
            for s in range(M):
                cid=rows.component_ids[i,s]
                if cid>=0: v[i]+=rows.compositions[i,s]*B[cid,k]
        cols.append(v); meta.append(FeatureSpec(len(meta),f'mixture.mean_desc_{k}','mixture_summary',(str(k),),{},'cause_proxy',(1,)))
    # slots
    for s in range(M):
        cols.append((rows.component_ids[:,s]>=0).float()); meta.append(FeatureSpec(len(meta),f'slot_{s}.present','component_slot',(str(s),),{},'cause_proxy',(1,)))
        cols.append(rows.compositions[:,s]); meta.append(FeatureSpec(len(meta),f'slot_{s}.fraction','component_slot',(str(s),),{},'cause_proxy',(1,)))
        for k in range(world.config.slot_descriptor_dim):
            v=torch.zeros(n); mask=rows.component_ids[:,s]>=0
            if mask.any(): v[mask]=B[rows.component_ids[mask,s],k]
            cols.append(v); meta.append(FeatureSpec(len(meta),f'slot_{s}.desc_{k}','component_slot',(str(s),str(k)),{},'cause_proxy',(1,)))
    while len(cols)<n_features:
        cols.append(torch.tensor(r.normal(size=n),dtype=torch.float32)); meta.append(FeatureSpec(len(meta),f'distractor_{len(meta)}','distractor',(),{},'distractor',(1,)))
    idx=r.permutation(len(cols))[:n_features]; x=torch.stack([cols[i] for i in idx],1).float(); new=[]
    for newi,old in enumerate(idx):
        fs=meta[int(old)]; new.append(FeatureSpec(newi,fs.name,fs.role,fs.source_ids,fs.transform,fs.causal_status,fs.dimension_signature))
    return x,new

def _make_debug(world,rows,ev,seed):
    cf={}
    from .interventions import SetCondition, SetComposition
    cf['condition_shift_0']=world.intervene(rows,SetCondition(tuple(range(rows.conditions.shape[0])),0,0.0),measurement_seed=split_seed(seed,'measurement')).y_clean
    cf['composition_shift_0']=world.intervene(rows,SetComposition((0,), (0,), (1.0,)),measurement_seed=split_seed(seed,'measurement')).y_clean
    cf['atom_edit_0']=ev.y_clean.clone()
    return EpisodeDebug(world.to_json(),world.to_json(),world.library,world.base_descriptors,rows.component_ids,rows.compositions,rows.conditions,ev.y_clean,ev.y_observed,ev.contributions,ev.latent_nodes,cf,{n:split_seed(seed,n) for n in ['world','molecule_library','descriptor_views','row_composition','conditions','mechanisms','measurement','schema','counterfactuals']})

def sample_episode(config,*,seed,n_rows=None,n_train=None,n_features=None,device=None,debug=None):
    config.validate(); device=device or config.device; root=seed
    n_rows=n_rows or int(rng(split_seed(root,'n_rows')).integers(config.n_rows_min,config.n_rows_max+1))
    if n_train is None:
        frac=float(rng(split_seed(root,'n_train')).uniform(config.train_fraction_min,config.train_fraction_max)); n_train=max(1,min(n_rows-1,int(round(frac*n_rows))))
    n_features=n_features or int(rng(split_seed(root,'n_features')).integers(config.n_features_min,config.n_features_max+1))
    world=sample_world(config,seed=split_seed(root,'world'))
    rows=world.sample_rows(n_rows,seed=split_seed(root,'row_composition'))
    ev=world.evaluate(rows,measurement_seed=split_seed(root,'measurement'),return_contributions=True)
    x,meta=observe_rows(world,rows,ev,n_features,split_seed(root,'schema'))
    y=ev.y_observed.clone()
    if config.task=='classification':
        ncls=config.n_classes_min; q=torch.quantile(y,torch.linspace(0,1,ncls+1)[1:-1]); y=torch.bucketize(y,q).long()
    dbg=_make_debug(world,rows,ev,root) if (debug if debug is not None else config.debug) else None
    return PriorEpisode(x.to(device),y.to(device),n_train,config.task,meta,ev.target_spec,seed,world.world_id,dbg)

def sample_batch(batch_size,config,*,seed,n_rows=None,n_train=None,n_features=None,device=None,debug=False):
    eps=[sample_episode(config,seed=split_seed(seed,'batch',i),n_rows=n_rows,n_train=n_train,n_features=n_features,device=device,debug=debug) for i in range(batch_size)]
    x=torch.stack([e.x for e in eps]); nt=eps[0].n_train; y=torch.stack([e.y.float() for e in eps])
    return PriorBatch(x,y[:,:nt],y[:,nt:],nt,config.task,eps if debug else None)

def rand_molecular_dataset_plain(x_cat_sizes,y_cat_sizes,n_samples,*,seed=None,config=None):
    assert len(y_cat_sizes)==1; config=config or ChemPriorConfig(task='classification' if y_cat_sizes[0]>0 else 'regression', n_classes_min=max(2,y_cat_sizes[0] or 2), filter_predictive_signal=False)
    ep=sample_episode(config,seed=0 if seed is None else seed,n_rows=n_samples,n_train=max(1,n_samples//2),n_features=len(x_cat_sizes))
    out={}
    for j,cs in enumerate(x_cat_sizes):
        col=ep.x[:,j:j+1]
        if cs>0:
            qs=torch.quantile(col.squeeze(),torch.linspace(0,1,cs+1)[1:-1]); col=torch.bucketize(col.squeeze(),qs).view(-1,1).long()
        out[f'x_{j}']=col
    y=ep.y.view(-1,1)
    if y_cat_sizes[0]>0: y=(y.long()%y_cat_sizes[0])
    out['y_0']=y
    return out
