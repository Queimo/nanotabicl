import argparse,json,sys,platform,torch,numpy as np
from nanotabicl_chem import ChemPriorConfig,sample_episode,sample_batch

def check(name,fn,required=True):
    try: details=fn() or {}; return name,{'required':required,'passed':True,'details':details}
    except Exception as e: return name,{'required':required,'passed':False,'details':{'error':repr(e)}}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config'); p.add_argument('--seed',type=int,default=17); p.add_argument('--episodes',type=int,default=4); p.add_argument('--output',default='artifacts/validation.json'); a=p.parse_args()
    cfg=ChemPriorConfig(debug=True,filter_predictive_signal=False)
    eps=[sample_episode(cfg,seed=a.seed+i,n_rows=64,n_train=48,n_features=32,debug=True) for i in range(a.episodes)]
    checks={}
    for n,c in [check('shape_contract',lambda:{'episodes':len(eps)} if all(e.x.shape==(64,32) and e.y.shape==(64,) and 0<e.n_train<64 for e in eps) else (_ for _ in ()).throw(AssertionError())), check('finite_values',lambda:{} if all(torch.isfinite(e.x).all() and torch.isfinite(e.y.float()).all() for e in eps) else (_ for _ in ()).throw(AssertionError())), check('composition_simplex',lambda:{} if all(torch.allclose(e.debug.compositions.sum(1),torch.ones(64),atol=1e-6) for e in eps) else (_ for _ in ()).throw(AssertionError())), check('deterministic_replay',lambda:{} if torch.equal(sample_episode(cfg,seed=a.seed,n_rows=64,n_train=48,n_features=32,debug=True).x, sample_episode(cfg,seed=a.seed,n_rows=64,n_train=48,n_features=32,debug=True).x) else (_ for _ in ()).throw(AssertionError())), check('batch_shape_contract',lambda:{} if sample_batch(2,cfg,seed=a.seed,n_rows=64,n_train=48,n_features=32).x.shape==(2,64,32) else (_ for _ in ()).throw(AssertionError()))]: checks[n]=c
    rp=sum(v['passed'] and v['required'] for v in checks.values()); rf=sum((not v['passed']) and v['required'] for v in checks.values())
    rep={'spec_version':'0.1.0','descriptor_basis_version':'0.1.0','config_sha256':cfg.sha256(),'root_seed':a.seed,'episodes':a.episodes,'environment':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'device':'cpu'},'checks':checks,'summary':{'required_passed':rp,'required_failed':rf,'optional_passed':0,'optional_failed':0,'passed':rf==0}}
    import os; os.makedirs(os.path.dirname(a.output) or '.',exist_ok=True); open(a.output,'w').write(json.dumps(rep,indent=2)); return 0 if rf==0 else 1
if __name__=='__main__': sys.exit(main())
