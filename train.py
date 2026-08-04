import argparse, torch
from model import NanoTabICLv2
from nanotabicl_chem import ChemPriorConfig,sample_batch
from nanotabicl_chem.losses import pinball_loss

def main():
    p=argparse.ArgumentParser(); p.add_argument('--steps',type=int,default=2); p.add_argument('--batch-size',type=int,default=2); p.add_argument('--n-rows',type=int,default=64); p.add_argument('--n-features',type=int,default=32); p.add_argument('--quantiles',type=int,default=19); p.add_argument('--seed',type=int,default=0); p.add_argument('--output',default='checkpoints/chem_prior_minimal.pt'); p.add_argument('--config'); p.add_argument('--task',default='regression'); p.add_argument('--device',default='cpu')
    a=p.parse_args(); cfg=ChemPriorConfig(task=a.task,filter_predictive_signal=False); levels=torch.linspace(.01,.99,a.quantiles)
    model=NanoTabICLv2(max_classes=0,out_dim=a.quantiles,embed_dim=32,col_num_blocks=1,row_num_blocks=1,icl_num_blocks=2,col_nhead=4,row_nhead=4,icl_nhead=4,n_cls_rows=16).to(a.device)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4)
    for step in range(a.steps):
        b=sample_batch(a.batch_size,cfg,seed=a.seed+step,n_rows=a.n_rows,n_train=int(a.n_rows*.75),n_features=a.n_features,device=a.device)
        loc=b.y_train.mean(1,keepdim=True); scale=b.y_train.std(1,keepdim=True)+1e-6
        pred=model(b.x,(b.y_train-loc)/scale); loss=pinball_loss(pred,(b.y_query-loc)/scale,levels.to(a.device)); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); print({'step':step,'loss':float(loss)})
    import os; os.makedirs(os.path.dirname(a.output) or '.',exist_ok=True); torch.save({'model_state_dict':model.state_dict(),'optimizer_state_dict':opt.state_dict(),'scheduler_state_dict':None,'step':a.steps,'model_config':{},'prior_config':cfg.to_dict(),'spec_version':'0.1.0','descriptor_basis_version':'0.1.0','root_seed':a.seed,'quantile_levels':levels.tolist()},a.output)
if __name__=='__main__': main()
