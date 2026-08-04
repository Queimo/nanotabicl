import json, numpy as np
from .world import ChemicalWorld
from .types import LatentRows

def write_episode_artifacts(episode, prefix):
    if episode.debug is None: raise ValueError('debug episode required')
    d=episode.debug
    np.savez(str(prefix)+'.npz', x=episode.x.cpu().numpy(), y=episode.y.cpu().numpy(), y_clean=d.y_clean.numpy(), y_observed=d.y_observed.numpy(), train_mask=(np.arange(len(episode.y))<episode.n_train), component_ids=d.component_ids.numpy(), compositions=d.compositions.numpy(), conditions=d.conditions.numpy(), **{f'contribution_{k}':v.numpy() for k,v in d.contributions.items()})
    open(str(prefix)+'.world.json','w').write(json.dumps(d.world_json,sort_keys=True,indent=2))
