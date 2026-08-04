from dataclasses import dataclass
from typing import Literal
import torch

@dataclass(frozen=True)
class MoleculeSpec:
    molecule_id:int; atom_types:tuple[int,...]; formal_charges:tuple[int,...]; edges:tuple[tuple[int,int,int],...]
@dataclass(frozen=True)
class FeatureSpec:
    index:int; name:str; role:Literal['component_slot','mixture_summary','condition','descriptor_view','distractor','measurement_artifact']; source_ids:tuple[str,...]; transform:dict; causal_status:Literal['cause_proxy','condition','effect_proxy','confounded_proxy','distractor']; dimension_signature:tuple[int,...]
@dataclass(frozen=True)
class TargetSpec:
    name:str='property'; task:str='regression'; n_classes:int|None=None
@dataclass
class LatentRows:
    component_ids:torch.Tensor; compositions:torch.Tensor; conditions:torch.Tensor
@dataclass
class Evaluation:
    y_clean:torch.Tensor; y_observed:torch.Tensor; contributions:dict; latent_nodes:dict; target_spec:TargetSpec
@dataclass
class EpisodeDebug:
    world_spec:dict; world_json:dict; molecular_graphs:list; base_descriptors:torch.Tensor; component_ids:torch.Tensor; compositions:torch.Tensor; conditions:torch.Tensor; y_clean:torch.Tensor; y_observed:torch.Tensor; contributions:dict; latent_nodes:dict; counterfactuals:dict; rng_manifest:dict
@dataclass
class PriorEpisode:
    x:torch.Tensor; y:torch.Tensor; n_train:int; task:str; feature_metadata:list; target_metadata:TargetSpec; seed:int; world_id:str; debug:EpisodeDebug|None=None
@dataclass
class PriorBatch:
    x:torch.Tensor; y_train:torch.Tensor; y_query:torch.Tensor; n_train:int; task:str; episodes:list|None
