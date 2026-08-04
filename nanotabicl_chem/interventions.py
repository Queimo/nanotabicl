from dataclasses import dataclass
import torch, copy
@dataclass(frozen=True)
class ReplaceAtomType: molecule_id:int; atom_index:int; new_atom_type:int
@dataclass(frozen=True)
class ToggleEdge: molecule_id:int; atom_i:int; atom_j:int; bond_type:int|None
@dataclass(frozen=True)
class SetComposition: row_indices:tuple[int,...]; component_ids:tuple[int,...]; fractions:tuple[float,...]
@dataclass(frozen=True)
class SetCondition: row_indices:tuple[int,...]; condition_index:int; value:float

def corrupt_observed_feature(episode,feature_index,values):
    ep=copy.copy(episode); ep.x=episode.x.clone(); ep.x[:,feature_index]=values.to(ep.x.device).reshape(-1); return ep
