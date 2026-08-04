from dataclasses import dataclass, field, asdict
from typing import Literal
import hashlib, json

class PriorGenerationError(RuntimeError): pass

@dataclass(frozen=True)
class ChemPriorConfig:
    spec_version: str = "0.1.0"
    n_rows_min: int = 128; n_rows_max: int = 512
    train_fraction_min: float = 0.30; train_fraction_max: float = 0.90
    library_size_min: int = 8; library_size_max: int = 32
    max_components_per_row: int = 4; min_atoms: int = 2; max_atoms: int = 24
    atom_type_count: int = 16; bond_type_count: int = 4
    known_descriptor_dim: int = 16; anonymous_descriptor_dim: int = 48; descriptor_basis_dim: int = 64
    message_passing_layers: int = 3; message_passing_width: int = 16
    n_features_min: int = 16; n_features_max: int = 96
    slot_descriptor_dim: int = 8; pooled_descriptor_dim: int = 16
    include_slot_features_probability: float = 0.70; distractor_fraction_max: float = 0.25
    hidden_descriptor_fraction_min: float = 0.20; hidden_descriptor_fraction_max: float = 0.70
    max_interaction_order: int = 3; pair_probability: float = 0.85; triple_probability: float = 0.25
    interaction_rank_min: int = 1; interaction_rank_max: int = 8
    latent_nodes_min: int = 1; latent_nodes_max: int = 6
    regime_change_probability: float = 0.15; hidden_confounder_probability: float = 0.10
    property_family_weights: dict[str, float] = field(default_factory=lambda:{"additive_excess":.45,"log_additive_excess":.20,"bounded_response":.10,"generic_scm":.25})
    noise_std_min: float = 0.00; noise_std_max: float = 0.20
    heteroscedastic_probability: float = 0.30; batch_effect_probability: float = 0.10
    missing_probability_max: float = 0.00
    filter_predictive_signal: bool = False; max_generation_attempts: int = 32; min_oob_r2: float = 0.02
    task: Literal["regression","classification"] = "regression"
    n_classes_min: int = 2; n_classes_max: int = 10
    dtype: str = "float32"; device: str = "cpu"; debug: bool = False
    def __post_init__(self): self.validate()
    def validate(self):
        assert self.known_descriptor_dim + self.anonymous_descriptor_dim == self.descriptor_basis_dim
        assert 1 <= self.max_components_per_row <= self.library_size_min
        assert 1 <= self.max_interaction_order <= 3
        assert 0 < self.train_fraction_min <= self.train_fraction_max < 1
        assert self.n_features_min <= self.n_features_max
        for k,v in asdict(self).items():
            if k.endswith('probability') or k.endswith('fraction_max') or k.endswith('fraction_min'):
                if isinstance(v,(int,float)): assert 0 <= v <= 1
        return True
    def to_dict(self): return asdict(self)
    def sha256(self): return hashlib.sha256(json.dumps(self.to_dict(),sort_keys=True).encode()).hexdigest()
