# SPEC.md — Minimal Structural-Causal Molecular Prior for NanoTabICL

**Status:** Draft for implementation  
**Spec version:** `0.1.0`  
**Target repository:** [`soda-inria/nanotabicl`](https://github.com/soda-inria/nanotabicl)  
**Primary task:** molecular-mixture property regression  
**Secondary task:** molecular-mixture classification  
**Implementation language:** Python 3.11+, PyTorch 2.x, NumPy, scikit-learn  
**Normative terms:** **MUST**, **SHOULD**, and **MAY** are used as requirements levels.

---

## 1. Summary

This specification defines a minimal, implementable synthetic prior for pretraining NanoTabICL on molecular and molecular-mixture datasets.

The prior samples an entire latent **chemical world**, then generates a tabular dataset from that world:

\[
W \sim \Pi_{\mathrm{chem}},
\qquad
D = \{(X_i,Y_i)\}_{i=1}^{N} \sim p(D\mid W).
\]

A world contains:

1. a finite structural descriptor basis derived from abstract molecular graphs;
2. a typed structural causal model;
3. symmetric unary, pair, and optional three-component interaction mechanisms;
4. thermodynamic or experimental condition variables;
5. an ideal-property generator;
6. a measurement and noise model;
7. an observation schema that exposes only partial views of the latent structure.

The resulting table is passed directly to `NanoTabICLv2` without changing the model’s core tensor contract:

```python
pred = model(x_train_and_test, y_train)
```

where:

```text
x_train_and_test: float tensor [batch, n_train + n_test, n_features]
y_train:          float tensor [batch, n_train]
pred:             float tensor [batch, n_test, out_dim]
```

The implementation MUST also support a debug mode that emits machine-validatable ground truth:

- serialized world definition;
- typed causal graph;
- molecular graphs and descriptors;
- compositions and conditions;
- clean and observed targets;
- unary, pair, triple, and latent-mechanism contributions;
- counterfactual targets under valid structural interventions;
- deterministic replay information.

The design deliberately separates:

- **causal variables**, such as molecular structure and temperature;
- **interaction factors**, which are reciprocal and permutation invariant;
- **observed descriptor columns**, which are partial views of structure and are not assumed to cause one another.

---

## 2. Upstream NanoTabICL contract

The implementation MUST build on the upstream repository rather than replacing it.

NanoTabICL currently consists primarily of:

```text
model.py
prior.py
README.md
```

The upstream `NanoTabICLv2` model:

- accepts all train and query rows in one `x` tensor;
- obtains `n_train` from the length of `y`;
- standardizes each input column using training rows only;
- embeds continuous targets with linear layers when `max_classes == 0`;
- emits predictions only for rows after `n_train`;
- can emit quantile predictions for regression.

The upstream prior exposes a simple dataset-generator convention:

```python
def rand_dataset_plain(
    x_cat_sizes: list[int],
    y_cat_sizes: list[int],
    n_samples: int,
) -> dict[str, torch.Tensor]:
    ...
```

The new molecular prior MUST preserve a compatibility wrapper with this convention. It MAY add richer typed APIs around it.

### 2.1 Files that MUST remain source-compatible

The implementation MUST NOT require changing the public signature of:

```python
NanoTabICLv2.forward(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor
```

The implementation SHOULD leave upstream `model.py` unchanged for version `0.1.0`.

A future version MAY add metadata embeddings or graph tokens, but those are explicitly out of scope here.

---

## 3. Goals

### 3.1 Primary goals

The implementation MUST:

1. generate finite tabular molecular-mixture datasets suitable for NanoTabICL pretraining;
2. make every target arise from a sampled structural-causal world;
3. derive observed descriptors from molecular structure rather than treating table columns as independent causes;
4. include unary, pair, and optional triple molecular interactions;
5. ensure molecular interaction functions are invariant to component ordering;
6. contain familiar chemical mixing models as low-complexity special cases;
7. include more abstract anonymous descriptor channels and mechanisms than named contemporary descriptors;
8. produce deterministic, serializable, replayable episodes;
9. emit explicit ground truth for validation and mechanistic evaluation;
10. remain small enough for a single researcher to implement and audit.

### 3.2 Secondary goals

The implementation SHOULD:

- support pure-component and multicomponent rows in the same episode;
- support regression and classification;
- support counterfactual molecular edits;
- support hidden confounding and measurement bias in controlled proportions;
- expose known and anonymous descriptor channels;
- permit future adapters for RDKit, quantum descriptors, or user-provided descriptor matrices.

### 3.3 Non-goals

Version `0.1.0` MUST NOT claim:

- chemically accurate molecule generation;
- quantum-mechanical fidelity;
- realistic phase-equilibrium simulation;
- causal identifiability from observational data alone;
- complete coverage of all future chemistry;
- atomistic molecular dynamics;
- a new graph-neural architecture for NanoTabICL;
- compatibility with pretrained upstream NanoTabICL weights.

The phrase “superset of current chemistry” is implemented operationally, not literally: known descriptor-like channels and familiar low-order mixing laws are included as explicit low-complexity submodels, while additional anonymous channels and generic typed interaction programs provide a larger hypothesis class.

---

## 4. Conceptual model

Each dataset episode is generated by:

\[
\boxed{
\text{molecular graphs}
\rightarrow
\text{latent structural capacities}
\rightarrow
\text{interaction factors}
\rightarrow
\text{latent state}
\rightarrow
\text{ideal property}
\rightarrow
\text{measurement}
}
\]

Observed descriptor columns are generated through a separate branch:

\[
\text{molecular graphs}
\rightarrow
\text{descriptor views}
\rightarrow
X.
\]

The normative causal graph is:

```text
Molecular structure S ──► latent descriptors B ──► interaction factors I ──► latent state H ──► ideal target Y*
        │                         │                         ▲                     │                │
        │                         │                         │                     │                ▼
        └─────────────────────────┴────► observed features X            conditions C ─────► measurement Y

Protocol P ───────────────────────────────────────────────────────────────────────────────► measurement Y
```

The graph MUST NOT contain arbitrary arrows among observed descriptors merely because the descriptors are correlated.

### 4.1 Causal nodes versus factor nodes

Directed nodes represent interventions and information flow.

Reciprocal molecular interactions MUST be represented as symmetric factor functions, not arbitrary directed edges between components.

For component descriptors \(b_c\), compositions \(x_c\), and conditions \(C\), the interaction expansion is:

\[
I^{(1)}
=
\sum_c x_c\,\kappa^{(1)}(b_c,C),
\]

\[
I^{(2)}
=
\sum_{c<d} x_cx_d\,\kappa^{(2)}(b_c,b_d,C),
\]

\[
I^{(3)}
=
\sum_{c<d<e}x_cx_dx_e\,\kappa^{(3)}(b_c,b_d,b_e,C).
\]

The kernels MUST be permutation invariant in their molecular arguments.

---

## 5. Repository layout

The implementation SHOULD produce the following tree:

```text
nanotabicl/
├── model.py                         # upstream, unchanged
├── prior.py                         # upstream generic prior, unchanged
├── chem_prior.py                    # compatibility exports
├── train.py                         # minimal classification/regression trainer
├── validate_prior.py                # CLI validation runner
├── configs/
│   ├── chem_prior_minimal.yaml
│   ├── chem_prior_additive.yaml
│   ├── chem_prior_pair_only.yaml
│   └── chem_prior_stress.yaml
├── nanotabicl_chem/
│   ├── __init__.py
│   ├── config.py
│   ├── rng.py
│   ├── types.py
│   ├── molecule.py
│   ├── descriptors.py
│   ├── mechanisms.py
│   ├── world.py
│   ├── episode.py
│   ├── observe.py
│   ├── interventions.py
│   ├── serialization.py
│   ├── validation.py
│   └── losses.py
└── tests/
    ├── test_contract.py
    ├── test_determinism.py
    ├── test_molecule.py
    ├── test_invariances.py
    ├── test_mechanisms.py
    ├── test_counterfactuals.py
    ├── test_serialization.py
    ├── test_validation_cli.py
    └── test_training_smoke.py
```

For the smallest acceptable implementation, `nanotabicl_chem/` MAY initially be consolidated into fewer files. The public APIs and validation behavior in this specification MUST still be preserved.

---

## 6. Public Python API

### 6.1 Configuration

```python
from dataclasses import dataclass, field
from typing import Literal

@dataclass(frozen=True)
class ChemPriorConfig:
    spec_version: str = "0.1.0"

    # Episode size
    n_rows_min: int = 128
    n_rows_max: int = 512
    train_fraction_min: float = 0.30
    train_fraction_max: float = 0.90

    # Molecular library
    library_size_min: int = 8
    library_size_max: int = 32
    max_components_per_row: int = 4
    min_atoms: int = 2
    max_atoms: int = 24
    atom_type_count: int = 16
    bond_type_count: int = 4

    # Structural basis
    known_descriptor_dim: int = 16
    anonymous_descriptor_dim: int = 48
    descriptor_basis_dim: int = 64
    message_passing_layers: int = 3
    message_passing_width: int = 16

    # Visible schema
    n_features_min: int = 16
    n_features_max: int = 96
    slot_descriptor_dim: int = 8
    pooled_descriptor_dim: int = 16
    include_slot_features_probability: float = 0.70
    distractor_fraction_max: float = 0.25
    hidden_descriptor_fraction_min: float = 0.20
    hidden_descriptor_fraction_max: float = 0.70

    # Interactions
    max_interaction_order: int = 3
    pair_probability: float = 0.85
    triple_probability: float = 0.25
    interaction_rank_min: int = 1
    interaction_rank_max: int = 8

    # Latent SCM
    latent_nodes_min: int = 1
    latent_nodes_max: int = 6
    regime_change_probability: float = 0.15
    hidden_confounder_probability: float = 0.10

    # Property
    property_family_weights: dict[str, float] = field(default_factory=lambda: {
        "additive_excess": 0.45,
        "log_additive_excess": 0.20,
        "bounded_response": 0.10,
        "generic_scm": 0.25,
    })

    # Noise and observation
    noise_std_min: float = 0.00
    noise_std_max: float = 0.20
    heteroscedastic_probability: float = 0.30
    batch_effect_probability: float = 0.10
    missing_probability_max: float = 0.00

    # Filtering
    filter_predictive_signal: bool = True
    max_generation_attempts: int = 32
    min_oob_r2: float = 0.02

    # Task
    task: Literal["regression", "classification"] = "regression"
    n_classes_min: int = 2
    n_classes_max: int = 10

    # Runtime
    dtype: str = "float32"
    device: str = "cpu"
    debug: bool = False
```

The constructor MUST validate internal consistency. At minimum:

```text
known_descriptor_dim + anonymous_descriptor_dim == descriptor_basis_dim
1 <= max_components_per_row <= library_size_min
1 <= max_interaction_order <= 3
0 < train_fraction_min <= train_fraction_max < 1
0 <= all probabilities <= 1
n_features_min <= n_features_max
```

### 6.2 Episode generator

```python
def sample_episode(
    config: ChemPriorConfig,
    *,
    seed: int,
    n_rows: int | None = None,
    n_train: int | None = None,
    n_features: int | None = None,
    device: torch.device | str | None = None,
    debug: bool | None = None,
) -> "PriorEpisode":
    ...
```

This function MUST be deterministic on CPU for a fixed:

```text
spec version + config + seed + package version
```

### 6.3 Batched generator

```python
def sample_batch(
    batch_size: int,
    config: ChemPriorConfig,
    *,
    seed: int,
    n_rows: int | None = None,
    n_train: int | None = None,
    n_features: int | None = None,
    device: torch.device | str | None = None,
    debug: bool = False,
) -> "PriorBatch":
    ...
```

Batch generation MUST derive independent episode seeds from the batch seed through a documented seed-splitting function.

### 6.4 Upstream compatibility wrapper

```python
def rand_molecular_dataset_plain(
    x_cat_sizes: list[int],
    y_cat_sizes: list[int],
    n_samples: int,
    *,
    seed: int | None = None,
    config: ChemPriorConfig | None = None,
) -> dict[str, torch.Tensor]:
    ...
```

Rules:

- `len(x_cat_sizes)` determines the requested observed column count.
- `x_cat_sizes[j] == 0` requests a numerical feature.
- categorical input columns MAY be generated by discretizing an observed numerical descriptor.
- `len(y_cat_sizes)` MUST equal `1` in version `0.1.0`.
- `y_cat_sizes[0] == 0` requests regression.
- `y_cat_sizes[0] > 1` requests classification with that cardinality.
- output keys MUST be `x_0`, ..., `x_{d-1}`, and `y_0`.
- every numerical feature MUST have shape `[n_samples, 1]`.
- every categorical feature and categorical target MUST have integer values in `[0, cardinality - 1]` and shape `[n_samples, 1]`.

### 6.5 World-level API

```python
def sample_world(
    config: ChemPriorConfig,
    *,
    seed: int,
) -> "ChemicalWorld":
    ...

class ChemicalWorld:
    spec: "WorldSpec"

    def sample_rows(
        self,
        n_rows: int,
        *,
        seed: int,
    ) -> "LatentRows": ...

    def evaluate(
        self,
        rows: "LatentRows",
        *,
        measurement_seed: int,
        return_contributions: bool = False,
    ) -> "Evaluation": ...

    def intervene(
        self,
        rows: "LatentRows",
        intervention: "Intervention",
        *,
        measurement_seed: int,
    ) -> "Evaluation": ...
```

The world object MUST be immutable after construction.

---

## 7. Output data structures

### 7.1 `PriorEpisode`

```python
@dataclass
class PriorEpisode:
    x: torch.Tensor                 # [n_rows, n_features], float32
    y: torch.Tensor                 # [n_rows], float32 or int64
    n_train: int
    task: str
    feature_metadata: list["FeatureSpec"]
    target_metadata: "TargetSpec"
    seed: int
    world_id: str

    # Required when debug=True
    debug: "EpisodeDebug | None" = None
```

Invariants:

```text
x.ndim == 2
y.ndim == 1
0 < n_train < n_rows
x.shape[0] == y.shape[0]
x is finite
regression y is finite
classification y is integer and in range
```

### 7.2 `PriorBatch`

```python
@dataclass
class PriorBatch:
    x: torch.Tensor                 # [batch, n_rows, n_features]
    y_train: torch.Tensor           # [batch, n_train]
    y_query: torch.Tensor           # [batch, n_rows - n_train]
    n_train: int
    task: str
    episodes: list[PriorEpisode] | None
```

The batched generator MUST use common `n_rows`, `n_train`, and `n_features` across episodes in one tensor batch.

### 7.3 `EpisodeDebug`

```python
@dataclass
class EpisodeDebug:
    world_spec: "WorldSpec"
    world_json: dict
    molecular_graphs: list["MoleculeSpec"]
    base_descriptors: torch.Tensor       # [library_size, descriptor_basis_dim]
    component_ids: torch.Tensor          # [n_rows, max_components], int64; -1 for padding
    compositions: torch.Tensor           # [n_rows, max_components]
    conditions: torch.Tensor             # [n_rows, n_conditions]
    y_clean: torch.Tensor                # [n_rows]
    y_observed: torch.Tensor             # [n_rows]
    contributions: dict[str, torch.Tensor]
    latent_nodes: dict[str, torch.Tensor]
    counterfactuals: dict[str, torch.Tensor]
    rng_manifest: dict[str, int]
```

The required contribution keys are:

```text
pure
unary
pair
triple
latent
measurement_bias
noise
```

Inactive terms MUST be represented by zero tensors rather than omitted.

The numerical identity MUST hold within tolerance:

\[
Y_{\mathrm{observed}}
=
\operatorname{link}
\left(
Y_{\mathrm{pure}}
+Y_{\mathrm{unary}}
+Y_{\mathrm{pair}}
+Y_{\mathrm{triple}}
+Y_{\mathrm{latent}}
\right)
+Y_{\mathrm{measurement\ bias}}
+\epsilon.
\]

If the property family applies the link before some named terms, the debug structure MUST also contain an unambiguous expression tree so the target can be recomputed exactly.

---

## 8. Random-number generation and deterministic replay

### 8.1 Seed hierarchy

The implementation MUST NOT rely on uncontrolled global NumPy or PyTorch RNG state.

Every episode MUST derive named child seeds from the episode seed:

```text
world
molecule_library
descriptor_views
row_composition
conditions
mechanisms
measurement
schema
counterfactuals
```

Use a stable hash-based splitter:

```python
def split_seed(root_seed: int, namespace: str, index: int = 0) -> int:
    payload = f"chem-prior-0.1.0:{root_seed}:{namespace}:{index}".encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") & 0x7FFF_FFFF_FFFF_FFFF
```

### 8.2 Determinism guarantee

For CPU execution, the following MUST produce bitwise-equal arrays:

```python
e1 = sample_episode(config, seed=17, device="cpu", debug=True)
e2 = sample_episode(config, seed=17, device="cpu", debug=True)
```

GPU generation MAY be numerically reproducible rather than bitwise reproducible. Validation MUST run on CPU.

### 8.3 World identity

`world_id` MUST be the first 16 hexadecimal characters of a SHA-256 hash of canonical serialized `WorldSpec` JSON.

---

## 9. Abstract molecular graph generator

### 9.1 Purpose

Version `0.1.0` uses abstract molecular graphs to create structural dependence without requiring RDKit.

The graphs are not asserted to be chemically valid molecules. They MUST, however, satisfy graph invariants and support a future adapter from actual molecules.

### 9.2 Molecule schema

```python
@dataclass(frozen=True)
class MoleculeSpec:
    molecule_id: int
    atom_types: tuple[int, ...]              # length n_atoms
    formal_charges: tuple[int, ...]          # length n_atoms
    edges: tuple[tuple[int, int, int], ...]  # atom_i, atom_j, bond_type
```

Invariants:

- `min_atoms <= n_atoms <= max_atoms`;
- atom types are in `[0, atom_type_count - 1]`;
- bond types are in `[0, bond_type_count - 1]`;
- no self-loops;
- no duplicate undirected edges;
- the graph is connected;
- edge endpoints are valid atom indices.

### 9.3 Graph generation algorithm

For each molecule:

1. sample `n_atoms` from a truncated log-uniform distribution;
2. sample atom types from a world-level categorical distribution;
3. create a random spanning tree;
4. add extra edges with probability sampled per molecule;
5. reject extra edges that exceed an abstract valence budget;
6. sample bond types conditional on endpoint atom types;
7. sample formal charges from `{-2,-1,0,+1,+2}` with strong prior mass on zero;
8. canonicalize edge ordering.

The abstract valence budget is a sampled lookup table:

```python
valence_budget[atom_type] in {1, 2, 3, 4, 5, 6}
```

Bond order costs are:

```text
bond type 0 -> 1
bond type 1 -> 2
bond type 2 -> 3
bond type 3 -> 1
```

These types need not be named as single, double, triple, or aromatic bonds; they are abstract edge types.

### 9.4 Actual-molecule adapter boundary

A future adapter MUST implement:

```python
class MoleculeProvider(Protocol):
    def sample_library(self, n: int, *, seed: int) -> list[MoleculeSpec]: ...
```

An RDKit adapter MAY map atomic numbers, charges, and bonds into `MoleculeSpec`. The remainder of the prior MUST not depend on RDKit.

---

## 10. Structural descriptor basis

### 10.1 Design

Every molecule is mapped deterministically to a finite latent basis:

\[
b(m) \in \mathbb R^{K},
\qquad K=64\text{ by default}.
\]

The basis is divided into:

```text
known-like channels: 16
anonymous structural channels: 48
```

The target mechanism MAY use all channels. The observed table sees only sampled projections or subsets.

This distinction ensures that the observed descriptor table is not assumed to contain the complete causal state.

### 10.2 Known-like descriptor channels

The first 16 channels MUST be deterministic graph statistics with stable semantics:

| Index | Name | Definition class |
|---:|---|---|
| 0 | atom_count | number of atoms |
| 1 | edge_count | number of bonds |
| 2 | total_abstract_mass | sum of atom-type mass lookup |
| 3 | total_formal_charge | sum of formal charges |
| 4 | absolute_charge | sum of absolute formal charges |
| 5 | hetero_fraction | fraction of non-reference atom types |
| 6 | mean_degree | mean graph degree |
| 7 | degree_variance | variance of graph degree |
| 8 | cycle_rank | `n_edges - n_atoms + 1` |
| 9 | branch_fraction | fraction of nodes with degree >= 3 |
| 10 | terminal_fraction | fraction of degree-1 nodes |
| 11 | bond_order_sum | sum of bond-order costs |
| 12 | spectral_moment_2 | normalized trace of adjacency squared |
| 13 | spectral_moment_3 | normalized trace of adjacency cubed |
| 14 | spectral_radius | largest absolute adjacency eigenvalue |
| 15 | charge_degree_coupling | normalized sum of charge times degree |

All channels MUST be standardized by fixed global constants or transformed with `log1p` where needed. They MUST NOT be standardized using the current episode’s rows.

### 10.3 Anonymous channels

The remaining channels MUST be generated by a fixed, seeded invariant message-passing basis.

For layer \(\ell\):

\[
h_v^{(\ell+1)}
=
\sigma_\ell\left(
W_{\mathrm{self}}^{(\ell)}h_v^{(\ell)}
+
\sum_{u\in\mathcal N(v)}
W_{e_{uv}}^{(\ell)}h_u^{(\ell)}
+b^{(\ell)}
\right).
\]

Molecule-level channels are pooled using:

```text
sum
mean
max
variance
```

The basis weights MUST be fixed by `descriptor_basis_seed`, which is part of the spec version rather than sampled independently for every episode. This gives stable latent structural semantics across pretraining episodes.

The anonymous channels MUST be invariant to atom ordering. A test MUST verify this.

### 10.4 Projective extensibility

Descriptor basis versions MUST be append-only:

```text
basis v0.1 channels 0..63
basis v0.2 channels 0..63 unchanged; new channels start at 64
```

This is the practical projective-superset requirement. Existing submodels remain valid when the basis is enlarged.

---

## 11. Mixture row generation

### 11.1 Component library

Each world samples one molecular library shared by all rows in the episode:

```text
library_size ~ log-uniform[library_size_min, library_size_max]
```

Sharing a library gives the episode repeated molecular entities whose effects can be inferred in context.

### 11.2 Row composition

For each row:

1. sample the number of active components `m` in `[1, max_components_per_row]`;
2. sample `m` distinct molecule IDs from the world library;
3. sample concentration parameters `alpha_j`;
4. sample composition from `Dirichlet(alpha)`;
5. canonicalize duplicate molecule IDs by summing mole fractions;
6. pad unused slots with component ID `-1` and composition `0`.

Required invariant:

```python
abs(compositions[row].sum() - 1.0) <= 1e-6
```

### 11.3 Conditions

Version `0.1.0` MUST generate three continuous condition variables:

```text
temperature-like variable T
pressure-like variable P
field/protocol-like variable E
```

They are dimensionless standardized latent quantities in the generator. Feature metadata MUST describe their roles but MUST NOT falsely assign real SI units.

Recommended sampling:

```python
T ~ Uniform(-2, 2)
P ~ Normal(0, 1)
E ~ mixture of point mass at 0 and Normal(0, 1)
```

The world MAY correlate row conditions with compositions to simulate biased experimental design.

---

## 12. Interaction representation

### 12.1 Symmetric pair representation

For descriptors \(b_c,b_d\), define:

\[
s_2(b_c,b_d)
=
\left[
 b_c+b_d,
 |b_c-b_d|,
 b_c\odot b_d
\right].
\]

Any pair kernel MUST consume only a symmetric representation such as `s_2`.

### 12.2 Symmetric triple representation

For \(b_a,b_b,b_c\), define channelwise moments:

\[
s_3
=
\left[
\sum_j b_j,
\sum_j b_j^{\odot 2},
\sum_j b_j^{\odot 3},
\max_j b_j,
\min_j b_j
\right].
\]

Any triple kernel MUST consume only a symmetric representation such as `s_3`.

### 12.3 Composition weighting

The default weighting MUST be:

```python
unary_weight(c) = x_c
pair_weight(c, d) = x_c * x_d
triple_weight(a, b, c) = x_a * x_b * x_c
```

This guarantees that an interaction vanishes when any participating component has zero composition.

### 12.4 Low-rank kernels

The minimal pair kernel SHOULD be low-rank:

\[
\kappa^{(2)}(b_c,b_d,C)
=
\sum_{r=1}^{R}
 a_r(C)\,
 u_r(b_c)u_r(b_d)
+
q(s_2(b_c,b_d),C).
\]

The minimal triple kernel SHOULD use a low-rank product plus a small nonlinear residual:

\[
\kappa^{(3)}(b_a,b_b,b_c,C)
=
\sum_{r=1}^{R}
 a_r(C)
 u_r(b_a)u_r(b_b)u_r(b_c)
+
q(s_3,C).
\]

`u_r`, `a_r`, and `q` are sampled mechanism programs defined below.

---

## 13. Mechanism program prior

### 13.1 Program families

Each mechanism node MUST sample one function family from:

```text
linear
quadratic_low_rank
saturating
periodic
threshold
small_mlp
random_fourier
```

Default weights:

```yaml
linear: 0.25
quadratic_low_rank: 0.20
saturating: 0.15
periodic: 0.05
threshold: 0.10
small_mlp: 0.20
random_fourier: 0.05
```

### 13.2 Sparse input selection

Each mechanism MUST use a sampled subset of its eligible inputs.

The number of active inputs SHOULD follow a truncated log-uniform prior. Coefficients SHOULD use a spike-and-slab prior:

\[
a_j \sim (1-\rho)\delta_0+\rho\mathcal N(0,\sigma^2).
\]

### 13.3 Function definitions

#### Linear

\[
f(z)=a_0+a^\top z.
\]

#### Low-rank quadratic

\[
f(z)=a_0+a^\top z+\sum_{r=1}^{R}\lambda_r(u_r^\top z)^2.
\]

#### Saturating

\[
f(z)=a_0+\sum_j a_j\frac{z_j}{k_j+|z_j|}.
\]

#### Periodic

\[
f(z)=a_0+\sum_j a_j\sin(\omega_j z_j+\phi_j).
\]

#### Threshold

\[
f(z)=a_0+\sum_j a_j\operatorname{softplus}(s_j(z_j-t_j)).
\]

#### Small MLP

A one- or two-hidden-layer MLP with width in `[4, 32]`, fixed sampled weights, and activation from:

```text
tanh
silu
gelu
softplus
```

#### Random Fourier

\[
f(z)=a_0+\sum_{r=1}^{R}a_r\cos(\omega_r^\top z+\phi_r).
\]

### 13.4 Stability

Every mechanism output MUST be scale-controlled.

After sampling a mechanism, evaluate it on a deterministic calibration sample of 256 latent inputs. Rescale its output to have standard deviation in `[0.25, 2.0]` unless the output is intentionally constant.

Mechanisms producing non-finite values MUST be rejected and resampled.

---

## 14. Property families

### 14.1 `additive_excess`

This is the default physically motivated family.

Each molecule has a pure-component response:

\[
y_c^{\mathrm{pure}}(C)=f_{\mathrm{pure}}(b_c,C).
\]

The mixture target before the final link is:

\[
z
=
\sum_cx_cy_c^{\mathrm{pure}}
+I^{(1)}+I^{(2)}+I^{(3)}+f_H(H,C).
\]

A linear mixing law is obtained when all excess and latent terms are zero.

Pairwise excess models are obtained when `I^(3) == 0` and the pair kernel is low-order.

### 14.2 `log_additive_excess`

For positive-valued properties:

\[
\log y
=
\sum_cx_c\log(y_c^{\mathrm{pure}}+\epsilon)
+I^{(1)}+I^{(2)}+I^{(3)}+f_H(H,C).
\]

The output is:

\[
y=\exp(\operatorname{clip}(z,-10,10)).
\]

### 14.3 `bounded_response`

For fraction-like or probability-like targets:

\[
y=\sigma(z).
\]

### 14.4 `generic_scm`

A generic typed DAG maps structural aggregates, interactions, and conditions to the target. This family provides open-ended mechanisms while retaining the causal ordering.

### 14.5 Pure-component limit

For `additive_excess` and `log_additive_excess`, when one component has composition `1`, pair and triple terms MUST be exactly zero.

The resulting target MUST equal the pure-component branch plus any permitted unary, condition, latent, and measurement terms documented in the world expression.

---

## 15. Latent structural causal model

### 15.1 Node types

The sampled DAG MAY contain:

```text
STRUCTURAL_AGGREGATE
PAIR_INTERACTION
TRIPLE_INTERACTION
CONDITION
HIDDEN_CONFOUNDER
LATENT_STATE
IDEAL_TARGET
MEASUREMENT_BIAS
OBSERVED_TARGET
```

### 15.2 Type rules

The following parent relationships are allowed:

```text
STRUCTURAL_AGGREGATE -> LATENT_STATE
PAIR_INTERACTION    -> LATENT_STATE
TRIPLE_INTERACTION  -> LATENT_STATE
CONDITION           -> interaction or LATENT_STATE or IDEAL_TARGET
HIDDEN_CONFOUNDER   -> observed feature view and/or LATENT_STATE and/or MEASUREMENT_BIAS
LATENT_STATE        -> later LATENT_STATE or IDEAL_TARGET
IDEAL_TARGET        -> OBSERVED_TARGET
MEASUREMENT_BIAS    -> OBSERVED_TARGET
```

Forbidden relationships include:

```text
OBSERVED_TARGET -> molecular structure
observed descriptor -> molecular structure
query target -> context feature
later latent node -> earlier latent node
```

### 15.3 Graph sampling

Sample `n_latent_nodes` in the configured range. Nodes are topologically ordered by construction.

For each node:

1. sample 1–4 parents from eligible previous nodes;
2. sample a mechanism program;
3. sample independent exogenous noise scale;
4. calibrate output scale;
5. record the node and program in `WorldSpec`.

### 15.4 Hidden confounding

With configured probability, sample a row-level hidden variable \(U\) that affects both:

- at least one observed descriptor view or row-selection process;
- at least one latent mechanism or measurement bias.

The hidden confounder MUST be present in debug metadata and absent from `x`.

This mode is intended to prevent the model from assuming every predictive association corresponds to a manipulable feature.

---

## 16. Observed feature schema

### 16.1 Overview

The model observes a finite table, not the latent world.

Observed features are constructed from:

1. component-slot views;
2. symmetric mixture summaries;
3. condition variables;
4. transformed descriptor programs;
5. distractors and measurement artifacts.

The total number of columns MUST be in `[n_features_min, n_features_max]` unless explicitly overridden.

### 16.2 Component-slot views

For at most `max_components_per_row` slots, expose:

```text
occupancy flag
composition
selected structural descriptor channels
```

Unused slots MUST be zero-filled. NaN padding MUST NOT be used because the minimal NanoTabICL model standardizes with ordinary mean and standard deviation.

Slot ordering MUST be randomly permuted per row after target generation.

The generator target MUST remain invariant under this permutation.

The same feature columns are used for every row:

```text
slot_0.present
slot_0.x
slot_0.desc_0
...
slot_1.present
...
```

### 16.3 Symmetric mixture summaries

The visible schema SHOULD include a sampled subset of:

\[
\mu_k=\sum_cx_cb_{ck},
\]

\[
v_k=\sum_cx_c(b_{ck}-\mu_k)^2,
\]

\[
d_k=\sum_{c<d}x_cx_d|b_{ck}-b_{dk}|,
\]

plus:

```text
component count
composition entropy
maximum mole fraction
minimum nonzero mole fraction
```

These are invariant to component ordering.

### 16.4 Descriptor programs

Additional observed columns MAY be generated as low-complexity programs over latent mixture summaries:

```text
single channel
sparse linear projection
ratio with stabilized denominator
log1p(abs(.))
signed square root
rank-preserving monotonic warp
thresholded count
low-rank pair projection
```

Each feature MUST carry metadata identifying:

```python
@dataclass(frozen=True)
class FeatureSpec:
    index: int
    name: str
    role: Literal[
        "component_slot",
        "mixture_summary",
        "condition",
        "descriptor_view",
        "distractor",
        "measurement_artifact",
    ]
    source_ids: tuple[str, ...]
    transform: dict
    causal_status: Literal[
        "cause_proxy",
        "condition",
        "effect_proxy",
        "confounded_proxy",
        "distractor",
    ]
    dimension_signature: tuple[int, ...]
```

The model does not consume this metadata in version `0.1.0`; it exists for validation and future extensions.

### 16.5 Hidden descriptors

At least `hidden_descriptor_fraction_min` of latent basis channels MUST not be directly exposed in any single observed column.

No more than `hidden_descriptor_fraction_max` need be hidden.

### 16.6 Distractors

Distractor columns MAY include:

- independent Gaussian noise;
- random projections of unused structural channels;
- batch labels;
- nonlinear transformations of other observed features;
- weakly predictive but noncausal proxies.

Distractors MUST be finite and recorded in metadata.

### 16.7 Column randomization

Before returning the table, the generator MUST randomly permute feature columns and update `FeatureSpec.index`.

This prevents memorization of a fixed semantic position.

---

## 17. Measurement model

### 17.1 Clean and observed targets

The generator MUST distinguish:

```text
y_clean: ideal property generated by the structural world
y_observed: reported property after protocol, bias, and noise
```

Training uses `y_observed` by default.

### 17.2 Noise

For regression:

\[
y_{\mathrm{obs},i}=y_{\mathrm{clean},i}+b_i+\sigma_i\epsilon_i,
\qquad \epsilon_i\sim\mathcal N(0,1).
\]

Homoscedastic mode:

```python
sigma_i = sigma
```

Heteroscedastic mode:

```python
sigma_i = softplus(a + w.T @ selected_latent_state_i)
```

Noise scale MUST be calibrated relative to the clean target standard deviation.

### 17.3 Batch effects

With configured probability, rows receive a hidden laboratory or batch assignment. Each batch contributes an additive or affine bias.

Batch assignments MAY be partially exposed as an observed artifact column.

### 17.4 Classification

Classification worlds SHOULD generate class logits from latent mechanism nodes:

\[
\ell_k=f_k(H,I,C),
\qquad
Y\sim\operatorname{Categorical}(\operatorname{softmax}(\ell/\tau)).
\]

Every requested class MUST appear at least twice in the generated episode. Otherwise the episode MUST be rejected and regenerated.

---

## 18. Valid structural interventions

### 18.1 Principle

An intervention acts on molecular structure or experimental conditions, not directly on a descriptor column.

The following intervention types MUST be implemented:

```python
@dataclass(frozen=True)
class ReplaceAtomType:
    molecule_id: int
    atom_index: int
    new_atom_type: int

@dataclass(frozen=True)
class ToggleEdge:
    molecule_id: int
    atom_i: int
    atom_j: int
    bond_type: int | None

@dataclass(frozen=True)
class SetComposition:
    row_indices: tuple[int, ...]
    component_ids: tuple[int, ...]
    fractions: tuple[float, ...]

@dataclass(frozen=True)
class SetCondition:
    row_indices: tuple[int, ...]
    condition_index: int
    value: float
```

### 18.2 Structural intervention semantics

A structural intervention MUST:

1. create a new valid `MoleculeSpec`;
2. recompute all structural descriptors;
3. recompute every downstream interaction and latent state;
4. retain the same exogenous row noise where a paired counterfactual is requested;
5. leave unrelated molecules unchanged.

### 18.3 Descriptor corruption control

The implementation MUST also expose a noncausal observation corruption utility:

```python
def corrupt_observed_feature(
    episode: PriorEpisode,
    feature_index: int,
    values: torch.Tensor,
) -> PriorEpisode:
    ...
```

This changes `x` only and MUST NOT recompute `y_clean` or `y_observed`.

This distinction is required for validation and future causal tasks.

### 18.4 Default counterfactuals

In debug mode, every episode MUST include at least:

```text
atom_edit_0
composition_shift_0
condition_shift_0
```

Each counterfactual tensor MUST have shape `[n_rows]`. Rows unaffected by the intervention MAY equal the factual target.

---

## 19. Serialization

### 19.1 World JSON

`WorldSpec` MUST serialize to canonical JSON with:

- sorted keys;
- no NaN or infinity;
- floating-point values represented with sufficient precision for replay;
- explicit spec and descriptor-basis versions;
- explicit mechanism parameters;
- explicit graph edges and node types;
- explicit seed manifest.

### 19.2 Episode artifacts

The sampling CLI MUST be able to write:

```text
episode_seed_<seed>.npz
episode_seed_<seed>.world.json
episode_seed_<seed>.validation.json
```

The `.npz` file MUST contain:

```text
x
y
y_clean
y_observed
train_mask
component_ids
compositions
conditions
contribution_pure
contribution_unary
contribution_pair
contribution_triple
contribution_latent
contribution_measurement_bias
contribution_noise
```

### 19.3 Round trip

The following MUST reproduce `y_clean` within `1e-6` absolute tolerance:

```python
world = ChemicalWorld.from_json(world_json)
rows = LatentRows.from_npz(npz)
y = world.evaluate(rows, measurement_seed=manifest["measurement"]).y_clean
```

---

## 20. Regression target normalization and loss

NanoTabICL standardizes `x` internally but does not standardize regression targets. The training loop MUST standardize target values per episode using context rows only.

For episode `b`:

\[
\mu_b=\operatorname{mean}(y_{b,1:n_{train}}),
\]

\[
s_b=\operatorname{std}(y_{b,1:n_{train}})+10^{-6},
\]

\[
\tilde y_{bi}=\frac{y_{bi}-\mu_b}{s_b}.
\]

The model receives `y_train_standardized`.

### 20.1 Quantile outputs

The minimal default SHOULD use 99 quantiles for inexpensive experimentation:

```python
quantile_levels = torch.linspace(0.01, 0.99, 99)
model = NanoTabICLv2(max_classes=0, out_dim=99, ...)
```

A full configuration MAY use 999 levels.

### 20.2 Pinball loss

For error \(e=y-\hat q_\alpha\):

\[
L_\alpha(e)=\max(\alpha e,(\alpha-1)e).
\]

The training loss is the mean across:

```text
batch
query rows
quantiles
```

```python
def pinball_loss(
    pred_quantiles: torch.Tensor,
    target: torch.Tensor,
    levels: torch.Tensor,
) -> torch.Tensor:
    error = target[..., None] - pred_quantiles
    return torch.maximum(levels * error, (levels - 1.0) * error).mean()
```

### 20.3 Point prediction

For validation, the default point prediction is the mean of predicted quantiles:

```python
y_pred_standardized = pred_quantiles.mean(dim=-1)
y_pred = y_pred_standardized * y_scale[:, None] + y_loc[:, None]
```

### 20.4 Classification loss

Classification uses cross entropy over query rows.

---

## 21. Minimal training loop

### 21.1 CLI

```bash
python train.py \
  --config configs/chem_prior_minimal.yaml \
  --task regression \
  --steps 10000 \
  --batch-size 16 \
  --n-rows 256 \
  --n-features 64 \
  --quantiles 99 \
  --device cuda \
  --seed 0 \
  --output checkpoints/chem_prior_minimal.pt
```

### 21.2 Training sequence

Each step MUST:

1. sample a `PriorBatch`;
2. standardize regression targets using context rows only;
3. call `NanoTabICLv2(x, y_train)`;
4. compute query loss;
5. backpropagate;
6. clip gradient norm;
7. update model and scheduler;
8. emit structured metrics.

### 21.3 Default small model

For smoke tests and local development:

```python
NanoTabICLv2(
    max_classes=0,
    out_dim=99,
    embed_dim=64,
    col_num_blocks=2,
    row_num_blocks=2,
    icl_num_blocks=4,
    col_nhead=4,
    row_nhead=4,
    icl_nhead=4,
    feature_group_size=3,
    n_cls_cols=4,
    n_cls_rows=32,
)
```

The original-size model remains configurable.

### 21.4 Optimizer

Version `0.1.0` MAY use AdamW for simplicity:

```yaml
optimizer: adamw
learning_rate: 0.0003
weight_decay: 0.01
betas: [0.9, 0.95]
grad_clip_norm: 1.0
scheduler: cosine
warmup_steps: 500
```

Muon or larger-scale curricula are out of scope for the minimal prior implementation.

### 21.5 Checkpoint

A checkpoint MUST contain:

```python
{
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "scheduler_state_dict": ...,
    "step": int,
    "model_config": dict,
    "prior_config": dict,
    "spec_version": "0.1.0",
    "descriptor_basis_version": "0.1.0",
    "root_seed": int,
    "quantile_levels": list[float] | None,
}
```

---

## 22. Configuration file

`configs/chem_prior_minimal.yaml` MUST contain at least:

```yaml
spec_version: "0.1.0"

task: regression

rows:
  min: 128
  max: 512
  train_fraction: [0.30, 0.90]

molecules:
  library_size: [8, 32]
  components_per_row: [1, 4]
  atoms: [2, 24]
  atom_type_count: 16
  bond_type_count: 4

descriptors:
  basis_version: "0.1.0"
  known_dim: 16
  anonymous_dim: 48
  visible_features: [16, 96]
  hidden_fraction: [0.20, 0.70]

interactions:
  max_order: 3
  pair_probability: 0.85
  triple_probability: 0.25
  rank: [1, 8]

scm:
  latent_nodes: [1, 6]
  regime_change_probability: 0.15
  hidden_confounder_probability: 0.10

property_family:
  additive_excess: 0.45
  log_additive_excess: 0.20
  bounded_response: 0.10
  generic_scm: 0.25

measurement:
  noise_std: [0.00, 0.20]
  heteroscedastic_probability: 0.30
  batch_effect_probability: 0.10

filter:
  predictive_signal: true
  min_oob_r2: 0.02
  max_attempts: 32
```

---

## 23. Predictive-signal filtering

The upstream NanoTabICL prior filters datasets using an out-of-bag ExtraTrees predictor. The molecular prior SHOULD retain a simpler deterministic filter.

For regression:

1. concatenate observed columns;
2. fit `ExtraTreesRegressor` using fixed hyperparameters and `random_state=1`;
3. compute out-of-bag \(R^2\);
4. accept if `oob_r2 >= min_oob_r2`.

Recommended estimator:

```python
ExtraTreesRegressor(
    n_estimators=32,
    bootstrap=True,
    oob_score=True,
    n_jobs=1,
    random_state=1,
    max_depth=8,
)
```

For classification, use `ExtraTreesClassifier` and require balanced accuracy above chance by at least `0.05`.

Filtering MUST have a bounded attempt count. If no episode passes after `max_generation_attempts`, raise:

```python
class PriorGenerationError(RuntimeError):
    ...
```

The exception MUST include seed, config hash, and rejection reasons.

---

## 24. Validation CLI and report schema

### 24.1 CLI

```bash
python validate_prior.py \
  --config configs/chem_prior_minimal.yaml \
  --seed 17 \
  --episodes 32 \
  --output artifacts/validation_seed17.json
```

Exit codes:

```text
0: all required checks pass
1: at least one required check fails
2: invalid configuration or runtime error
```

### 24.2 Report schema

```json
{
  "spec_version": "0.1.0",
  "descriptor_basis_version": "0.1.0",
  "config_sha256": "...",
  "root_seed": 17,
  "episodes": 32,
  "environment": {
    "python": "3.11.x",
    "torch": "2.x",
    "numpy": "2.x",
    "device": "cpu"
  },
  "checks": {
    "shape_contract": {
      "required": true,
      "passed": true,
      "details": {}
    },
    "finite_values": {
      "required": true,
      "passed": true,
      "details": {}
    }
  },
  "summary": {
    "required_passed": 0,
    "required_failed": 0,
    "optional_passed": 0,
    "optional_failed": 0,
    "passed": true
  }
}
```

Every check MUST include:

```text
required
passed
details
```

---

## 25. Required validation checks

### V-001: Shape contract

For every episode:

```text
x.shape == [n_rows, n_features]
y.shape == [n_rows]
0 < n_train < n_rows
```

For a batch:

```text
batch.x.shape == [batch_size, n_rows, n_features]
batch.y_train.shape == [batch_size, n_train]
batch.y_query.shape == [batch_size, n_rows - n_train]
```

### V-002: Finite values

All returned training tensors MUST contain no NaN or infinity.

### V-003: Composition simplex

For every row:

```text
all fractions >= 0
sum fractions == 1 within 1e-6
padding fraction == 0
active component IDs are valid
```

### V-004: Connected molecular graphs

Every generated molecular graph MUST be connected and satisfy all schema invariants.

### V-005: Atom permutation invariance

Randomly permuting atom indices and remapping edges MUST leave all 64 base descriptors equal within `1e-6`.

### V-006: Component permutation invariance

Permuting active component slots and matching compositions MUST leave:

```text
y_clean
unary contribution
pair contribution
triple contribution
latent mechanism values
```

unchanged within `1e-6`.

Observed slot columns MAY change order because they represent slots.

### V-007: Zero-composition interaction law

Setting a component fraction to zero and renormalizing the remaining components MUST make every interaction containing that component vanish.

### V-008: Pure-component limit

For a pure row:

```text
pair contribution == 0
triple contribution == 0
```

within `1e-7`.

### V-009: Duplicate merge invariance

If the same molecule appears in two slots, canonicalization MUST merge it before evaluation. Splitting a mole fraction across duplicate slots MUST leave `y_clean` unchanged within `1e-6`.

### V-010: Deterministic replay

Two CPU generations with the same seed and config MUST be bitwise identical for:

```text
x
y
world JSON
molecular graphs
compositions
conditions
contributions
counterfactuals
```

### V-011: Serialization round trip

World JSON plus latent-row arrays MUST reproduce `y_clean` within `1e-6`.

### V-012: Contribution reconstruction

The debug expression tree and contribution tensors MUST reconstruct `y_clean` and `y_observed` within `1e-6`.

### V-013: Structural intervention replay

A structural intervention evaluated twice with the same paired exogenous noise MUST produce identical counterfactual targets.

### V-014: Observation corruption is noncausal

Calling `corrupt_observed_feature` MUST change the requested `x` column and MUST NOT alter:

```text
y_clean
y_observed
world_id
latent rows
```

### V-015: Requested class cardinality

For classification compatibility output:

```text
0 <= y < n_classes
all requested classes appear at least twice
```

### V-016: NanoTabICL forward compatibility

The following MUST run without error:

```python
batch = sample_batch(
    batch_size=2,
    config=config,
    seed=17,
    n_rows=64,
    n_train=48,
    n_features=32,
)

model = NanoTabICLv2(
    max_classes=0,
    out_dim=19,
    embed_dim=32,
    col_num_blocks=1,
    row_num_blocks=1,
    icl_num_blocks=2,
    col_nhead=4,
    row_nhead=4,
    icl_nhead=4,
    n_cls_rows=16,
)

pred = model(batch.x, batch.y_train)
assert pred.shape == (2, 16, 19)
assert torch.isfinite(pred).all()
```

### V-017: Bounded generation

No episode-generation request may loop indefinitely. Generation MUST either return or raise `PriorGenerationError` within `max_generation_attempts`.

### V-018: Known-submodel fixtures

The test suite MUST contain deterministic fixture worlds demonstrating:

1. exact linear mixing;
2. pair-only excess mixing;
3. triple-only synergy;
4. positive log-additive response;
5. bounded sigmoid response.

For each fixture, expected targets MUST be computable by a direct reference formula and match within `1e-6`.

---

## 26. Statistical validation checks

These checks are required in the validation CLI but SHOULD use tolerances broad enough to avoid flaky tests.

### S-001: Property-family frequencies

Across at least 1,000 sampled worlds, empirical family frequencies MUST be within `±0.05` absolute error of configured probabilities.

### S-002: Interaction-order frequencies

Across at least 1,000 worlds:

```text
fraction with pair term ~= pair_probability ± 0.05
fraction with triple term ~= triple_probability ± 0.05
```

### S-003: Feature hiding

For every episode, the hidden latent-channel fraction MUST lie in the configured interval.

### S-004: Predictive signal

For filtered regression episodes, at least 95% of accepted episodes MUST satisfy the configured OOB threshold when independently recomputed.

### S-005: Target diversity

Across 100 episodes, fewer than 5% MAY have clean target standard deviation below `1e-3`.

### S-006: Counterfactual nontriviality

Across 100 debug episodes, at least 70% MUST have at least one default intervention with:

```python
mean(abs(y_counterfactual - y_factual)) > 1e-3 * std(y_factual)
```

---

## 27. Training smoke-test acceptance criteria

The implementation MUST include a CPU- or single-GPU-compatible smoke test.

### 27.1 Tiny-batch overfit

Using a fixed batch of four episodes, a small model trained for at most 500 optimizer steps MUST reduce training loss by at least 30% relative to the median of the first ten steps.

This test verifies wiring, not generalization.

### 27.2 Online-prior learning smoke test

Using newly sampled episodes each step:

- train for 2,000 steps;
- evaluate on 128 fixed unseen synthetic episodes;
- compare against a context-mean baseline.

Regression acceptance:

```text
model normalized RMSE <= 0.95 * context-mean normalized RMSE
```

Classification acceptance:

```text
model balanced accuracy >= chance + 0.05
```

These thresholds are intentionally modest for a minimal implementation.

---

## 28. Example debug artifact

A debug episode’s world JSON SHOULD resemble:

```json
{
  "spec_version": "0.1.0",
  "descriptor_basis_version": "0.1.0",
  "world_id": "3ec1c21c07e9ef55",
  "seeds": {
    "root": 17,
    "molecule_library": 101238491,
    "mechanisms": 98031142,
    "measurement": 7211981
  },
  "library": {
    "size": 12,
    "molecule_ids": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
  },
  "property": {
    "family": "additive_excess",
    "link": "identity",
    "interaction_order": 3
  },
  "scm": {
    "nodes": [
      {
        "id": "pair_0",
        "type": "PAIR_INTERACTION",
        "parents": ["structure", "condition_T"],
        "program": {
          "family": "quadratic_low_rank",
          "rank": 3
        }
      },
      {
        "id": "latent_0",
        "type": "LATENT_STATE",
        "parents": ["pair_0", "condition_P"],
        "program": {
          "family": "saturating"
        }
      },
      {
        "id": "target_clean",
        "type": "IDEAL_TARGET",
        "parents": ["pure", "pair_0", "triple_0", "latent_0"]
      }
    ]
  },
  "features": {
    "count": 64,
    "hidden_descriptor_fraction": 0.4375,
    "permuted": true
  }
}
```

The exact numeric values are not normative. The presence and type of fields are normative.

---

## 29. Reference algorithms

### 29.1 Sampling one episode

```python
def sample_episode(config, *, seed, n_rows=None, n_train=None, n_features=None, debug=None):
    config.validate()
    rng = SeedTree(seed, spec_version=config.spec_version)

    world = sample_world(config, seed=rng["world"])

    n_rows = n_rows or sample_int(config.n_rows_min, config.n_rows_max, rng["n_rows"])
    n_train = n_train or sample_train_size(n_rows, config, rng["n_train"])
    n_features = n_features or sample_int(config.n_features_min, config.n_features_max, rng["n_features"])

    for attempt in range(config.max_generation_attempts):
        rows = world.sample_rows(n_rows, seed=rng.child("rows", attempt))
        evaluation = world.evaluate(
            rows,
            measurement_seed=rng.child("measurement", attempt),
            return_contributions=True,
        )

        x, feature_specs = observe_rows(
            world,
            rows,
            evaluation,
            n_features=n_features,
            seed=rng.child("schema", attempt),
        )

        y = evaluation.y_observed

        if valid_basic_tensors(x, y) and (
            not config.filter_predictive_signal
            or passes_signal_filter(x, y, config)
        ):
            break
    else:
        raise PriorGenerationError(...)

    debug_obj = None
    if debug if debug is not None else config.debug:
        debug_obj = build_debug_artifact(...)

    return PriorEpisode(
        x=x,
        y=y,
        n_train=n_train,
        task=config.task,
        feature_metadata=feature_specs,
        target_metadata=evaluation.target_spec,
        seed=seed,
        world_id=world.world_id,
        debug=debug_obj,
    )
```

### 29.2 Symmetric interaction aggregation

```python
def aggregate_pair_interactions(descriptors, fractions, kernel):
    # descriptors: [m, d], fractions: [m]
    total = 0.0
    for c in range(len(fractions)):
        for d in range(c + 1, len(fractions)):
            if fractions[c] == 0 or fractions[d] == 0:
                continue
            total = total + fractions[c] * fractions[d] * kernel(
                symmetric_pair_features(descriptors[c], descriptors[d])
            )
    return total
```

A vectorized implementation is preferred, but it MUST agree with this reference.

### 29.3 Canonicalizing a row

```python
def canonicalize_components(component_ids, fractions):
    merged: dict[int, float] = {}
    for component_id, fraction in zip(component_ids, fractions):
        if component_id < 0 or fraction <= 0:
            continue
        merged[component_id] = merged.get(component_id, 0.0) + float(fraction)

    ids = sorted(merged)
    x = torch.tensor([merged[i] for i in ids], dtype=torch.float64)
    x = x / x.sum()
    return ids, x
```

Target evaluation MUST occur after canonicalization.

---

## 30. Implementation phases

### Phase 1 — Core deterministic prior

Deliver:

- config and validation;
- seed tree;
- abstract molecule generator;
- 64-dimensional descriptor basis;
- component library and composition sampler;
- unary and pair interactions;
- additive-excess regression target;
- observed feature schema;
- compatibility wrapper;
- required checks V-001 through V-012 and V-016 through V-018.

Phase 1 MAY omit:

- triple terms;
- hidden confounding;
- classification;
- regime changes;
- structural interventions.

### Phase 2 — Complete version `0.1.0`

Add:

- triple interactions;
- all property families;
- latent SCM;
- heteroscedastic noise and batch effects;
- structural interventions;
- classification;
- all validation checks;
- training loop and smoke tests.

### Phase 3 — Optional chemistry adapters

Not required for `0.1.0`:

- RDKit molecule provider;
- user-supplied descriptor matrices;
- real-unit metadata;
- graph-token architecture extensions;
- actual molecular-mixture benchmark adapters.

---

## 31. Acceptance criteria

The implementation is complete when all of the following are true:

1. `pytest -q` passes all required deterministic tests on CPU.
2. `python validate_prior.py ...` exits with code `0` for the minimal config.
3. the validation JSON conforms to the report schema.
4. the generator returns finite NanoTabICL-compatible tensors.
5. component and atom permutation invariance tests pass.
6. pure-component and zero-composition limits pass.
7. serialization and deterministic replay pass.
8. contribution reconstruction passes.
9. structural intervention and observation-corruption semantics pass.
10. known-submodel fixture targets match direct formulas.
11. the tiny-batch training loss falls by at least 30%.
12. the online-prior smoke test beats its specified baseline.
13. no required dependency beyond PyTorch, NumPy, PyYAML, and scikit-learn is introduced.
14. upstream `model.py` remains usable without API changes.

---

## 32. Design rationale

### 32.1 Why an abstract graph rather than only a random vector?

A random vector would be simpler, but it would not guarantee that descriptor relationships arise from a common structure. An abstract graph gives multiple descriptors shared ancestry and makes structural interventions well-defined.

### 32.2 Why retain anonymous channels?

A prior containing only named contemporary descriptors would teach the model that the current descriptor vocabulary is complete. Anonymous invariant channels create additional structurally meaningful distinctions while keeping the implementation finite.

### 32.3 Why expose both slot and symmetric features?

Symmetric summaries make mixture invariance easy, while slot features preserve component-level information needed to infer nonadditive effects. Random slot permutations prevent a fixed slot from acquiring a permanent semantic meaning.

### 32.4 Why use factor kernels inside a causal graph?

Equilibrium interactions are reciprocal. Forcing them into arbitrary directed component-to-component arrows would create false causal orientation. Symmetric factors describe the interaction; directed arrows describe how structure and conditions determine factors and how factors determine observations.

### 32.5 Why debug metadata?

Predictive accuracy alone cannot validate that the prior has the intended causal semantics. The sampled world, exact contributions, and counterfactuals make the generator auditable and allow mechanistic evaluations beyond test loss.

### 32.6 Why keep the model unchanged?

The first scientific question is whether a better structured prior can induce useful molecular in-context learning in an existing tabular architecture. Changing the architecture simultaneously would make the result difficult to attribute.

---

## 33. Open questions intentionally deferred

The following should be tracked as future design questions rather than silently decided in `0.1.0`:

1. Should descriptor metadata be embedded into NanoTabICL?
2. Should component permutation invariance be architectural rather than learned by augmentation?
3. Should property families be derived from a sampled free-energy functional?
4. Should the prior include trajectory and memory-kernel mechanisms for transport properties?
5. How should actual molecular graphs and conformers be mixed with abstract graphs during pretraining?
6. Should one world generate several correlated target properties for multitask ICL?
7. How should units be represented once real descriptors are introduced?
8. How should causal posterior quality be scored when several worlds are observationally equivalent?
9. What prior mixture best balances familiar chemistry, open-ended mechanisms, and adversarial stress worlds?
10. At what point does a structure-aware architecture outperform a pure tabular model trained on structural priors?

---

## 34. Upstream references

- NanoTabICL repository: <https://github.com/soda-inria/nanotabicl>
- NanoTabICL raw model implementation: <https://raw.githubusercontent.com/soda-inria/nanotabicl/main/model.py>
- NanoTabICL raw prior implementation: <https://raw.githubusercontent.com/soda-inria/nanotabicl/main/prior.py>
- TabICLv2 paper: <https://arxiv.org/abs/2602.11139>

---

## 35. Final normative statement

Version `0.1.0` is a prior over **dataset-generating structural worlds**, not a collection of random descriptor-to-target functions.

A conforming implementation MUST ensure that:

```text
molecular structure determines latent descriptors;
latent descriptors and conditions determine symmetric interactions;
interactions determine latent state and ideal properties;
measurement processes determine observed targets;
observed features are partial, randomized views of the latent world;
valid interventions recompute downstream variables;
feature corruption does not alter the physical world;
and every important step can be replayed and validated.
```
