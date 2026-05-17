from pathlib import Path
from typing import Literal
import yaml
from pydantic import BaseModel


class MoleculeCfg(BaseModel):
    smiles: str
    charge: int = 0
    multiplicity: int = 1


class MDCfg(BaseModel):
    temperature_K: float = 300.0
    timestep_fs: float = 0.5
    friction: float = 0.01
    total_steps: int = 200_000
    log_every: int = 100
    checkpoint_every: int = 1000


class ModelCfg(BaseModel):
    backbone: Literal["mace-off-small", "mace-off-medium", "mace-off-large"] = "mace-off-small"
    ensemble_mode: Literal["input-perturbation", "seed-fine-tune"] = "input-perturbation"
    n_probes: int = 3
    position_noise_A: float = 0.005
    device: str = "cuda"
    dtype: str = "float32"


class UncertaintyCfg(BaseModel):
    metric: Literal["max_atom_force_std"] = "max_atom_force_std"
    threshold: float = 0.2
    warmup_steps: int = 500


class PerturbCloudCfg(BaseModel):
    enabled: bool = True
    n_samples: int = 5
    dihedral_jitter_deg: float = 15.0


class OracleCfg(BaseModel):
    method: Literal["gfn-ff"] = "gfn-ff"
    perturb_cloud: PerturbCloudCfg = PerturbCloudCfg()


class TrainingCfg(BaseModel):
    lr: float = 1e-4
    epochs_per_cycle: int = 5
    batch_size: int = 8
    grad_clip: float = 10.0
    ema_decay: float = 0.999
    val_force_mae_regression_tol: float = 0.1


class IOCfg(BaseModel):
    run_dir: str = "runs/"
    oracle_cache: str = "data/oracle_cache/"


class WandbCfg(BaseModel):
    enabled: bool = False
    project: str = "torsion-scan-guardian"


class GuardianCfg(BaseModel):
    molecule: MoleculeCfg
    md: MDCfg = MDCfg()
    model: ModelCfg = ModelCfg()
    uncertainty: UncertaintyCfg = UncertaintyCfg()
    oracle: OracleCfg = OracleCfg()
    training: TrainingCfg = TrainingCfg()
    io: IOCfg = IOCfg()
    wandb: WandbCfg = WandbCfg()


def load_config(path: str | Path) -> GuardianCfg:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return GuardianCfg(**raw)
