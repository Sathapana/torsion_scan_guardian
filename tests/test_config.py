from pathlib import Path
from guardian.config import load_config


def test_default_config_loads():
    cfg = load_config(Path(__file__).parents[1] / "config" / "default.yaml")
    assert cfg.molecule.smiles
    assert cfg.model.n_probes >= 2
    assert cfg.model.ensemble_mode == "input-perturbation"
    assert cfg.oracle.method == "gfn-ff"
