from dataclasses import dataclass, asdict
from enum import Enum, auto
from pathlib import Path
import json
import time
import numpy as np
from ase import Atoms
from ase.io.trajectory import Trajectory

from ..config import GuardianCfg
from ..md.driver import make_langevin, run_with_callback, MDState
from ..models.ensemble import EnsembleCalculator, MACEOffEnsemble
from ..uncertainty.monitor import ForceVarianceMonitor, TriggerEvent
from ..oracle.gfnff import label_with_gfnff, label_cloud_with_gfnff
from ..training.replay import ReplayBuffer, DataPoint


class State(Enum):
    RUN = auto()
    PAUSE = auto()
    LABEL = auto()
    RETRAIN = auto()
    RESUME = auto()
    DONE = auto()


@dataclass
class CycleLog:
    cycle: int
    trigger_step: int
    trigger_std: float
    trigger_atom: int
    oracle_ok: bool
    oracle_energy_eV: float | None
    labels_acquired: int


class GuardianController:
    """Runs the AL loop. Without Phase-4 fine-tuning, stops cleanly after `max_triggers` events."""

    def __init__(self, cfg: GuardianCfg, atoms: Atoms, ensemble: MACEOffEnsemble,
                 max_triggers: int = 1, run_dir: str | Path | None = None,
                 online_finetune: bool = False,
                 seed_data_file: str | Path | None = None,
                 member_checkpoints: list[str] | None = None,
                 finetune_epochs: int = 2,
                 finetune_lr: float = 1e-4,
                 cooldown_steps: int = 200,
                 cloud_size: int = 0,
                 cloud_jitter_A: float = 0.05,
                 ft_regression_tol: float = 0.10):
        self.cfg = cfg
        self.atoms = atoms
        self.ensemble = ensemble
        self.calc = EnsembleCalculator(ensemble)
        self.atoms.calc = self.calc
        self.monitor = ForceVarianceMonitor(cfg.uncertainty.threshold, cfg.uncertainty.warmup_steps)
        self.buffer = ReplayBuffer()
        self.cycles: list[CycleLog] = []
        self.state = State.RUN
        self.max_triggers = max_triggers
        self.rng = np.random.default_rng(0)

        # Phase-4 online fine-tune state
        self.online_finetune = online_finetune
        self.seed_data_file = Path(seed_data_file) if seed_data_file else None
        self.member_checkpoints = list(member_checkpoints) if member_checkpoints else []
        self.finetune_epochs = finetune_epochs
        self.finetune_lr = finetune_lr
        self.cooldown_steps = cooldown_steps
        self.cloud_size = cloud_size
        self.cloud_jitter_A = cloud_jitter_A
        self.ft_regression_tol = ft_regression_tol
        self._last_trigger_step = -1

        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.run_dir = Path(run_dir) if run_dir else Path(cfg.io.run_dir) / stamp
        self.run_dir.mkdir(parents=True, exist_ok=True)
        Path(cfg.io.oracle_cache).mkdir(parents=True, exist_ok=True)

        self._traj: Trajectory | None = None
        self._csv = None
        self._pending_trigger: TriggerEvent | None = None
        self._global_step = 0

    def _on_step(self, state: MDState) -> bool:
        std = self.calc.last_force_std
        if std is None:
            return False
        max_std = float(std.max())
        gs = self._global_step + state.step
        if self._csv is not None:
            self._csv.write(f"{gs},{state.energy:.6f},{state.temperature:.3f},{max_std:.6f}\n")
            self._csv.flush()
        if self._traj is not None:
            self._traj.write(self.atoms)
        # Cooldown: suppress triggers for `cooldown_steps` after the previous one
        # so the molecule has a chance to move away from the trigger geometry.
        if gs - self._last_trigger_step < self.cooldown_steps:
            return False
        event = self.monitor.check(gs, std)
        if event is not None:
            self._pending_trigger = event
            return True
        return False

    def run(self) -> None:
        traj_path = self.run_dir / "traj.traj"
        csv_path = self.run_dir / "md.csv"
        self._traj = Trajectory(str(traj_path), "w", self.atoms)
        self._csv = open(csv_path, "w")
        self._csv.write("step,energy_eV,temperature_K,max_force_std_eVperA\n")
        t0 = time.time()
        steps_remaining = self.cfg.md.total_steps
        try:
            while steps_remaining > 0 and self.state is not State.DONE:
                dyn = make_langevin(
                    self.atoms,
                    self.cfg.md.temperature_K,
                    self.cfg.md.timestep_fs,
                    self.cfg.md.friction,
                )
                self._pending_trigger = None
                consumed = run_with_callback(
                    dyn, steps_remaining, self._on_step, self.cfg.md.log_every,
                )
                self._global_step += consumed
                steps_remaining -= consumed
                if self._pending_trigger is None:
                    self.state = State.DONE
                    break
                event = self._pending_trigger
                self._handle_trigger(event)
                if len(self.cycles) >= self.max_triggers:
                    self.state = State.DONE
                    break
        finally:
            self._traj.close()
            self._csv.close()
            self._write_summary(elapsed_s=time.time() - t0)

    def _handle_trigger(self, event: TriggerEvent) -> None:
        cycle_idx = len(self.cycles)
        flagged_xyz = self.run_dir / f"trigger_{cycle_idx:03d}.xyz"
        self.atoms.write(str(flagged_xyz))
        oracle_ok, energy, labels = False, None, 0
        try:
            label = label_with_gfnff(self.atoms)
            self.buffer.add(DataPoint(
                positions=label.positions, symbols=label.symbols,
                energy=label.energy, forces=label.forces,
                source="acquired", cycle=cycle_idx,
            ))
            oracle_ok, energy, labels = True, label.energy, 1
            if self.cloud_size > 0:
                cloud_labels = label_cloud_with_gfnff(
                    self.atoms, self.cloud_size, self.cloud_jitter_A, self.rng,
                )
                for cl in cloud_labels:
                    self.buffer.add(DataPoint(
                        positions=cl.positions, symbols=cl.symbols,
                        energy=cl.energy, forces=cl.forces,
                        source="acquired-cloud", cycle=cycle_idx,
                    ))
                labels += len(cloud_labels)
        except Exception as e:
            (self.run_dir / f"trigger_{cycle_idx:03d}.err").write_text(repr(e))
        self.cycles.append(CycleLog(
            cycle=cycle_idx,
            trigger_step=event.step,
            trigger_std=event.max_atom_std,
            trigger_atom=event.atom_index,
            oracle_ok=oracle_ok,
            oracle_energy_eV=energy,
            labels_acquired=labels,
        ))
        self.buffer.save(self.run_dir / "replay.pkl")
        self._last_trigger_step = event.step
        if self.online_finetune and oracle_ok:
            self._do_finetune_cycle(cycle_idx)

    def _do_finetune_cycle(self, cycle_idx: int) -> None:
        """Phase-4 hook: rebuild combined train file, fine-tune each member, reload ensemble."""
        from ..training.finetune import write_combined_train, online_finetune_member
        from ..models.ensemble import load_mace_checkpoint, SeedFinetuneEnsemble, _pick_device
        effective_device = _pick_device(self.cfg.model.device)
        cycle_dir = self.run_dir / f"cycle_{cycle_idx:03d}"
        cycle_dir.mkdir(parents=True, exist_ok=True)
        train_file = cycle_dir / "train.xyz"
        n = write_combined_train(self.seed_data_file, self.buffer, train_file)
        print(f"[guardian] cycle {cycle_idx}: combined train file has {n} frames "
              f"(seed + {len(self.buffer.points)} acquired)", flush=True)
        new_paths: list[str] = []
        reports = []
        for i, member_path in enumerate(self.member_checkpoints):
            out_dir = cycle_dir / f"member_seed{i}"
            rep = online_finetune_member(
                Path(member_path), train_file, out_dir, seed=i,
                epochs=self.finetune_epochs, lr=self.finetune_lr,
                device=effective_device, valid_fraction=0.10,
                regression_tol=self.ft_regression_tol,
            )
            rep.cycle = cycle_idx
            reports.append(rep)
            new_paths.append(rep.new_checkpoint_path)
            tag = "ACCEPT" if rep.accepted else "REVERT"
            print(f"[guardian]   member {i} [{tag}]: {rep.seconds:.1f}s  "
                  f"RMSE_F init={rep.initial_force_rmse_meV_per_A} -> "
                  f"final={rep.final_force_rmse_meV_per_A} meV/A  ({rep.reason})",
                  flush=True)
        # Reload ensemble in-place
        new_calcs = [load_mace_checkpoint(p, device=effective_device,
                                          dtype=self.cfg.model.dtype) for p in new_paths]
        if isinstance(self.ensemble, SeedFinetuneEnsemble):
            self.ensemble.calcs = new_calcs
        else:
            raise RuntimeError("Online fine-tune requires SeedFinetuneEnsemble.")
        self.member_checkpoints = new_paths
        # Reset ASE calculator cache so the next predict() uses the new weights.
        self.atoms.calc = self.calc
        self.calc.results.clear()

    def _write_summary(self, elapsed_s: float) -> None:
        summary = {
            "elapsed_s": round(elapsed_s, 2),
            "global_steps": self._global_step,
            "n_triggers": len(self.cycles),
            "threshold": self.cfg.uncertainty.threshold,
            "molecule": self.cfg.molecule.smiles,
            "cycles": [asdict(c) for c in self.cycles],
        }
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
