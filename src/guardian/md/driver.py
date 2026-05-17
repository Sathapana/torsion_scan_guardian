from dataclasses import dataclass
from typing import Callable
from ase import Atoms, units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution


@dataclass
class MDState:
    step: int
    energy: float
    temperature: float


def make_langevin(
    atoms: Atoms,
    temperature_K: float,
    timestep_fs: float,
    friction: float,
) -> Langevin:
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)
    return Langevin(
        atoms,
        timestep=timestep_fs * units.fs,
        temperature_K=temperature_K,
        friction=friction,
    )


def run_with_callback(
    dyn: Langevin,
    total_steps: int,
    on_step: Callable[[MDState], bool],
    log_every: int = 100,
) -> int:
    """Run integrator step-by-step; on_step returns True to halt (e.g., guardian trigger)."""
    for step in range(total_steps):
        dyn.run(1)
        if step % log_every == 0 or step == total_steps - 1:
            atoms = dyn.atoms
            ke = atoms.get_kinetic_energy()
            t = ke / (1.5 * units.kB * len(atoms))
            state = MDState(step=step, energy=float(atoms.get_potential_energy()), temperature=float(t))
            if on_step(state):
                return step
    return total_steps
