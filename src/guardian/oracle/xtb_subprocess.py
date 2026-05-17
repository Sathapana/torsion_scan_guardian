"""ASE Calculator that invokes the standalone `xtb` CLI via subprocess.

Used because `tblite-python` and `xtb-python` both fail with a Windows delay-load
DLL bug on this environment (Fortran error-handling path crashes inside
`singlepoint` → `handle_context_error`). The `xtb.exe` binary itself is fine, so
we drive it via subprocess and parse Turbomole-format energy/gradient outputs.

Supports `--gfnff` (default) and `--gfn 1` / `--gfn 2` via the `method` kwarg.
"""
from __future__ import annotations
from pathlib import Path
import shutil
import subprocess
import tempfile
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.units import Hartree, Bohr


def _resolve_xtb_exe(explicit: str | None) -> str:
    if explicit:
        return explicit
    found = shutil.which("xtb") or shutil.which("xtb.exe")
    if found:
        return found
    # Conda Windows default location
    import os, sys
    candidate = Path(sys.prefix) / "Library" / "bin" / "xtb.exe"
    if candidate.exists():
        return str(candidate)
    raise FileNotFoundError("xtb executable not found; pass `xtb_path` explicitly")


def _parse_turbomole_energy(path: Path) -> float:
    """Return total energy in Hartree from a Turbomole `energy` file."""
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    # Second non-blank line: "  1  E_total  E_kin  E_pot"
    parts = lines[1].split()
    return float(parts[1])


def _parse_turbomole_gradient(path: Path, n_atoms: int) -> np.ndarray:
    """Return gradient in Hartree/Bohr, shape (n_atoms, 3)."""
    lines = [ln.rstrip() for ln in path.read_text().splitlines() if ln.strip()]
    # Layout: $grad, cycle header, N coord lines, N gradient lines, $end
    grad_start = 2 + n_atoms
    grad = np.empty((n_atoms, 3))
    for i in range(n_atoms):
        toks = lines[grad_start + i].replace("D", "E").split()
        grad[i] = [float(t) for t in toks[:3]]
    return grad


class XTBCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, method: str = "gfnff", xtb_path: str | None = None,
                 extra_args: tuple[str, ...] = (), **kwargs):
        super().__init__(**kwargs)
        self.method = method
        self.xtb_path = _resolve_xtb_exe(xtb_path)
        self.extra_args = tuple(extra_args)

    def _method_flags(self) -> list[str]:
        m = self.method.lower()
        if m in ("gfnff", "gfn-ff", "gff"):
            return ["--gfnff"]
        if m in ("gfn2", "gfn2-xtb", "gfn-2"):
            return ["--gfn", "2"]
        if m in ("gfn1", "gfn1-xtb", "gfn-1"):
            return ["--gfn", "1"]
        raise ValueError(f"Unsupported xtb method: {self.method}")

    def calculate(self, atoms: Atoms = None, properties=("energy", "forces"),
                  system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        with tempfile.TemporaryDirectory(prefix="xtbwork_") as tmpdir:
            tmp = Path(tmpdir)
            xyz_path = tmp / "in.xyz"
            self.atoms.write(str(xyz_path), format="xyz")
            cmd = [self.xtb_path, "in.xyz", *self._method_flags(),
                   "--grad", *self.extra_args]
            res = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True)
            if res.returncode != 0:
                raise RuntimeError(
                    f"xtb failed (exit {res.returncode})\nSTDERR:\n{res.stderr[-2000:]}"
                )
            energy_h = _parse_turbomole_energy(tmp / "energy")
            grad_h_bohr = _parse_turbomole_gradient(tmp / "gradient", len(self.atoms))
        # Energy: Hartree → eV. Forces: -gradient(Eh/Bohr) → eV/Å.
        e_ev = energy_h * Hartree
        f_ev_per_A = -grad_h_bohr * Hartree / Bohr
        self.results["energy"] = e_ev
        self.results["forces"] = f_ev_per_A
