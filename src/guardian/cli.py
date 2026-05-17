import argparse
from pathlib import Path

from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(prog="guardian")
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    parser.add_argument("--smiles", type=str, default=None,
                        help="Override molecule.smiles in the config")
    parser.add_argument("--steps", type=int, default=None,
                        help="Override md.total_steps in the config")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Override md.temperature_K in the config")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override uncertainty.threshold in the config")
    parser.add_argument("--relax", action="store_true",
                        help="BFGS-relax the geometry with MACE-OFF before anything else")
    parser.add_argument("--calibrate", action="store_true",
                        help="After relaxation, calibrate the uncertainty threshold and exit")
    parser.add_argument("--calibrate-samples", type=int, default=50)
    parser.add_argument("--calibrate-sigma", type=float, default=0.04,
                        help="Thermal-jitter amplitude in Angstrom (~300K stiff-bond RMS)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build atoms + ensemble and predict once, then exit")
    parser.add_argument("--max-triggers", type=int, default=1,
                        help="Stop after this many Guardian events (default 1; Phase-4 fine-tuning lets it run further)")
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Override run output directory")
    parser.add_argument("--no-relax", action="store_true",
                        help="Skip the pre-MD geometry relaxation (not recommended)")
    parser.add_argument("--checkpoints", type=Path, nargs="+", default=None,
                        help="Use a SeedFinetuneEnsemble loaded from these MACE checkpoints "
                             "(Phase 2 mode). If omitted, falls back to input-perturbation.")
    parser.add_argument("--online-finetune", action="store_true",
                        help="Phase 4: after each trigger, re-fine-tune all members on "
                             "(seed + acquired) data and continue MD with the updated ensemble.")
    parser.add_argument("--seed-data-file", type=Path, default=None,
                        help="Seed extxyz used as the base training set when --online-finetune is on.")
    parser.add_argument("--finetune-epochs", type=int, default=2)
    parser.add_argument("--finetune-lr", type=float, default=1e-4)
    parser.add_argument("--cooldown-steps", type=int, default=200,
                        help="Suppress triggers for this many steps after a trigger fires.")
    parser.add_argument("--cloud-size", type=int, default=0,
                        help="Number of perturbed copies of the trigger geometry to label "
                             "with GFN-FF (0 = label only the trigger geometry).")
    parser.add_argument("--cloud-jitter", type=float, default=0.05,
                        help="Position-perturbation sigma (A) for acquisition cloud points.")
    parser.add_argument("--ft-regression-tol", type=float, default=0.10,
                        help="Reject a fine-tune update if val RMSE_F worsens by more than this fraction.")
    parser.add_argument("--max-parallel-finetunes", type=int, default=3,
                        help="Cap concurrent mace_run_train subprocesses per cycle (default 3). "
                             "Lower for smaller GPUs; can raise to N-members for big GPUs.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.smiles is not None:
        cfg.molecule.smiles = args.smiles
    if args.steps is not None:
        cfg.md.total_steps = args.steps
    if args.temperature is not None:
        cfg.md.temperature_K = args.temperature
    if args.threshold is not None:
        cfg.uncertainty.threshold = args.threshold

    print(f"[guardian] molecule={cfg.molecule.smiles}  steps={cfg.md.total_steps}  "
          f"backbone={cfg.model.backbone}  mode={cfg.model.ensemble_mode}  "
          f"probes={cfg.model.n_probes}  oracle={cfg.oracle.method}")

    from .io.structures import smiles_to_atoms
    from .models.ensemble import MACEOffEnsemble, SeedFinetuneEnsemble, load_mace_off

    atoms = smiles_to_atoms(cfg.molecule.smiles)
    print(f"[guardian] built {len(atoms)} atoms from SMILES")

    if args.checkpoints:
        print(f"[guardian] mode=seed-fine-tune  members={len(args.checkpoints)}")
        ensemble = SeedFinetuneEnsemble.from_checkpoints(
            [str(p) for p in args.checkpoints],
            device=cfg.model.device, dtype=cfg.model.dtype,
        )
        calc = ensemble.calcs[0]
    else:
        size = cfg.model.backbone.removeprefix("mace-off-")
        calc = load_mace_off(size=size, device=cfg.model.device, dtype=cfg.model.dtype)
        ensemble = MACEOffEnsemble(
            calc=calc,
            n_probes=cfg.model.n_probes,
            position_noise_A=cfg.model.position_noise_A,
        )

    do_relax = (args.relax or args.calibrate) or (not args.no_relax and not args.dry_run)
    if do_relax:
        from .calibration import relax_geometry
        report = relax_geometry(atoms, calc, fmax=0.05, max_steps=200)
        print(f"[guardian] relax: converged={report.converged}  steps={report.n_steps}  "
              f"fmax={report.fmax_final:.4f} eV/A  E={report.energy_final:.4f} eV")

    pred = ensemble.predict(atoms)
    print(f"[guardian] E={pred.energy:.4f} eV  "
          f"|F|max={(pred.forces**2).sum(axis=1).max()**0.5:.3f} eV/A  "
          f"std_max={pred.forces_std_per_atom.max():.4f} eV/A")

    if args.calibrate:
        from .calibration import calibrate_threshold
        cal = calibrate_threshold(
            ensemble, atoms,
            n_samples=args.calibrate_samples,
            sigma_A=args.calibrate_sigma,
        )
        print(f"[guardian] calibration (n={cal.n_samples}, sigma={cal.sigma_A} A): "
              f"p50={cal.max_std_p50:.4f}  p95={cal.max_std_p95:.4f}  "
              f"p99={cal.max_std_p99:.4f} eV/A")
        print(f"[guardian] suggested uncertainty.threshold = {cal.suggested_threshold:.4f} eV/A "
              f"(currently {cfg.uncertainty.threshold})")
        return

    if args.dry_run:
        return

    from .pipeline.controller import GuardianController
    ctrl = GuardianController(
        cfg, atoms, ensemble,
        max_triggers=args.max_triggers, run_dir=args.run_dir,
        online_finetune=args.online_finetune,
        seed_data_file=args.seed_data_file,
        member_checkpoints=[str(p) for p in (args.checkpoints or [])] or None,
        finetune_epochs=args.finetune_epochs,
        finetune_lr=args.finetune_lr,
        cooldown_steps=args.cooldown_steps,
        cloud_size=args.cloud_size,
        cloud_jitter_A=args.cloud_jitter,
        ft_regression_tol=args.ft_regression_tol,
        max_parallel_finetunes=args.max_parallel_finetunes,
    )
    print(f"[guardian] run_dir={ctrl.run_dir}  threshold={cfg.uncertainty.threshold}  "
          f"max_triggers={args.max_triggers}  online_ft={args.online_finetune}")
    if args.online_finetune and not args.seed_data_file:
        print("[guardian] WARNING: --online-finetune without --seed-data-file; "
              "fine-tunes will only see acquired points.")
    ctrl.run()
    print(f"[guardian] done. steps={ctrl._global_step}  triggers={len(ctrl.cycles)}")
    for c in ctrl.cycles:
        print(f"  cycle {c.cycle}: step={c.trigger_step}  std={c.trigger_std:.3f}  "
              f"atom={c.trigger_atom}  oracle_ok={c.oracle_ok}  "
              f"E_oracle={c.oracle_energy_eV}")


if __name__ == "__main__":
    main()
