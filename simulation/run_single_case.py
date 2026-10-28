import argparse
import csv
import json
import math
import os
import random
import subprocess
import time
from datetime import datetime
from itertools import product
from pathlib import Path

try:
    import lammps as lammps_module
    from lammps import lammps
except ImportError:
    lammps_module = None
    lammps = None


def package_potential_path() -> Path:
    candidates = []

    if lammps_module is not None:
        package_root = Path(lammps_module.__file__).resolve().parent
        candidates.append(package_root / "share" / "lammps" / "potentials" / "Cu_u3.eam")

        # The official Windows installer keeps the Python module in
        # <install>/Python/lammps and the potential files in <install>/Potentials.
        if len(package_root.parents) >= 2:
            candidates.append(package_root.parents[1] / "Potentials" / "Cu_u3.eam")

    env_potentials = os.environ.get("LAMMPS_POTENTIALS")
    if env_potentials:
        candidates.insert(0, Path(env_potentials) / "Cu_u3.eam")

    lammps_exe = os.environ.get("LAMMPS_EXE")
    if lammps_exe:
        exe_path = Path(lammps_exe).resolve()
        candidates.append(exe_path.parent.parent / "Potentials" / "Cu_u3.eam")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    if candidates:
        return candidates[0]
    raise SystemExit(
        "Cu_u3.eam could not be located. Set LAMMPS_POTENTIALS to the directory "
        "containing the potential file."
    )


def read_external_run_results(stress_csv: Path, log_path: Path):
    with stress_csv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"LAMMPS produced no stress-strain rows: {stress_csv}")
    natoms = int(float(rows[-1]["natoms"]))

    version = "external executable"
    with log_path.open(encoding="utf-8", errors="replace") as f:
        first_line = f.readline().strip()
        if first_line.startswith("LAMMPS"):
            version = first_line
    return natoms, version


def periodic_distance(a, b, box_length):
    sq = 0.0
    for ai, bi in zip(a, b):
        delta = abs(ai - bi)
        delta = min(delta, box_length - delta)
        sq += delta * delta
    return math.sqrt(sq)


def periodic_image_centers(center, radius, box_length):
    """Return the sphere centers needed to represent one pore in a periodic box."""
    axis_shifts = []
    for coordinate in center:
        shifts = [0.0]
        if coordinate < radius:
            shifts.append(box_length)
        if coordinate > box_length - radius:
            shifts.append(-box_length)
        axis_shifts.append(shifts)

    return [
        tuple(coordinate + shift for coordinate, shift in zip(center, shifts))
        for shifts in product(*axis_shifts)
    ]


def generate_pore_centers(args, box_length):
    rng = random.Random(args.pore_seed if args.pore_seed is not None else args.seed)
    pore_radius = args.pore_radius_a
    sphere_volume = 4.0 * math.pi * pore_radius**3 / 3.0
    target_pore_count = args.pore_count
    if target_pore_count is None:
        target_pore_count = max(1, math.ceil(args.porosity * box_length**3 / sphere_volume))

    min_center_distance = 2.0 * pore_radius + args.pore_gap_a
    if pore_radius <= 0.0:
        raise ValueError("Pore radius must be positive.")
    if target_pore_count < 1:
        raise ValueError("Pore count must be at least one.")
    if min_center_distance > box_length:
        raise ValueError(
            "Pore diameter plus pore gap exceeds the periodic box length."
        )

    centers = []
    attempts = 0
    max_attempts = max(args.max_pore_attempts, target_pore_count * 2000)
    while len(centers) < target_pore_count and attempts < max_attempts:
        attempts += 1
        # Sample the entire primary cell. A pore may cross any face; overlap is
        # checked with the minimum-image convention below.
        center = tuple(rng.random() * box_length for _ in range(3))
        if all(periodic_distance(center, old, box_length) >= min_center_distance for old in centers):
            centers.append(center)

    if len(centers) < target_pore_count:
        raise ValueError(
            f"Only placed {len(centers)} of {target_pore_count} pores. "
            "Try reducing --pore-radius-a or --pore-gap-a."
        )
    return centers


def pore_setup(args, case_dir: Path, box_length):
    if args.pore_mode == "single":
        target_void_volume = args.porosity * box_length**3
        void_radius = (3.0 * target_void_volume / (4.0 * math.pi)) ** (1.0 / 3.0)
        centers = [(box_length / 2.0, box_length / 2.0, box_length / 2.0)]
        pore_radius = void_radius
    else:
        centers = generate_pore_centers(args, box_length)
        pore_radius = args.pore_radius_a

    pore_centers_path = case_dir / "pore_centers.csv"
    with pore_centers_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pore_id", "x_angstrom", "y_angstrom", "z_angstrom", "radius_angstrom"])
        for idx, center in enumerate(centers, start=1):
            writer.writerow([idx, center[0], center[1], center[2], pore_radius])

    lines = []
    image_region_count = 0
    for idx, center in enumerate(centers, start=1):
        for image_idx, image_center in enumerate(
            periodic_image_centers(center, pore_radius, box_length), start=1
        ):
            region_id = f"pore{idx:04d}_{image_idx:02d}"
            lines.append(
                f"region {region_id} sphere "
                f"{image_center[0]:.8f} {image_center[1]:.8f} {image_center[2]:.8f} "
                f"{pore_radius:.8f} units box"
            )
            lines.append(f"delete_atoms region {region_id} compress yes")
            image_region_count += 1
    return (
        "\n".join(lines),
        centers,
        pore_radius,
        pore_centers_path,
        image_region_count,
    )


def build_input(args, case_dir: Path, potential_path: Path) -> str:
    lattice_a = 3.615
    box_length = args.cells * lattice_a
    initial_atoms = 4 * args.cells**3
    (
        pore_commands,
        pore_centers,
        pore_radius,
        pore_centers_path,
        pore_image_region_count,
    ) = pore_setup(args, case_dir, box_length)
    erate_per_ps = args.target_strain / (args.timestep_ps * args.tensile_steps)
    strain_rate_s = erate_per_ps * 1.0e12

    stress_csv = (case_dir / "stress_strain.csv").as_posix()
    trajectory_dump = (case_dir / "trajectory.lammpstrj").as_posix()
    potential = potential_path.as_posix()
    dump_block = ""
    undump_block = ""
    if args.dump_every > 0:
        dump_block = (
            f"\ndump tensile_dump all custom {args.dump_every} {trajectory_dump} "
            "id type x y z\n"
            "dump_modify tensile_dump sort id first yes"
        )
        undump_block = "\nundump tensile_dump"

    commands = f"""
clear
units metal
dimension 3
boundary p p p
atom_style atomic

lattice fcc {lattice_a}
region box block 0 {box_length:.8f} 0 {box_length:.8f} 0 {box_length:.8f} units box
create_box 1 box
create_atoms 1 box
mass 1 63.546

{pore_commands}

pair_style eam
pair_coeff 1 1 "{potential}"
neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes

timestep {args.timestep_ps}
thermo {args.sample_every}
thermo_style custom step atoms temp pe press pxx lx ly lz

minimize 1.0e-8 1.0e-10 1000 10000
velocity all create {args.temperature_k} {args.seed} mom yes rot no dist gaussian
fix equil all npt temp {args.temperature_k} {args.temperature_k} 0.1 iso 0.0 0.0 1.0
run {args.equil_steps}
unfix equil

run 0
variable L0 equal $(lx)
variable strain equal (lx-v_L0)/v_L0
variable stress_x_gpa equal -pxx/10000.0
variable natoms equal count(all)
variable step_out equal step
variable temp_out equal temp
variable pe_out equal pe
variable press_out equal press

fix pull_temp all npt temp {args.temperature_k} {args.temperature_k} 0.1 y 0.0 0.0 1.0 z 0.0 0.0 1.0
fix pull all deform 1 x erate {erate_per_ps:.12f} units box remap x
fix csv all print {args.sample_every} "${{step_out}},${{strain}},${{stress_x_gpa}},${{temp_out}},${{pe_out}},${{press_out}},${{natoms}}" file {stress_csv} screen no title "step,strain,stress_gpa,temp_K,pe_eV,press_bar,natoms"
{dump_block}
run {args.tensile_steps}
unfix csv
{undump_block}
unfix pull
unfix pull_temp
""".strip()

    metadata = {
        "cells": args.cells,
        "lattice_a_angstrom": lattice_a,
        "box_length_angstrom": box_length,
        "initial_atoms": initial_atoms,
        "pore_mode": args.pore_mode,
        "target_porosity": args.porosity,
        "pore_radius_angstrom": pore_radius,
        "void_radius_angstrom": pore_radius,
        "pore_count": len(pore_centers),
        "pore_gap_angstrom": args.pore_gap_a if args.pore_mode == "multi" else None,
        "pore_wall_margin_angstrom": None,
        "pore_boundary_mode": "periodic_minimum_image",
        "pore_center_domain": "[0,L)",
        "pore_image_region_count": pore_image_region_count,
        "pore_seed": args.pore_seed if args.pore_seed is not None else args.seed,
        "pore_centers_csv": str(pore_centers_path),
        "temperature_k": args.temperature_k,
        "equil_steps": args.equil_steps,
        "tensile_steps": args.tensile_steps,
        "timestep_ps": args.timestep_ps,
        "target_strain": args.target_strain,
        "effective_strain_rate_s^-1": strain_rate_s,
        "sample_every": args.sample_every,
        "dump_every": args.dump_every,
        "trajectory_dump": str(case_dir / "trajectory.lammpstrj") if args.dump_every > 0 else None,
        "seed": args.seed,
        "potential": str(potential_path),
    }
    (case_dir / "input_parameters.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return commands


def parse_args():
    parser = argparse.ArgumentParser(description="Run one nanoporous Cu tensile MD case.")
    parser.add_argument("--cells", type=int, default=10, help="FCC unit cells per box side")
    parser.add_argument("--porosity", type=float, default=0.10, help="Target porosity used to derive pore size or pore count")
    parser.add_argument("--pore-mode", choices=["single", "multi"], default="single")
    parser.add_argument("--pore-radius-a", type=float, default=3.6, help="Fixed pore radius for --pore-mode multi")
    parser.add_argument("--pore-count", type=int, default=None, help="Fixed pore count for --pore-mode multi")
    parser.add_argument("--pore-gap-a", type=float, default=0.2, help="Extra minimum gap between multi-pore surfaces")
    parser.add_argument(
        "--pore-wall-margin-a",
        type=float,
        default=0.0,
        help="Deprecated compatibility option; ignored because pores use periodic boundaries",
    )
    parser.add_argument("--pore-seed", type=int, default=None, help="Random seed for Monte Carlo pore placement")
    parser.add_argument("--max-pore-attempts", type=int, default=250000)
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--equil-steps", type=int, default=500)
    parser.add_argument("--tensile-steps", type=int, default=1000)
    parser.add_argument("--strain-rate-s", type=float, default=None, help="If set, derive tensile steps from target strain")
    parser.add_argument("--target-strain", type=float, default=0.05)
    parser.add_argument("--timestep-ps", type=float, default=0.001)
    parser.add_argument("--sample-every", type=int, default=20)
    parser.add_argument("--dump-every", type=int, default=0, help="Write trajectory every N tensile steps; 0 disables dumps")
    parser.add_argument("--seed", type=int, default=87287)
    parser.add_argument("--label", default=None, help="Optional case folder label")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.strain_rate_s is not None:
        timestep_s = args.timestep_ps * 1.0e-12
        args.tensile_steps = max(1, round(args.target_strain / (args.strain_rate_s * timestep_s)))

    root = Path(__file__).resolve().parents[1]
    label = args.label or datetime.now().strftime("case_%Y%m%d_%H%M%S")
    case_dir = root / "results" / label
    case_dir.mkdir(parents=True, exist_ok=True)

    potential_path = package_potential_path()
    if not potential_path.exists():
        raise SystemExit(f"Potential file not found: {potential_path}")

    input_text = build_input(args, case_dir, potential_path)
    input_file = case_dir / "in.generated.lammps"
    input_file.write_text(input_text + "\n", encoding="utf-8")

    log_path = case_dir / "log.lammps"
    start = time.perf_counter()
    lammps_exe = os.environ.get("LAMMPS_EXE")
    if lammps_exe:
        result = subprocess.run(
            [lammps_exe, "-log", str(log_path), "-in", str(input_file)],
            cwd=case_dir,
            check=False,
        )
        if result.returncode != 0:
            raise SystemExit(f"LAMMPS failed with return code {result.returncode}. See {log_path}")
        natoms, lammps_version = read_external_run_results(case_dir / "stress_strain.csv", log_path)
    else:
        if lammps is None:
            raise SystemExit(
                "The LAMMPS Python module is unavailable. Set LAMMPS_EXE to lmp.exe "
                "or install the LAMMPS Python package."
            )
        lmp = lammps(cmdargs=["-log", str(log_path)])
        lammps_version = int(lmp.version())
        lmp.file(str(input_file))
        natoms = int(lmp.get_natoms())
        lmp.close()
    runtime_s = time.perf_counter() - start

    metadata = json.loads((case_dir / "input_parameters.json").read_text(encoding="utf-8"))
    initial_atoms = int(metadata["initial_atoms"])
    actual_porosity = 1.0 - natoms / initial_atoms
    metadata.update(
        {
            "natoms_after_void": natoms,
            "atoms_after_pores": natoms,
            "actual_porosity": actual_porosity,
            "runtime_seconds": runtime_s,
            "lammps_version": lammps_version,
            "case_dir": str(case_dir),
            "stress_strain_csv": str(case_dir / "stress_strain.csv"),
            "generated_input": str(input_file),
            "log": str(log_path),
        }
    )
    (case_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Case folder: {case_dir}")
    print(f"Initial atoms: {initial_atoms}")
    print(f"Atoms after void deletion: {natoms}")
    print(f"Actual porosity from atom count: {actual_porosity:.5f}")
    print(f"Runtime seconds: {runtime_s:.2f}")
    print(f"Effective strain rate: {metadata['effective_strain_rate_s^-1']:.3e} s^-1")
    print(f"Stress-strain CSV: {case_dir / 'stress_strain.csv'}")


if __name__ == "__main__":
    main()
