import argparse
import csv
import json
from pathlib import Path


def read_stress_strain(path: Path):
    rows = []
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "strain": float(row["strain"]),
                    "stress_gpa": float(row["stress_gpa"]),
                    "temp_K": float(row["temp_K"]),
                    "pe_eV": float(row["pe_eV"]),
                }
            )
    return rows


def linear_slope(points):
    n = len(points)
    if n < 2:
        return None
    sx = sum(p["strain"] for p in points)
    sy = sum(p["stress_gpa"] for p in points)
    sxx = sum(p["strain"] ** 2 for p in points)
    sxy = sum(p["strain"] * p["stress_gpa"] for p in points)
    denom = n * sxx - sx * sx
    if abs(denom) < 1.0e-20:
        return None
    return (n * sxy - sx * sy) / denom


def trapezoid_area(rows):
    if len(rows) < 2:
        return None
    area = 0.0
    for left, right in zip(rows, rows[1:]):
        dx = right["strain"] - left["strain"]
        area += 0.5 * (left["stress_gpa"] + right["stress_gpa"]) * dx
    return area


def metrics(rows):
    elastic = [row for row in rows if 0.002 <= row["strain"] <= 0.02]
    young_modulus = linear_slope(elastic)
    uts_row = max(rows, key=lambda row: row["stress_gpa"])
    yield_candidates = [row for row in rows if row["strain"] >= 0.002]
    yield_row = max(yield_candidates[: max(3, len(yield_candidates) // 5)], key=lambda row: row["stress_gpa"])
    return {
        "young_modulus_gpa": young_modulus,
        "yield_strength_gpa": yield_row["stress_gpa"],
        "yield_strain": yield_row["strain"],
        "ultimate_tensile_strength_gpa": uts_row["stress_gpa"],
        "uts_strain": uts_row["strain"],
        "toughness_gpa": trapezoid_area(rows),
        "final_stress_gpa": rows[-1]["stress_gpa"],
        "final_strain": rows[-1]["strain"],
        "mean_temperature_k": sum(row["temp_K"] for row in rows) / len(rows),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Extract ML-ready labels from completed MD cases.")
    parser.add_argument("--scan", default="study_nanoporous_cu/scan_200.csv")
    parser.add_argument("--output", default="study_nanoporous_cu/ml_dataset.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    workspace = Path.cwd()
    scan_path = workspace / args.scan
    output_path = workspace / args.output
    rows = list(csv.DictReader(scan_path.open(encoding="utf-8")))

    output_rows = []
    for row in rows:
        case_dir = workspace / "study_nanoporous_cu" / "results" / row["case_id"]
        metadata_path = case_dir / "metadata.json"
        stress_path = case_dir / "stress_strain.csv"
        if not metadata_path.exists() or not stress_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        series = read_stress_strain(stress_path)
        if len(series) < 3:
            continue
        row_out = {
            "case_id": row["case_id"],
            "cells": metadata["cells"],
            "target_porosity": metadata["target_porosity"],
            "actual_porosity": metadata.get("actual_porosity", metadata["target_porosity"]),
            "porosity": metadata.get("actual_porosity", metadata["target_porosity"]),
            "pore_mode": metadata.get("pore_mode", "single"),
            "pore_radius_angstrom": metadata.get("pore_radius_angstrom", metadata.get("void_radius_angstrom")),
            "void_radius_angstrom": metadata.get("void_radius_angstrom", metadata.get("pore_radius_angstrom")),
            "pore_count": metadata.get("pore_count", 1),
            "temperature_k": metadata["temperature_k"],
            "strain_rate_s": metadata["effective_strain_rate_s^-1"],
            "seed": metadata["seed"],
            "pore_seed": metadata.get("pore_seed", metadata["seed"]),
            "initial_atoms": metadata.get("initial_atoms", ""),
            "atoms_after_pores": metadata.get("atoms_after_pores", metadata["natoms_after_void"]),
            "natoms_after_void": metadata["natoms_after_void"],
            "runtime_seconds": metadata["runtime_seconds"],
            "figure_case": row.get("figure_case", "0"),
        }
        row_out.update(metrics(series))
        output_rows.append(row_out)

    if not output_rows:
        raise SystemExit("No completed cases found.")

    fields = list(output_rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Wrote {len(output_rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
