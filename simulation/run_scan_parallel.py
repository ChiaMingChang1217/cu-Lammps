import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def truthy(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_args():
    parser = argparse.ArgumentParser(description="Run nanoporous Cu scan with multiple concurrent cases.")
    parser.add_argument("--scan", default="study_nanoporous_cu/scan_multi_200.csv")
    parser.add_argument("--workers", type=int, default=3, help="Number of concurrent LAMMPS cases")
    parser.add_argument("--omp-threads", type=int, default=4, help="OpenMP threads per LAMMPS case")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-at", type=int, default=1)
    parser.add_argument("--only-figures", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record failed cases and continue with the remaining queue",
    )
    return parser.parse_args()


def build_command(single_case: Path, row):
    cmd = [
        sys.executable,
        str(single_case),
        "--label",
        row["case_id"],
        "--cells",
        row["cells"],
        "--porosity",
        row.get("porosity", row.get("target_porosity", "0.10")),
        "--pore-mode",
        row.get("pore_mode", "single"),
        "--temperature-k",
        row["temperature_k"],
        "--strain-rate-s",
        row["strain_rate_s"],
        "--target-strain",
        row["target_strain"],
        "--timestep-ps",
        row["timestep_ps"],
        "--equil-steps",
        row["equil_steps"],
        "--sample-every",
        row["sample_every"],
        "--dump-every",
        row["dump_every"],
        "--seed",
        row["seed"],
    ]
    if row.get("pore_mode", "single") == "multi":
        cmd.extend(
            [
                "--pore-radius-a",
                row["pore_radius_a"],
                "--pore-count",
                row["pore_count"],
                "--pore-gap-a",
                row["pore_gap_a"],
                "--pore-wall-margin-a",
                row["pore_wall_margin_a"],
                "--pore-seed",
                row["pore_seed"],
            ]
        )
    return cmd


def main():
    args = parse_args()
    workspace = Path.cwd()
    scan_path = workspace / args.scan
    scripts_dir = Path(__file__).resolve().parent
    single_case = scripts_dir / "run_single_case.py"
    status_path = scan_path.with_name(f"run_status_{scan_path.stem}_parallel.csv")

    rows = list(csv.DictReader(scan_path.open(encoding="utf-8")))
    queue = []
    for row_index, row in enumerate(rows, start=1):
        if row_index < args.start_at:
            continue
        if args.only_figures and not truthy(row.get("figure_case", "0")):
            continue
        case_dir = workspace / "study_nanoporous_cu" / "results" / row["case_id"]
        if (case_dir / "metadata.json").exists() and not args.force:
            continue
        queue.append((row_index, row))
        if args.limit is not None and len(queue) >= args.limit:
            break

    status_fields = ["time", "row_index", "case_id", "status", "runtime_seconds", "message"]
    new_status_file = not status_path.exists()
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(args.omp_threads)

    running = []
    completed = 0
    failed = 0
    next_idx = 0

    with status_path.open("a", newline="", encoding="utf-8") as status_file:
        writer = csv.DictWriter(status_file, fieldnames=status_fields)
        if new_status_file:
            writer.writeheader()

        print(
            f"Queued {len(queue)} cases from {scan_path} with {args.workers} workers x {args.omp_threads} OpenMP threads",
            flush=True,
        )

        while next_idx < len(queue) or running:
            while next_idx < len(queue) and len(running) < args.workers:
                row_index, row = queue[next_idx]
                next_idx += 1
                cmd = build_command(single_case, row)
                print(f"[{row_index}] start {row['case_id']}", flush=True)
                process = subprocess.Popen(
                    cmd,
                    cwd=workspace,
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                running.append(
                    {
                        "process": process,
                        "row_index": row_index,
                        "case_id": row["case_id"],
                        "start": time.perf_counter(),
                    }
                )

            time.sleep(2.0)
            still_running = []
            for item in running:
                code = item["process"].poll()
                if code is None:
                    still_running.append(item)
                    continue
                runtime = time.perf_counter() - item["start"]
                status = "done" if code == 0 else "failed"
                completed += int(code == 0)
                failed += int(code != 0)
                writer.writerow(
                    {
                        "time": datetime.now().isoformat(timespec="seconds"),
                        "row_index": item["row_index"],
                        "case_id": item["case_id"],
                        "status": status,
                        "runtime_seconds": f"{runtime:.3f}",
                        "message": f"returncode={code}",
                    }
                )
                status_file.flush()
                print(f"[{item['row_index']}] {status} {item['case_id']} ({runtime:.1f} s)", flush=True)
                if code != 0 and not args.continue_on_error:
                    for other in running:
                        if other["process"].poll() is None:
                            other["process"].terminate()
                    raise SystemExit(f"Case failed: {item['case_id']}")
            running = still_running

    print(f"Completed: {completed}; failed: {failed}; status log: {status_path}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

