#!/usr/bin/env python3
"""Combine benchmark CSVs and rank configurations against the production baseline."""

### ADJUSTED: Preserve and rank complete accelerated-backprop performance matrices.

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


BASELINE = ("lux", "reverse_diff_compiled", "production")
KEY_FIELDS = ("window_T", "window_N_obs")
CONFIG_FIELDS = ("network", "vjp", "solver_autodiff")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--all-output", required=True, type=Path)
    parser.add_argument("--ranking-output", required=True, type=Path)
    parser.add_argument("--workload-output", required=True, type=Path)
    return parser.parse_args()


def read_rows(paths):
    rows = []
    for path in paths:
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                row["source_csv"] = str(path)
                rows.append(row)
    return rows


def write_rows(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def successful_rows(rows):
    return [row for row in rows if row["status"] == "ok"]


def build_baselines(rows):
    baselines = {}
    for row in successful_rows(rows):
        config = tuple(row[field] for field in CONFIG_FIELDS)
        if config == BASELINE:
            baselines[tuple(row[field] for field in KEY_FIELDS)] = float(row["gradient_median_seconds"])
    return baselines


def rank_rows(rows, baselines):
    grouped = defaultdict(list)
    for row in successful_rows(rows):
        workload = tuple(row[field] for field in KEY_FIELDS)
        if workload not in baselines:
            continue
        config = tuple(row[field] for field in CONFIG_FIELDS)
        grouped[config].append((row, baselines[workload] / float(row["gradient_median_seconds"])))

    rankings = []
    required_workloads = set(baselines)
    for config, values in grouped.items():
        workloads = {tuple(row[field] for field in KEY_FIELDS) for row, _ in values}
        if workloads != required_workloads:
            continue
        speedups = [speedup for _, speedup in values]
        rankings.append(
            {
                "network": config[0],
                "vjp": config[1],
                "solver_autodiff": config[2],
                "workloads": len(values),
                "geomean_speedup_vs_production": math.exp(sum(math.log(value) for value in speedups) / len(speedups)),
                "minimum_speedup_vs_production": min(speedups),
                "maximum_speedup_vs_production": max(speedups),
                "mean_gradient_seconds": sum(float(row["gradient_median_seconds"]) for row, _ in values) / len(values),
                "mean_loss_seconds": sum(float(row["loss_median_seconds"]) for row, _ in values) / len(values),
            }
        )
    rankings.sort(key=lambda row: float(row["geomean_speedup_vs_production"]), reverse=True)
    return rankings


def rank_workloads(rows, baselines):
    rankings = []
    for row in successful_rows(rows):
        workload = tuple(row[field] for field in KEY_FIELDS)
        if workload not in baselines:
            continue
        rankings.append(
            {
                "window_T": row["window_T"],
                "window_N_obs": row["window_N_obs"],
                "network": row["network"],
                "vjp": row["vjp"],
                "solver_autodiff": row["solver_autodiff"],
                "gradient_median_seconds": row["gradient_median_seconds"],
                "loss_median_seconds": row["loss_median_seconds"],
                "first_call_seconds": row["first_call_seconds"],
                "gradient_allocations": row["gradient_allocations"],
                "relative_gradient_error": row["relative_gradient_error"],
                "relative_directional_error": row["relative_directional_error"],
                "speedup_vs_production": baselines[workload] / float(row["gradient_median_seconds"]),
            }
        )
    rankings.sort(
        key=lambda row: (
            float(row["window_T"]),
            int(row["window_N_obs"]),
            -float(row["speedup_vs_production"]),
        )
    )
    return rankings


def main():
    args = parse_args()
    rows = read_rows(args.inputs)
    all_fields = list(rows[0])
    write_rows(args.all_output, rows, all_fields)

    baselines = build_baselines(rows)
    rankings = rank_rows(rows, baselines)
    workload_rankings = rank_workloads(rows, baselines)
    ranking_fields = list(rankings[0])
    write_rows(args.ranking_output, rankings, ranking_fields)
    write_rows(args.workload_output, workload_rankings, list(workload_rankings[0]))

    print(f"Combined rows: {len(rows)}")
    print(f"Matched workloads: {len(baselines)}")
    print(f"Ranked complete configurations: {len(rankings)}")
    print(f"Ranked workload rows: {len(workload_rankings)}")
    print(f"Fastest: {rankings[0]}")


if __name__ == "__main__":
    main()
