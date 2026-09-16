#!/usr/bin/env python3
"""Run benchmark_uniform_fmm over a matrix of geometries, backends and P2P packings.

Each case is one benchmark_uniform_fmm invocation with a fixed geometry and
repeated moment updates; construction is excluded from the evaluation time.
The rows of every case CSV are collected into one table keyed by the case
name, so packings can be compared on identical operators.

Usage:
    python benchmarks/run_p2p_packing_matrix.py \
        --binary build/benchmarks/benchmark_uniform_fmm \
        --output results.csv --suite finite

Suites: periodic, finite, cuda-finite, crossover, all.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

COLUMNS = [
    "case", "backend", "precision", "sources", "depth", "order",
    "source_geometry", "target_geometry", "irregular_bodies", "regular_grid",
    "periodic", "p2p_packing_requested", "p2p_packing",
    "evaluation_median", "p2p", "cuda_p2p_kernel", "cuda_p2p_wait", "m2l",
    "p2m", "m2m", "l2l", "l2p", "near_field_pairs", "occupied_target_leaves",
    "p2p_unique_tensors", "p2p_dictionary_token_width_bytes",
    "p2p_dictionary_total_bytes", "p2p_canonical_total_bytes",
    "near_field_operator_bytes", "cuda_p2p_tensor_bytes",
    "cuda_persistent_device_bytes", "static_plan_bytes", "fmm_setup_seconds",
    "max_relative_error",
]


def run_case(binary: str, name: str, arguments: list[str], evaluations: int,
             warmups: int, samples: int, threads: int, scratch: Path) -> dict:
    output = scratch / (name.replace("/", "_") + ".csv")
    command = [
        binary, "--no-direct", "--no-workload-comparison",
        "--accuracy-targets", "0", "--warmups", str(warmups),
        "--evaluations", str(evaluations), "--samples", str(samples),
        "--threads", str(threads), "--output", str(output), *arguments,
    ]
    print("+", " ".join(command), file=sys.stderr, flush=True)
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        message = result.stderr.strip().splitlines()[-1][:160] if result.stderr.strip() else "unknown"
        print("  FAILED:", message, file=sys.stderr, flush=True)
        return {"case": name, "backend": "FAILED: " + message}
    with output.open() as stream:
        row = next(csv.DictReader(stream))
    row["case"] = name
    row["backend"] = row.get("execution_backend", "")
    row["precision"] = (arguments[arguments.index("--precision") + 1]
                        if "--precision" in arguments else "float32")
    row["regular_grid"] = "1" if "--regular-grid" in arguments else "0"
    return {column: row.get(column, "") for column in COLUMNS}


def geometry_arguments(kind: str) -> list[str]:
    if kind == "point":
        return []
    if kind == "prism":
        return ["--source-geometry", "prism", "--target-geometry", "prism"]
    if kind == "tetrahedron":
        return ["--source-geometry", "tetrahedron", "--target-geometry",
                "tetrahedron"]
    raise ValueError(kind)


def periodic_cases():
    for size, depth, order in ((10000, 3, 4), (50000, 4, 6)):
        for packing in ("particle-row-soa", "point-geometry"):
            for precision in ("float32", "float64"):
                yield (f"periodic/N{size}/d{depth}/{packing}/{precision}",
                       ["--backend", "cpu-static-matrix", "--sources",
                        str(size), "--targets", str(size), "--depth",
                        str(depth), "--order", str(order), "--periodic",
                        "--p2p-packing", packing, "--precision", precision])


def finite_cases():
    # Regular identical bodies (high tensor reuse) and irregular bodies,
    # CPU rows versus dictionary.
    for kind in ("prism", "tetrahedron"):
        for size, depth in ((4096, 3), (32768, 4)):
            for layout in ("regular", "irregular"):
                layout_arguments = (["--regular-grid"] if layout == "regular"
                                    else ["--irregular-bodies"])
                for packing in ("particle-row-soa", "tensor-dictionary"):
                    for precision in ("float32", "float64"):
                        yield (f"finite/{kind}/{layout}/N{size}/{packing}/{precision}",
                               ["--backend", "cpu-static-matrix", "--sources",
                                str(size), "--targets", str(size), "--depth",
                                str(depth), "--order", "6", *layout_arguments,
                                *geometry_arguments(kind), "--p2p-packing",
                                packing, "--precision", precision])
    # Regular point lattice for comparison with the finite reuse.
    for packing in ("particle-row-soa", "point-geometry", "tensor-dictionary"):
        yield (f"finite/point/regular/N32768/{packing}/float32",
               ["--backend", "cpu-static-matrix", "--sources", "32768",
                "--targets", "32768", "--depth", "4", "--order", "6",
                "--regular-grid", "--p2p-packing", packing,
                "--precision", "float32"])


def cuda_finite_cases():
    executors = (("canonical-aos", []), ("leaf-block", []), ("cuda-bsr3", []),
                 ("tensor-dictionary", []),
                 ("tensor-dictionary", ["--dictionary-target-owned"]),
                 ("tensor-dictionary", ["--dictionary-power2-microtiles"]))
    for kind in ("prism", "tetrahedron"):
        for layout in ("regular", "irregular"):
            layout_arguments = (["--regular-grid"] if layout == "regular"
                                else ["--irregular-bodies"])
            for backend in ("cuda-full", "cuda-partial"):
                for packing, extra in executors:
                    label = packing + "".join(extra).replace("--dictionary-", "/")
                    yield (f"cuda-finite/{kind}/{layout}/N32768/{backend}/{label}",
                           ["--backend", backend, "--sources", "32768",
                            "--targets", "32768", "--depth", "4", "--order",
                            "6", *layout_arguments, *geometry_arguments(kind),
                            "--p2p-packing", packing, *extra,
                            "--precision", "float32"])


def crossover_cases():
    # Random points at increasing particles per leaf; Auto and matched
    # packings on both CUDA backends.
    for size, depth in ((4096, 3), (8192, 3), (16384, 3), (24576, 3),
                        (32768, 3), (49152, 3), (65536, 3)):
        for backend in ("cuda-partial", "cuda-full"):
            for packing in ("auto", "leaf-block", "cuda-bsr3"):
                yield (f"crossover/random/N{size}/d{depth}/{backend}/{packing}",
                       ["--backend", backend, "--sources", str(size),
                        "--targets", str(size), "--depth", str(depth),
                        "--order", "6", "--p2p-packing", packing,
                        "--precision", "float32"])
    # Regular lattice with the dictionary and leaf blocks.
    for size, depth in ((32768, 4), (262144, 5)):
        for backend in ("cuda-partial", "cuda-full"):
            for packing in ("tensor-dictionary", "leaf-block"):
                yield (f"crossover/regular/N{size}/d{depth}/{backend}/{packing}",
                       ["--backend", backend, "--sources", str(size),
                        "--targets", str(size), "--depth", str(depth),
                        "--order", "6", "--regular-grid", "--p2p-packing",
                        packing, "--precision", "float32"])
    # Increasing N with the automatic policy.
    for size, depth in ((100000, 4), (200000, 5)):
        for backend in ("cuda-partial", "cuda-full"):
            yield (f"crossover/random/N{size}/d{depth}/{backend}/auto",
                   ["--backend", backend, "--sources", str(size),
                    "--targets", str(size), "--depth", str(depth),
                    "--order", "6", "--precision", "float32"])
    # Exact finite P2P on both backends (large stored-tensor near field).
    for kind in ("prism", "tetrahedron"):
        for backend in ("cuda-partial", "cuda-full"):
            for packing in ("auto", "cuda-bsr3", "leaf-block"):
                yield (f"crossover/{kind}/regular/N32768/{backend}/{packing}",
                       ["--backend", backend, "--sources", "32768",
                        "--targets", "32768", "--depth", "4", "--order", "6",
                        "--regular-grid", *geometry_arguments(kind),
                        "--p2p-packing", packing, "--precision", "float32"])


def cases(suite: str, cuda: bool):
    """Yield (name, arguments) pairs of the requested suite."""
    if suite in ("periodic", "all"):
        yield from periodic_cases()
    if suite in ("finite", "all"):
        yield from finite_cases()
    if suite in ("cuda-finite", "all") and cuda:
        yield from cuda_finite_cases()
    if suite in ("crossover", "all") and cuda:
        yield from crossover_cases()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--suite", default="all")
    parser.add_argument("--evaluations", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--filter", default="", help="substring of case names")
    parser.add_argument("--no-cuda", action="store_true")
    arguments = parser.parse_args()
    scratch = Path(arguments.output).with_suffix("")
    scratch.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, case_arguments in cases(arguments.suite, not arguments.no_cuda):
        if arguments.filter and arguments.filter not in name:
            continue
        rows.append(run_case(arguments.binary, name, case_arguments,
                             arguments.evaluations, arguments.warmups,
                             arguments.samples, arguments.threads, scratch))
        with open(arguments.output, "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
