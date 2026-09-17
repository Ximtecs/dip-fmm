#!/usr/bin/env python3
"""Run benchmark_operator_representation over the Phase-3B.5b matrix.

Every case is one invocation of the benchmark for one operator family,
geometry pair, layout and mode; the benchmark appends its rows to a common
CSV with the reproduction metadata (compiler, CPU, GPU, commit).  The
analysis script `analyse_operator_representation.py` turns that CSV into the
precomputed-versus-procedural tables with the amortisation break-even.

Usage:
    python benchmarks/run_operator_representation.py \
        --binary build/benchmarks/benchmark_operator_representation \
        --output results/operator_representation.csv --suite all

Suites:
    p2p-hot        eight finite pairs plus point->point, every separation class
    p2p-streaming  complete list-1 neighbourhoods, regular and irregular
    p2p-large      prism<->point and tetrahedron<->point at --large-depth so
                   the stored operator exceeds the LLC (CPU) / L2 (GPU)
    p2p-cuda       the streaming sets with the CUDA stored representations and
                   the CUDA procedural prism kernel (needs a CUDA build)
    p2m, l2p       finite prism / tetrahedron expansions, hot and streaming
    all            everything above except p2p-cuda (add --cuda to include it)
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

FINITE_PAIRS = [
    ("point", "prism"), ("prism", "point"), ("prism", "prism"),
    ("point", "tetrahedron"), ("tetrahedron", "point"),
    ("prism", "tetrahedron"), ("tetrahedron", "prism"),
    ("tetrahedron", "tetrahedron"),
]
ALL_PAIRS = [("point", "point")] + FINITE_PAIRS

SEPARATIONS = ["self", "adjacent", "list1", "small-far", "far-safeguard"]


def git_commit(root: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


CASE_KEY = ("label", "stage", "source_geometry", "target_geometry", "layout",
            "separation", "mode", "backend")


def completed_cases(output: Path) -> set[tuple]:
    """Cases already in the CSV, so an interrupted matrix resumes."""
    if not output.exists():
        return set()
    with output.open() as stream:
        return {tuple(row.get(column, "") for column in CASE_KEY)
                for row in csv.DictReader(stream)}


def case_key(arguments: list[str]) -> tuple:
    def value(flag: str, default: str = "") -> str:
        return arguments[arguments.index(flag) + 1] if flag in arguments else default
    stage = value("--stage")
    source = value("--source", "-") if stage != "l2p" else "-"
    target = value("--target", "-") if stage != "p2m" else "-"
    if stage == "p2p":
        source = value("--source", "prism")
        target = value("--target", "prism")
    mode = value("--mode", "streaming")
    separation = value("--separation", "list1" if mode == "hot" else "lattice")
    if stage != "p2p":
        separation = "lattice"
    backend = "cuda" if "--no-cpu-procedural" in arguments else "cpu"
    return (value("--label"), stage, source, target, value("--layout", "regular"),
            separation, mode, backend)


def run(binary: str, arguments: list[str], dry_run: bool,
        done: set[tuple] = frozenset()) -> int:
    command = [binary, *arguments]
    if case_key(arguments) in done:
        print("=", " ".join(command), "(done, skipped)", file=sys.stderr, flush=True)
        return 0
    print("+", " ".join(command), file=sys.stderr, flush=True)
    if dry_run:
        return 0
    result = subprocess.run(command)
    if result.returncode != 0:
        print(f"! exit {result.returncode}", file=sys.stderr, flush=True)
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--output", required=True, help="CSV to append to")
    parser.add_argument("--suite", default="all",
                        choices=["p2p-hot", "p2p-streaming", "p2p-large",
                                 "p2p-cuda", "p2m", "l2p", "all"])
    parser.add_argument("--cuda", action="store_true",
                        help="include p2p-cuda in --suite all")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--bodies-per-leaf-axis", type=int, default=2)
    parser.add_argument("--pairs", type=int, default=512,
                        help="pairs of a hot P2P set")
    parser.add_argument("--hot-bodies", type=int, default=64,
                        help="bodies of a hot P2M/L2P set (one L1-resident slice)")
    parser.add_argument("--orders", default="4,6,8,10")
    parser.add_argument("--tetrahedron-streaming-orders", default="4,6,8",
                        help="orders of the tetrahedron streaming P2M/L2P runs; "
                             "the exact tetrahedron builder costs about 185 ms per "
                             "body at p 10, so the hot set carries that order")
    parser.add_argument("--evaluations", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--procedural-budget-seconds", type=float, default=60.0)
    parser.add_argument("--large-depth", type=int, default=4,
                        help="depth of the LLC / GPU-L2 exceeding streaming sets "
                             "(prism<->point and tetrahedron<->point only)")
    parser.add_argument("--label", default="")
    parser.add_argument("--commit", default="")
    parser.add_argument("--pairs-filter", default="",
                        help="comma separated source-target pairs, e.g. "
                             "prism-prism,tetrahedron-point")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true",
                        help="rerun cases whose rows are already in the CSV")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    commit = args.commit or git_commit(root)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    common = ["--output", args.output, "--commit", commit,
              "--evaluations", str(args.evaluations),
              "--warmups", str(args.warmups), "--samples", str(args.samples),
              "--depth", str(args.depth),
              "--bodies-per-leaf-axis", str(args.bodies_per_leaf_axis),
              "--procedural-budget-seconds", str(args.procedural_budget_seconds)]
    done = set() if args.no_resume else completed_cases(Path(args.output))

    pairs = ALL_PAIRS
    if args.pairs_filter:
        wanted = {tuple(item.split("-")) for item in args.pairs_filter.split(",")}
        pairs = [pair for pair in ALL_PAIRS if pair in wanted]

    suites = ([args.suite] if args.suite != "all"
              else ["p2p-hot", "p2p-streaming", "p2p-large", "p2m", "l2p"] +
                   (["p2p-cuda"] if args.cuda else []))
    failures = 0
    for suite in suites:
        label = f"{args.label}-{suite}" if args.label else suite
        common_labelled = [*common, "--label", label]
        if suite == "p2p-hot":
            for source, target in pairs:
                for separation in SEPARATIONS:
                    if separation == "self" and source == "point":
                        continue  # the point self pair is excluded by identity
                    layouts = ["regular", "irregular"] if separation == "list1" else ["regular"]
                    for layout in layouts:
                        failures += run(args.binary, [
                            "--stage", "p2p", "--source", source, "--target", target,
                            "--layout", layout, "--mode", "hot",
                            "--separation", separation, "--pairs", str(args.pairs),
                            *common_labelled], args.dry_run, done) != 0
        elif suite in ("p2p-streaming", "p2p-cuda"):
            for source, target in pairs:
                for layout in ["regular", "irregular"]:
                    extra = (["--cuda", "--no-cpu-precomputed", "--no-cpu-procedural"]
                             if suite == "p2p-cuda" else [])
                    failures += run(args.binary, [
                        "--stage", "p2p", "--source", source, "--target", target,
                        "--layout", layout, "--mode", "streaming", *extra,
                        *common_labelled], args.dry_run, done) != 0
            if suite == "p2p-cuda":
                # The GPU L2 (96 MB on an RTX 5090) holds the depth-3 tensors;
                # the large prism<->point sets exceed it.
                large = list(common_labelled)
                large[large.index("--depth") + 1] = str(args.large_depth)
                large[large.index("--label") + 1] = label + "-large"
                for source, target in [("prism", "point"), ("point", "prism")]:
                    if (source, target) not in pairs:
                        continue
                    for layout in ["regular", "irregular"]:
                        failures += run(args.binary, [
                            "--stage", "p2p", "--source", source, "--target", target,
                            "--layout", layout, "--mode", "streaming", "--cuda",
                            "--no-cpu-precomputed", "--no-cpu-procedural", *large],
                            args.dry_run, done) != 0
        elif suite == "p2p-large":
            large = list(common_labelled)
            large[large.index("--depth") + 1] = str(args.large_depth)
            for source, target in [("prism", "point"), ("point", "prism"),
                                   ("tetrahedron", "point"), ("point", "tetrahedron")]:
                if (source, target) not in pairs:
                    continue
                for layout in ["regular", "irregular"]:
                    failures += run(args.binary, [
                        "--stage", "p2p", "--source", source, "--target", target,
                        "--layout", layout, "--mode", "streaming", *large],
                        args.dry_run, done) != 0
        elif suite in ("p2m", "l2p"):
            role = "--source" if suite == "p2m" else "--target"
            for geometry in ["prism", "tetrahedron"]:
                for layout in ["regular", "irregular"]:
                    for mode in ["hot", "streaming"]:
                        orders = (args.tetrahedron_streaming_orders
                                  if geometry == "tetrahedron" and mode == "streaming"
                                  else args.orders)
                        failures += run(args.binary, [
                            "--stage", suite, role, geometry, "--layout", layout,
                            "--mode", mode, "--orders", orders,
                            "--pairs", str(args.hot_bodies), *common_labelled],
                            args.dry_run, done) != 0
    print(f"done, {failures} failed case(s), rows in {args.output}",
          file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
