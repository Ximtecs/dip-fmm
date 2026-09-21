# Phase-5 internal timing overhead study

**INTERNAL TIMING OVERHEAD STUDY — NOT ARTICLE1.**

These numbers measure one thing: what the solver's own timing instrumentation
costs a repeated evaluation, and that the opt-in `TimingLevel::Off` path is
indistinguishable from a solver with no instrumentation at all. They are an
engineering record for the implementation freeze, not a publication
benchmark, not a comparison with any other framework, and not a regression
baseline for the workload classes (that is `benchmarks/baselines/phase3d/`,
which is kept as measured: its rows include the then-unconditional
instrumentation and the Phase-3D runner now requests `--timing detailed` so
later runs compare like with like).

## Protocol

- Hardware: i9-14900KF (8 threads, `OMP_PROC_BIND=close`,
  `OMP_PLACES=cores`), RTX 5090 (sm_120, driver 595.84); g++ 15.3, nvcc 13.3,
  oneMKL, `benchmark-all` preset (Release, LTO).
- Headline: `benchmark_uniform_fmm`'s `evaluation_median`, an external
  `steady_clock` around complete `evaluate_into` calls on a warm plan
  (20 evaluations x 9 samples, 5 warm-ups); the standalone P2P rows are
  `benchmark_p2p`'s `host_total_s`, the minimum external host time over 50
  synchronous `evaluate()` calls. Every CUDA call had reached the same
  completion point as the public API (the functional `d2h` synchronisation).
- Builds: `start` is the untouched starting HEAD `8f54e6f`; `hard-off` is a
  temporary tree of that HEAD with every diagnostic event record, every
  `cudaEventElapsedTime` and every host evaluation clock removed and the
  three functional events kept (created with `cudaEventDisableTiming`);
  `off`, `coarse` and `detailed` are the final implementation at each
  `--timing` level.
- Each (case, build) pair ran 5 times with the builds interleaved; the value
  retained is the median of the five per-run medians. Run-to-run spread on
  the CUDA cases was about 1 us.

## Files

| File | Content |
|---|---|
| `phaseB_start_vs_hardoff.csv` | Starting HEAD against the hard-off lower bound. |
| `phaseJ_all_builds.csv` | Hard-off, starting HEAD and the final build at `off`, `coarse`, `detailed`. |
| `phaseB.json`, `phaseJ.json` | Every individual run behind the two tables. |
| `construction_clocks_before.csv` | Closure follow-up: construction wall time of the branch before the residual construction clocks were gated (`current`) against a hard-off copy with those clocks deleted (`hardoff`), 7 interleaved rounds, 8 threads. |
| `construction_clocks_final.csv` | The same study with the final implementation added at `Off` (`final`) and `Detailed` (`final-detailed`), 7 interleaved rounds, 8 threads. |
| `construction_clocks_serial.csv` | `hardoff`, `current` and `final` again with `OMP_NUM_THREADS=1`, which removes the thread wake-up jitter that dominated the 50 us dense point case. |

## Construction clocks (closure follow-up)

The evaluation study above left three construction-time clocks unconditional
(`UniformTree`, `AdaptiveTree`, the dense-direct construction records). The
follow-up measured them with an external `steady_clock` around complete
constructor calls, one process per build, cases and repetitions chosen so
that the smallest builds (a 6 us standalone tree, a 50 us dense point plan)
could resolve tens of nanoseconds. Each row is the median over the interleaved
rounds of the per-round median. `final` builds the standalone trees with
`collect_build_timings = false` and every plan at `TimingLevel::Off`;
`final-detailed` is the same binary at `Detailed`. The clocks cost about
17 ns per read, resolvable only on the 6 us tree (+3.8 %), and `final` equals
`hardoff` there to 0.01 us; the narrative is in
`agent_docs/performance_optimization.md`, Phase 5, section M.

## Reading them

`off / hardoff` is the acceptance ratio (target 1.00 within noise: at most
about 1 % on the medium cases and 2 % on the sub-millisecond CUDA cases);
`detailed / off` is what the old always-on instrumentation cost; `start /
hardoff` is the same cost measured before the redesign. The narrative and the
per-event accounting are in `agent_docs/performance_optimization.md`,
"Phase 5 — timing overhead and implementation freeze".
