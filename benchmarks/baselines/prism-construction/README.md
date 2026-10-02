# Prism endpoint construction follow-up, 2 October 2026

Engineering measurements only; not Article1 publication results. One common
rectangular-prism record, 512 centres on an 8 x 8 x 8 general grid, exact source
and target models, spherical basis, p10/d2, CPU static FP64, detailed timings,
and persistent caches disabled. The same deterministic moments and geometry
were used before and after. The baseline is the previous extension copied to
`/tmp/dipfmm-construction-baseline`; the final build is `build` from this
checkout. Runs used GCC 15.3.0, Release, native architecture, OpenMP and LTO,
fast math off, `OMP_NUM_THREADS=8`, and CPU affinity `0,2,4,6,8,10,12,14`.
The baseline is one sample and the final number is the median of three.

| Phase | Before | After | Ratio |
|---|---:|---:|---:|
| Total cold setup | 76.078 s | 0.114 s | 667x |
| Universal operator bank | 75.977 s | 65.388 ms | 1,162x |
| P2M | 14.829 ms | 1.131 ms | 13.1x |
| L2P | 21.014 ms | 1.281 ms | 16.4x |

The total setup and universal-bank improvements include shared multi-index
and Laplace-derivative changes. Endpoint clocks combine those common changes
with prism moment-table reuse, so they are not attributed to the prism table
alone. The table itself is tested against the unchanged scalar formula by
bit pattern through order 14.

Against the older extension, relative L2 differences were `3.87e-16` for
`H_far`, `1.01e-17` for `H_p2p`, and `1.84e-17` for `H_total`; output arrays
were not bit-identical. See `prism-grid8-order10-after.json` and its adjacent
NPZ for complete phase timings, byte counts, and field snapshots. The single
baseline measurement is retained at `/tmp/prism-before-p10.json` in the
measurement workspace; it is not committed because that temporary extension
is not portable.

Reproduce the final run from the repository root:

```bash
env OMP_NUM_THREADS=8 taskset -c 0,2,4,6,8,10,12,14 \
  /home/mihaa/.conda/envs/cdfmm/bin/python \
  benchmarks/benchmark_prism_construction.py \
  --build build --grid 8 --depth 2 --order 10 --precision float64 \
  --backend cpu_static --repetitions 3 \
  --output /tmp/prism-grid8-order10.json
```
