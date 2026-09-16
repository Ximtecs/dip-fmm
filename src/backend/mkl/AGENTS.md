# oneMKL backend guidance

Inherit `../../AGENTS.md`. This directory owns guarded oneMKL vendor calls,
thread control, derived execution packing, and persistent reusable workspace.
It consumes canonical plans and must not reconstruct operator mathematics or
own high-level FMM pass sequencing.

For grouped M2L, preserve source/local scaling, explicit source and target
levels, stable matrix-column order, dynamic OpenMP group scheduling for the
gather and GEMM loops, the canonical per-target scatter order (the scatter is
parallel over targets, each visiting its (group, column) contributions in
group-ascending, column-ascending order, which reproduces the former serial
accumulation exactly), SGEMM/DGEMM arguments, one `mkl_set_num_threads_local`
bracket per thread per level, and evaluation-time buffer reuse (the gather
and translated scratch hold one level at a time). Keep
`<mkl.h>`, `MKL_INT`, `cblas_*`, and `mkl_set_num_threads_local` out of the
FMM orchestration layer.

`one_mkl_available()`, defined in `m2l.cpp`, is declared publicly by
`include/cdfmm/backend/mkl/availability.hpp` (the first public header under
`include/cdfmm/backend/mkl/`); `uniform_fmm.hpp` includes it and keeps
re-exporting the name.
