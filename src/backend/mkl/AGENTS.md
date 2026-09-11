# oneMKL backend guidance

Inherit `../../AGENTS.md`. This directory owns guarded oneMKL vendor calls,
thread control, derived execution packing, and persistent reusable workspace.
It consumes canonical plans and must not reconstruct operator mathematics or
own high-level FMM pass sequencing.

For grouped M2L, preserve source/local scaling, explicit source and target
levels, stable matrix-column order, dynamic OpenMP group scheduling, serial
scatter, SGEMM/DGEMM arguments, and evaluation-time buffer reuse. Keep
`<mkl.h>`, `MKL_INT`, `cblas_*`, and `mkl_set_num_threads_local` out of the
FMM orchestration layer.
