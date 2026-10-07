# Adaptive versus uniform tree on graded prism meshes (2026-10-07, branch adaptive_devel)

Per-body `RectangularPrism` plans on octree-refined Voronoi grain meshes, compared with a dense FP64
`DenseDirectPlan` on 512-1024 random targets. CPU static backend, spherical basis, 8 threads on the 8
physical P-cores (taskset 0,2,...,14, OMP_PROC_BIND=false), one plan at a time, idle machine, one
warm-up evaluation and the median of five timed ones. Commands: `run_comparison.sh`; driver
`../../adaptive_voronoi_sweep.py`; figure/table `../../plot_adaptive_comparison.py`. Engineering
evidence for this branch, not an Article1 benchmark.

Meshes (tile sizes relative to the coarsest tile):

| name | prisms | tile sizes | origin |
|---|---|---|---|
| smoke | 3,473 | 1, 1/2, 1/4 | MagTense grain_dipfmm smoke mesh (53 nm, 5 grains, base 4, 2 levels) |
| magtense_lev1 | 30,591 | 1, 1/2 | MagTense 400 nm, 200 grains, 8.33 nm band, base 16, 1 level (FP64) |
| magtense_lev2_fp32 | 192,284 | 1, 1/2, 1/4 | same structure, base 16, 2 levels (25 / 12.5 / 6.25 nm), FP32 |
| magtense_lev2_centre_fp32 | 40,636 | 1, 1/2, 1/4 | same, but only tiles whose centre is in the band refined (FP32) |
| synthetic_b16_l2 | 108,550 | 1, 1/2, 1/4 | `voronoi_octree_mesh`, 30 grains, base 16, 2 levels (FP64) |
| synthetic_b12_l1 | 8,742 | 1, 1/2 | base 12: tiles NOT aligned with dyadic boxes (alignment check) |

## Findings

1. **Containment decides correctness.** Every plan whose leaves are smaller than the coarsest tile
   (uniform depth = coarsest level + 1) is wrong by O(1): relative L2 0.07-12 on the MagTense and
   synthetic meshes, growing with the expansion order. The new leaf-containment warning flags every
   one of them and no contained plan. Contained plans of either tree agree with dense to the
   expected truncation error (order 4: ~4e-3, order 6: ~5e-4, order 8: ~8e-5).
2. **Alignment is part of containment.** On the non-dyadic base-12 mesh tiles straddle leaf
   boundaries whatever the tree does: the adaptive plan has 453 protruding bodies and 7.5x the error
   of the contained depth-2 uniform plan. Meshes for dip-fmm should be dyadic and share the FMM
   root; the MagTense base-16 meshes are, and the C ABI adaptive creator infers that root.
3. **Production case (MagTense 192k, FP32, order 6): the adaptive tree wins.** Capacity 32:
   rel. L2 5.8e-4, 103 ms per evaluation, 4.6 GB plan, 16 s build. The best contained uniform tree
   (depth 4): 5.7e-4, 131 ms, 19.0 GB, 24 s. That is 1.27x faster evaluation, 4.1x less memory and
   1.5x faster construction at the same accuracy; capacity 16 is similar (108 ms, 3.6 GB).
4. **On the other meshes the uniform tree is as good or better.** One-level MagTense mesh: identical
   (adaptive capacity 16/32 reproduces uniform depth 4). Synthetic FP64 two-level mesh at order 6:
   uniform depth 4 108 ms against adaptive 212-512 ms, because the adaptive tree creates 3-6x more
   M2L interactions (cross-level ones included) and FP64 M2L at order >= 6 dominates; the adaptive
   plans still use 30-50 % less memory there. Centre-criterion mesh: uniform 15.6 ms, adaptive
   18-23 ms.
5. **Defaults.** max_depth 5 versus 6 never changed a result: capacity stops the subdivision first
   on all these meshes, so the depth cap only matters as a safety limit. Capacity 16 and 32 are
   within a few per cent of each other; 64 collapses to the depth-3-like tree on the MagTense and
   synthetic meshes (7x the near-field pairs, slowest build). Capacity 32 / depth 5 is a sensible
   default.

## Open points

- CUDA execution with per-body sizes and on adaptive topologies is untested (CPU build only).
- Adaptive far-field storage is dominated by per-body finite P2M/L2P operators: mixed-level leaves
  give more distinct body offsets and so less exact reuse.
- The cross-level M2L cost in FP64 at high order is the main performance gap of the adaptive tree.
- The MagTense adapter (`DipFmmDemag.f90`) still accepts only uniform grids; the bindings for it now
  exist (`cdfmm_create_variable_cuboids`, `cdfmm_create_adaptive_variable_cuboids`).
