# Advisory parameter selection

## Manual mode — default

Normal construction remains explicit and unchanged. The advisers never alter
an existing evaluator or construct the evaluator used in production:

```python
options = cdfmm.UniformFmmOptions()
options.expansion_order = 6
options.tree.max_level = 4
fmm = cdfmm.UniformFmm(sources, targets, options)
```

## Advisory performance tuning

```python
suggestion = cdfmm.suggest_depth_for_performance(
    sources, targets, moments, order=6,
    backend=cdfmm.ExecutionBackend.CPU_STATIC,
    candidate_depths=[1, 2, 3, 4, 5],
)
```

Each candidate builds the same static `UniformFmm` used normally, performs a
warm-up, and reports median near-field (complete P2P branch), far-field
(P2M–L2P branch), and complete wall time over repeated evaluations. Sequential
backends are ranked by measured wall time. The partial CUDA backend overlaps
its near and far branches and additionally reports the heuristic
`max(near_seconds, far_seconds)` and their balance ratio. Complete wall time is
always retained as a sanity check. Failed or conservatively memory-limited
candidates remain in the diagnostics with a reason.

## Advisory accuracy tuning

```python
suggestion = cdfmm.suggest_parameters_for_accuracy(
    sources, targets, moments,
    desired_accuracy=1e-4,
    candidate_orders=range(2, 9),
    candidate_depths=range(1, 6),
)
```

A deterministic, configurable subset of targets is evaluated once with the
direct point-dipole reference. Every normal FMM candidate is compared at the
same indices, with explicit target/source identities preserved when supplied.
The adviser reports mean, RMS, and maximum relative error and mean and maximum
absolute field error. It recommends the fastest measured candidate whose
sampled RMS relative error meets the tolerance, rather than merely the lowest
passing order.

Both results are empirical suggestions, not global error bounds or universally
optimal settings. Performance depends on geometry, particle count, target
distribution, CPU, GPU, thread count, backend, and compiler optimisation.
Accuracy depends on geometry, moments, targets, order, depth, and sample
selection. For a repeated static problem, tune once during setup and explicitly
copy the returned order and depth into the production options.
