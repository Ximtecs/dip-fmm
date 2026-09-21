# Getting started

This page is the shortest complete path from an installed library
([Installation](installation.md)) to a repeated field evaluation. Tutorial 1
under `examples/tutorials/` covers the same ground interactively with timings
and an exact reference.

## Units and shapes

| Quantity | Symbol | Unit | Shape |
|---|---|---|---|
| positions of sources and targets | $x$ | m | `(N, 3)` |
| total dipole moment per source | $m = V M$ | A m$^2$ | `(N, 3)` |
| magnetic field | $H = -\nabla\phi$ | A/m | `(N, 3)`, in the plan's precision |
| scalar potential (optional) | $\phi$ | A | `(N,)` |

Inputs are total moments, never magnetisation: the library performs no volume
scaling, for point dipoles or finite bodies alike. Positions may be given in
any consistent unit; the field comes back in the matching unit.

## The plan lifecycle

```text
define geometry              positions, optional prism/tetrahedron records,
                             optional self-identity map
    -> construct a plan      UniformFmm(sources, targets, options): tree,
                             operators, exact near field, backend resources
    -> evaluate              plan.evaluate(moments): the field of one state
    -> evaluate again        new moments, same plan
```

Everything geometry-dependent is done once at construction; `evaluate`
performs only operator application. Changing the moments or the identity map
is an `evaluate` argument; changing positions, records, order, depth, basis,
precision, backend or the periodic cell needs a new plan.

## Python

```python
import numpy as np
import cdfmm

rng = np.random.default_rng(0)
positions = rng.uniform(-50e-9, 50e-9, size=(4000, 3))     # metres
moments = rng.normal(size=(4000, 3)) * 1e-19                # A m^2
identities = np.arange(len(positions))                      # target i is source i

options = cdfmm.UniformFmmOptions()
options.expansion_order = 6            # p: accuracy versus cost
options.tree.max_level = 3             # octree depth
options.precision = cdfmm.StaticPrecision.FLOAT32   # the default
options.backend = cdfmm.ExecutionBackend.AUTO       # resolves to CPU_STATIC
options.fixed_target_source_indices = identities.tolist()

plan = cdfmm.UniformFmm(positions, positions, options)

result = plan.evaluate(moments, output="field", target_source_indices=identities)
H = result["H"]                        # (4000, 3) float32

for state in (rng.normal(size=(4000, 3)) * 1e-19 for _ in range(10)):
    H = plan.evaluate(state, target_source_indices=identities)["H"]
```

When targets are the sources themselves, the identity map tells each target
which source is *itself* so that the singular self pair is omitted; identity
is by index, never by coordinate equality. Pass `output="both"` for the
potential as well (CPU and hybrid backends). The exact $O(N^2)$ references
`cdfmm.direct_p2p_reference(targets, sources, moments, target_source_indices=...)`
and `cdfmm.DenseDirectPlan(...)` validate any configuration. Every
construction prints an initialisation summary of the requested and resolved
options ([Execution backends](backends.md)).

Assign `"cartesian"` to `options.expansion_basis` for the Cartesian basis, set
`options.source_geometry` and the records for finite bodies
([Geometry](geometry.md)), and see [Trees and parameter selection](trees-and-parameter-selection.md)
for choosing `expansion_order` and `tree.max_level`.

## C++

```cpp
#include <vector>
#include "cdfmm/cdfmm.hpp"

int main()
{
    std::vector<cdfmm::Vec3> positions = /* ... */;
    std::vector<cdfmm::Vec3> moments = /* total moments, A m^2 */;
    std::vector<int> identities(positions.size());
    for (int i = 0; i < static_cast<int>(identities.size()); ++i) {
        identities[i] = i;
    }

    cdfmm::UniformFmmOptions options;
    options.expansion_order = 6;
    options.tree.max_level = 3;
    options.precision = cdfmm::StaticPrecision::Float32;
    options.fixed_target_source_indices = identities;

    cdfmm::UniformFmm plan(positions, positions, options);

    // FP32 plans return FloatPotentialField; evaluate() widens to double.
    const auto first = plan.evaluate_float32(moments, cdfmm::OutputFlags::Field, identities);
    for (const auto& state : /* moment states */) {
        const auto values = plan.evaluate_float32(state, cdfmm::OutputFlags::Field, identities);
    }
}
```

`evaluate_into` / `evaluate_into_float32` write into caller-owned storage
without allocating. One `UniformFmm` is not re-entrant (its coefficients,
scratch and timers are mutable); separate objects may be evaluated
concurrently. The single-pair and summed direct formulas are
`cdfmm::p2p_dipole_pair` and `cdfmm::p2p_dipole_sum` (with `self_index` to
skip a source's own singular pair), and the low-level operators
`p2m_dipole`, `m2m_add`, `m2l_add`, `l2l_add`, `l2p_eval`, `m2p_eval` are
exposed for validation and teaching (tutorial 6).

## Diagnostics

`plan.last_timings` (C++ `last_timings()`) gives the phases of the last
evaluation, `plan.static_plan_statistics` the construction phases and retained
bytes of every operator, and `plan.cuda_plan_statistics` device residency and
per-evaluation transfer bytes. Timing is opt-in: the default
`options.timing_level = TimingLevel.OFF` reads no clock and leaves every
timing field at zero, `COARSE` measures the whole evaluation and its far and
near branches, and `DETAILED` every phase; `plan.set_timing_level(...)`
switches a built plan. Byte and count statistics are always populated, and
the level never changes results or the resolved plan
([Benchmarks and profiling](benchmarks.md)). The cache keys `universal_cache_key`,
`geometry_cache_key` and `periodic_cache_key` identify what a later process
will reuse ([Caching and periodicity](caching-and-periodicity.md)).
