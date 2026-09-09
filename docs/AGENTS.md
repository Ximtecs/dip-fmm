# Documentation guidance

Inherit `../AGENTS.md`. Keep current capabilities, normative mathematics, the
target architecture, historical context, and performance studies distinct.

## Current structure

```text
docs/
|-- index.md, overview.md                 entry points and capability boundary
|-- architecture.md                      target developer architecture
|-- static-architecture.md, caching.md    current static-plan implementation
|-- math.md, *expansions.md,
|   laplace-derivatives.md                mathematical conventions
|-- geometry-models.md, uniform-tree.md,
|   operators.md, periodic-boundaries.md  subsystem guides
|-- backends.md, static-p2p.md,
|   cuda-m2l-performance.md, profiling.md backend/performance detail
|-- installation.md, getting-started.md,
|   fortran-interface.md, examples.md     user workflows
|-- validation.md, benchmarks.md          verification and measurement
|-- pre-refactor-baseline-v0.1.0.md       immutable-baseline explanation
`-- roadmap.md                            future work
```

## Ownership

- `architecture.md` is authoritative for target layers, dependencies, target
  taxonomy, and deferred decomposition.
- `static-architecture.md` describes how the current static plan actually owns
  and moves data. Keep it factual until production code changes.
- `math.md` is normative for signs, normalisation, notation, and formulas.
- `overview.md` and focused subsystem guides define the current support matrix.
- Performance reports state configuration and remain evidence, not universal
  architectural rules.

Always label current structure versus target structure. Do not describe future
directories or features as present. Use British English and relative links.
Architecture documents explain stable decisions, not a chronological diary.

## Validation

Check referenced commands against `CMakePresets.json`, CMake, and CI. Build the
Sphinx/MyST site with warnings as errors when documentation dependencies are
available:

```bash
sphinx-build -W --keep-going -b html docs docs/_build/html
```

For mathematical or capability claims, run the affected automated tests; a
documentation build checks links and syntax, not truth.
