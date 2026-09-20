# Documentation guidance

Inherit `../AGENTS.md`. The public documentation describes the **current**
library. History (refactor phases, closed audits, performance campaigns) lives
in Git and under `agent_docs/`, never in `docs/`.

## Current structure

```text
docs/
|-- index.md                          overview, capability boundary, toctree
|-- installation.md, getting-started.md
|-- geometry.md                       point / prism / tetrahedron models, records, self fields
|-- backends.md                       backends, precision, P2P packings, expansion execution, policy
|-- caching-and-periodicity.md        cache keys/format/environment, periodic cells
|-- trees-and-parameter-selection.md  uniform and adaptive trees, order/depth advisers
|-- examples.md                       tutorials, C++ demos, validation notebooks
|-- c-and-fortran.md                  C ABI and Fortran wrapper
|-- benchmarks.md                     drivers, timing categories, baselines, profiling
|-- architecture.md                   layers, ownership, static-plan lifecycle, invariants, validation map
|-- api.rst                           Doxygen/Breathe reference
`-- math/
    |-- conventions.md                kernel, notation, operators, normalisation (normative)
    |-- spherical-expansions.md       real spherical-harmonic basis and translations
    `-- finite-geometry.md            prism and tetrahedron formulas and safeguards
```

## Ownership

- `math/conventions.md` is normative for signs, normalisation, notation and
  formulas; `math/spherical-expansions.md` and `math/finite-geometry.md`
  extend it and must agree with it.
- `architecture.md` is authoritative for layers, dependencies and the
  representation invariants; it describes the current tree, not a target.
- `backends.md` owns the support matrix of backends, precisions and packings;
  `geometry.md` owns the geometry-model matrix. Keep one statement of each
  fact and link to it.
- `benchmarks.md` explains how to run and read measurements. Performance
  numbers that justify a policy are quoted sparingly and point to
  `agent_docs/performance_optimization.md`, which remains evidence, not a
  user promise.
- Every retained tutorial is listed in `examples.md`; a removed notebook is
  removed from the docs in the same change.

Do not describe future directories or features as present. Use British
English, relative links, and `$...$` MyST mathematics. Preserve every unique
mathematical statement when merging pages; move, do not delete.

## Validation

Check referenced commands against `CMakePresets.json`, CMake and CI. Build the
Sphinx/MyST site with warnings as errors (Doxygen must be on the path):

```bash
sphinx-build -W --keep-going -b html docs docs/_build/html
```

Grep the repository (source comments, `AGENTS.md` files, README, notebooks)
for links to a page you rename or remove. For mathematical or capability
claims, run the affected automated tests; a documentation build checks links
and syntax, not truth.
