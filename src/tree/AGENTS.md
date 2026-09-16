# Tree implementation

Inherit `../../AGENTS.md` and the public tree guidance. `common/` owns node
queries and deterministic Morton/indexing machinery; `uniform/` owns complete
octree construction and classical list1/list2 topology; `adaptive/` owns the
current compact subdivision and directed interaction traversal.

Preserve node IDs, sorting, permutations, near-neighbour definitions, M2L
admissibility, and interaction ordering exactly. `StaticFmmTopology` is
tree-owned interaction topology, not a plan artefact: `uniform/` and
`adaptive/` each populate it directly (via `build_uniform_fmm_topology` and
`AdaptiveTree`'s own construction respectively), and `fmm`/`plan` consume the
result by reference without rebuilding it. Do not read this as tree owning
plan or backend concepts; it owns spatial hierarchy and interaction facts
only.
