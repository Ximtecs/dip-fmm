# Tree implementation

Inherit `../../AGENTS.md` and the public tree guidance. `common/` owns node
queries and deterministic Morton/indexing machinery; `uniform/` owns complete
octree construction and classical list1/list2 topology; `adaptive/` owns the
current compact subdivision and directed interaction traversal.

Preserve node IDs, sorting, permutations, near-neighbour definitions, M2L
admissibility, and interaction ordering exactly. Static-plan adaptation remains
in the transitional root implementation until the plan refactor.
