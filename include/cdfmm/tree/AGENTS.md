# Tree public interfaces

Inherit `../../../AGENTS.md` and `../AGENTS.md`. Common node, Morton, indexing,
and tree-statistics contracts live here beside uniform and adaptive interfaces.
Tree may depend on `math` and `core`, never geometry, FMM orchestration, CUDA,
or vendor execution libraries.

`AdaptiveTree` still exposes `StaticFmmTopology` as a compatibility seam. Do
not deepen that dependency; its removal belongs to the later plan refactor.
