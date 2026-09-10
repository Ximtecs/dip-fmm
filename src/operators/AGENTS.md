# Mathematical operator implementations

Implementations here define mathematical maps or static mathematical data.
Keep plans, backend resources, vendor libraries, and repeated-evaluation
orchestration out of this tree. These files are the authoritative homes for
static operator construction; compatibility names and structured names must
share these implementations. Do not create a second P2P definition with
different self or geometry semantics.

Current tree:

```text
operators/
|-- AGENTS.md
|-- l2l.cpp
|-- l2p.cpp
|-- m2l.cpp
|-- m2m.cpp
|-- p2m.cpp
|-- p2p.cpp
`-- spherical_cartesian_conversion.hpp
```
