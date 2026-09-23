# Cache guidance

Inherit `../AGENTS.md` and `../../AGENTS.md`. This subsystem persists solver
data that other layers define. It has no public header: everything here lives
below `src/`, and `internal.hpp` is the only interface shared between its
translation units.

## Current structure

```text
src/cache/
|-- internal.hpp    shared constants, CacheKind/CacheDescriptor, CachePayload,
|                   Writer/Reader, the io/format declarations, and the
|                   identity/payload records the boundary below is built on
|-- io.cpp          cache root and environment policy, the validated file
|                   container, checksum, and atomic temporary-file writes
|-- format.cpp      field-wise records for the persisted solver types
|-- keys.cpp        cache identity: digest, canonicalisation, key strings
|-- universal.cpp   depth-independent translation bank and periodic root
`-- geometry.cpp    geometry-dependent plan payload
```

The file container lives in `io.cpp`; `format.cpp` owns only the payload
records inside it. `src/fmm/plan_preparation.cpp` remains the coordinator that
asks for cached data, builds what is missing, and asks for a write.

## Cache/`UniformFmm` boundary

No file under `src/cache/` defines a `UniformFmm` member function, and
`internal.hpp` does not include `cdfmm/uniform_fmm.hpp`. Every cache entry
point is a free function in `cdfmm::detail::cache` taking an explicit
identity/payload record — `CacheIdentityInputs`/`CacheIdentity`,
`UniversalCacheIdentity`/`UniversalCachePayload`, or
`GeometryCacheIdentity`/`GeometryCachePayload`, all declared in
`internal.hpp` — built from specific solver/plan/tree/topology types
(`UniformTree`, `StaticFmmTopology`, `StaticM2LPlan`, `P2MPlan`, and similar),
never from `UniformFmm` itself. `src/fmm/execution_setup.cpp` (identity) and
`src/fmm/plan_preparation.cpp` (universal/geometry load and write) assemble
these records from `UniformFmm`'s private state, call the free functions, and
copy results back; they remain the only callers and the sole owners of the
cache-vs-build decision. Do not add a `UniformFmm` member function to
`src/cache/` to make a future change more convenient — extend the identity or
payload record instead, and keep the new field's origin (identity vs.
persisted payload vs. statistics) explicit, matching the classification below.

`P2MPlan`/`FloatP2MPlan` moved from private nested types of `UniformFmm` to
`include/cdfmm/plan/static_coefficient.hpp` specifically so the geometry
payload could reference them without seeing the rest of `UniformFmm`;
`ExpansionBasis` moved from `uniform_fmm.hpp` to `include/cdfmm/core/precision.hpp`
for the same reason (`CacheDescriptor` needs it). Neither move changed
`sizeof(UniformFmm)` or any public/ABI surface: both were already effectively
internal-only types reachable solely through `UniformFmm`'s public API by
value, never by their now-relocated name.

## Local invariants

- Cache defines no solver mathematics. Operator construction, tree building,
  plan policy, backend selection, and P2P packing policy belong elsewhere.
  `read_p2p_blocks` still builds the derived `StaticP2PCompactPlan` in the
  same pass that decodes the canonical blocks (the compact plan is not
  stored), but the canonical-to-compact row rule itself now lives once, in
  `assign_static_p2p_compact_row` (`src/plan/p2p/compact_row.hpp`, internal
  and not installed). Both the ordinary builder in `src/plan/p2p/compact.cpp`
  and this fused decode call that shared function per row; `format.cpp` no
  longer restates the field mapping. Do not unfuse this loop into
  decode-then-build as a tidy-up: keeping the two passes fused here was a
  deliberate performance choice, not part of the duplication that was
  resolved.
- The persistent binary format is a compatibility contract. Field order, widths,
  magic, `kCacheSchemaVersion`, `kOperatorVersion`, `kEndianMarker`,
  `kChecksumAlgorithm`, `CacheKind` values, the checksum algorithm, and the
  length-prefixed unaligned encoding are on-disk state. Changing any of them
  requires an explicit schema/version review, never an incidental cleanup.
- Cache keys are equally a compatibility surface. Every hash input, its order,
  its ASCII marker, the `1e-9` canonical coordinate, the compact grid and
  permutation-layout recognition, and the digest representation determine which
  files an existing installation can still read. Filename padding and the
  `_v02`/`_v04` suffixes are part of that identity.
- A plan whose near field is position-based (`PointGeometry`, resolved before
  preparation; `CacheIdentityInputs::position_based_p2p`) builds no pair
  tensors and writes an empty, format-valid P2P section. It is keyed apart:
  the filename segment is `_p2p_positions_` and the `"POSG"` marker is hashed
  after `use_reduced_symmetry_p2p` **only** for such plans, so every
  stored-tensor key and digest is byte-identical to before and no older
  binary ever looks such a file up. Its load treats any file holding pair
  tensors as a miss. Never persist an empty P2P section under a stored-tensor
  key: nothing validates the section against the topology, so a later
  stored-tensor plan would silently evaluate without a near field.
- Cache misses are non-fatal. A missing, truncated, corrupt, mismatched, or
  incompatible file must remain a rebuildable miss, and a failed write must
  remain a silent zero-byte result. A cache failure cannot alter an evaluation
  result.
- Cold and warm solver state must be equivalent, including the plan memory and
  statistics a warm construction reports.
- FP32 plans are decoded directly into FP32 storage. Do not route an FP32 warm
  cache through FP64.
- Cache writes stay atomic and tolerant of a concurrent writer: unique
  temporary file, complete write, `fsync`, `rename`, and an existing valid file
  counts as success. No global locks, no in-place truncation.
- Validation on load is deliberately strict where it exists today. Do not
  weaken the topology, permutation, identity, or plan-metadata checks, and do
  not symmetrise the FP32-only failure cleanup.

## Validation

Run the portable build and the cache tag before and after any change here:

```bash
cmake --fresh --preset dev
cmake --build --preset dev -j
ctest --preset dev
ctest --test-dir build --output-on-failure -R 'cache' \
  -E 'tetrahedron pairs|tetrahedron dispatch'
```

The existing tests compare keys between instances, not against golden
literals, so they cannot detect a uniform key change. A format or identity
claim also needs a before/after check against cache files written by the
previous build under a temporary `CDFMM_CACHE_DIR`: same filenames, a reported
hit, zero bytes written, and byte-identical files.
