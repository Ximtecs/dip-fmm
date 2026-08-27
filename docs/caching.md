# Persistent plan caches

`UniformFmm` normalises every problem to the root cube
`[-0.5,+0.5]^3` before constructing its static operators. This makes the
translation bank independent of physical placement and scale, and enables
validated binary reuse between processes.

The cache is enabled by default. Set `options.enable_cache = false` in C++ or
Python, or set `CDFMM_DISABLE_CACHE=1`, to force analytical reconstruction.
By default, the build stores persistent files under the source repository's
`dip-fmm/caches/v1/` directory. `CDFMM_CACHE_DIR` remains available as an
explicit base-directory override. Files use this structure:

```text
v1/
  universal/operators_spherical_p06_f32_m2m-m2l-l2l_v02.bin
  periodic/periodic_spherical_p06_f32_zerok0_tol1e-12_v02.bin
  plans/plan_spherical_p06_d05_f32_<sha256>_v02.bin
```

The compiled default is an absolute path to the source repository used for the
build. This keeps later MagTense processes and Python sessions on the same
machine pointed at the same on-disk cache even when launched from another
working directory. Generated files below `caches/` are ignored by Git.

Universal files contain the eight M2M templates, the standard 316-class M2L
bank, and the eight L2L templates. Periodic root operators are separate because
their setup tolerance is independently selectable. Geometry files contain tree
topology and Morton permutations, self metadata, P2M, M2L connectivity and
level scaling, L2P, and exact canonical P2P data; they do not duplicate the
universal matrices. The selected execution packing is deliberately not stored:
portable CPU derives particle-row SoA, while CUDA derives canonical AoS or
BSR(3) according to the fixed-identity contract and memory budget. CUDA then
uploads that derived representation to the selected device. Backend-only host
state and CUDA device allocations are recreated for each process.

FP32 geometry files are decoded directly into canonical FP32 operators. This
avoids widening the persistent plan to FP64 and quantising it back to FP32
before executor packing.

The geometry SHA-256 covers canonical source and target geometry, cuboid
dimensions and far-field flags, source/target modes, self identities, depth,
periodicity, basis, order, precision, and cache/math versions. Physical root
centre, root length, and changing moments are excluded. Complete regular grids
use a validated compact descriptor (dimensions, canonical origin, spacing, and
enumeration layout) instead of hashing every coordinate.

Each binary file has a validated header containing magic, schema/math versions,
endianness and data-model markers, kind, basis, order, precision, applicable depth, one
section offset and size, exact cache key, geometry digest where applicable, and
a fast 64-bit payload checksum. Writers use a unique temporary file, `fsync`, and
an atomic rename. Missing, truncated, corrupt, or incompatible files are
ignored and rebuilt; a cache failure cannot alter an evaluation result.

## Precomputing universal operators

The installed `cdfmm-precompute` utility populates universal files without a
physical experiment:

```console
cdfmm-precompute --basis spherical --orders 4,6,8 --precision f32
cdfmm-precompute --basis spherical --orders 6 --precision f32 \
  --periodic --periodic-tolerance 1e-12
```

Construction diagnostics and Python's `static_plan_statistics` report phase
times, hit states, and validated bytes read or written. Cache keys are exposed
as `universal_cache_key`, `periodic_cache_key`, and `geometry_cache_key`.

The unexecuted
[`simple_persistent_cache_reuse.ipynb`](../examples/simple_notebooks/simple_persistent_cache_reuse.ipynb)
notebook provides a small portable-CPU check of cold/warm loading, binary files
on disk, and translated/scaled geometry reuse.
