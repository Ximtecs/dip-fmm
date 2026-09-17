// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace cdfmm {

/** @brief Location and implementation used for a complete FMM evaluation. */
enum class ExecutionBackend {
    /// Resolve conservatively to the portable CPU static backend.
    Auto,
    /// Independent Cartesian CPU traversal for validation.
    CpuReference,
    /// Canonical static plan executed on the CPU.
    CpuStatic,
    /// Hybrid CPU hierarchy with CUDA M2L and P2P.
    CudaM2LP2P,
    /// User-facing alias for the hybrid CUDA backend.
    CudaPartial = CudaM2LP2P,
    /// Complete field-only device-resident static FMM.
    CudaFull,
    /// Compatibility alias for the hybrid CUDA backend.
    CudaM2L = CudaM2LP2P,
    /// Compatibility alias for the hybrid CUDA backend.
    CudaM2LStaticP2P = CudaM2LP2P
};

/**
 * @brief User-supplied hint about the arrangement of the point coordinates.
 *
 * The hint only steers derived execution choices (which CUDA P2P packing and
 * kernel execute the same canonical operator); it never changes the physical
 * problem or the mathematical result. An inaccurate hint can only cost
 * performance.
 */
enum class SpatialLayout {
    /// Arbitrary, random, or irregular coordinates: the measured defaults.
    General,
    /// Points on a regular lattice with repeated displacement structure, so
    /// the reduced-symmetry (signed tensor dictionary) P2P packing pays off.
    RegularGrid
};

/**
 * @brief Execution strategy of the point-source P2M and point-target L2P
 *        operators.
 *
 * The stored operators are coefficient rows precomputed at construction
 * (`3 C` scalars per point). For point far-field models in the spherical
 * basis the same operator can instead be recomputed from the sorted positions
 * during every evaluation (`Procedural`), which retains no rows; `Auto` keeps
 * the measured policy. A finite far-field model (prism or tetrahedron P2M or
 * L2P) always keeps its exact precomputed rows. The mathematical result is
 * the same for every value.
 */
enum class PointExpansionExecution {
    Auto,
    Precomputed,
    Procedural
};

} // namespace cdfmm
