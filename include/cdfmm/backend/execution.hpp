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

} // namespace cdfmm
