# Agent instructions for cartesian-dipole-fmm

## Project scope

This repository implements a fixed-geometry fast multipole method specialised
for magnetic dipole interactions. The current implementation includes:

- P2M: particle/dipole to multipole
- M2M: multipole to multipole
- M2L: multipole to local
- L2L: local to local
- L2P: local to particle/target
- P2P: direct near-field dipole interaction
- complete, non-adaptive `UniformTree` traversal;
- geometry-only `AdaptiveTree` construction and shared static-topology plans;
- point-dipole, rectangular-prism, and tetrahedron geometry where documented;
- portable CPU, oneMKL, optional CUDA-partial, and CUDA-full backends;
- persistent fixed-geometry plans, Python bindings, an unconditional C ABI,
  and an optional Fortran wrapper; and
- deterministic tests, examples, notebooks, and benchmark infrastructure.

The current architecture is intentionally retained during baseline work.
MagTense comparison notebooks and formulas are included, but a runtime
MagTense backend and broader micromagnetic integration remain outside this
repository's current scope.

The main quantity of interest is the magnetic field `H`. The scalar potential `phi` should be optional.

Use C++20.

Use namespace `cdfmm`.

## Mathematical convention

The Laplace Green's function is:

    G(r) = 1 / (4*pi*|r|)

For point dipoles `m_j` at source positions `x_j`, the scalar potential is:

    phi(x) = sum_j m_j dot (x - x_j) / (4*pi*|x - x_j|^3)

The magnetic field is:

    H(x) = -grad phi(x)

The direct pair field from source dipole `j` to target `i` is:

    H_ij =
        1/(4*pi) * [
            3*r_ij*(m_j dot r_ij)/|r_ij|^5
            - m_j/|r_ij|^3
        ]

where:

    r_ij = x_i - x_j

For source-point evaluation, self-interactions must be skipped.

## Comment and documentation style

Use a structured scientific-code comment style inspired by Fortran scientific code, adapted to modern C++.

General principles:

- Use Doxygen-style comments for public classes, structs, enums, and functions.
- Use short `//` comments for implementation details.
- Use British English in comments and documentation.
- Prefer `centre`, `initialise`, `normalise`, and `behaviour`.
- Explain mathematical intent, assumptions, indexing conventions, and non-obvious implementation details.
- Do not comment obvious C++ syntax.
- Public headers should be well documented.
- Implementation files should document algorithms and tricky indexing logic.
- Keep mathematical notation consistent with `docs/math.md` when that file exists.

Use this separator style for major file sections:

    //------------------------------------------------------------------------------
    // Section title
    //------------------------------------------------------------------------------

Use this style for public classes, structs, enums, and functions:

    /**
     * @brief Short one-sentence description.
     *
     * Longer explanation if needed. Describe the mathematical object,
     * storage convention, or algorithmic role.
     *
     * @param name Description of parameter.
     * @return Description of return value.
     */

For very short public functions, this is also acceptable:

    /// @brief Computes the dot product of two 3D vectors.

For mathematical operators, include the formula in the comment when useful.

Use implementation comments to explain indexing logic.

Good:

    // Skip invalid terms where alpha - e_k would have a negative component.

Avoid comments like:

    // Increment i
    // Create vector
    // Return result

because they add no information.

Use TODO comments in this exact format:

    // TODO(cdfmm): Short description of future work.

Examples:

    // TODO(cdfmm): Replace this generic contraction with a precomputed term list.
    // TODO(cdfmm): Cache M2L derivative tensors for repeated relative offsets.
    // TODO(cdfmm): Move this kernel to CUDA once the CPU reference is validated.

Use FIXME comments only for known incorrect behaviour:

    // FIXME(cdfmm): This assumes a uniform tree and is not valid for adaptive neighbours.

Use NOTE comments for important implementation constraints:

    // NOTE(cdfmm): Local coefficients include order zero even though dipole source
    // moments have no monopole term.

Use WARNING comments for assumptions that can break correctness:

    // WARNING(cdfmm): Self-interactions must be excluded when targets are source points.

For mathematical comments in source code, use ASCII-friendly notation rather than heavy LaTeX.

Good:

    // H_ij = 1/(4*pi) * [3*r*(m.r)/|r|^5 - m/|r|^3]

Keep full LaTeX formulas in `docs/math.md`.

## Formatting and readability

All C++ code must be formatted as readable multi-line code. Do not compress function bodies, loops, conditionals, lambdas, examples, or benchmarks into dense one-line blocks.

Prefer clarity over compactness.

Rules:

- Use one statement per line except for very small, obvious declarations.
- Use clear indentation and whitespace to show code structure.
- Dense one-line function bodies are forbidden.
- Dense one-line loops, conditionals, and lambdas are forbidden.
- Do not use minified or code-golf style formatting anywhere in the repository.
- Use descriptive variable names in examples, benchmarks, tests, and public-facing code.
- Short mathematical variable names are allowed only when they match nearby mathematical notation or established FMM conventions.
- Examples and benchmarks should be especially readable because they act as user-facing documentation.
- Comments should explain intent, setup, assumptions, indexing conventions, or mathematical meaning, not obvious syntax.
- Future agent contributions must preserve this style.

Unacceptable:

    int main(){ using namespace cdfmm; std::vector<Vec3> xs(10000),ms(10000); for(int i=0;i<10000;++i){xs[i]={0,0,0};ms[i]={1,0,0};} auto r=p2p_dipole_sum({0,0,1},xs,ms); std::cout<<r.H.z<<"\n"; }

Acceptable:

    int main()
    {
        using namespace cdfmm;

        constexpr int n_sources = 10000;

        std::vector<Vec3> source_positions(n_sources);
        std::vector<Vec3> source_moments(n_sources);

        // Initialise a deterministic source distribution for the example.
        for (int i = 0; i < n_sources; ++i) {
            source_positions[i] = {0.0, 0.0, 0.0};
            source_moments[i] = {1.0, 0.0, 0.0};
        }

        const Vec3 target_position{0.0, 0.0, 1.0};

        const auto result = p2p_dipole_sum(
            target_position,
            source_positions,
            source_moments
        );

        std::cout << "H_z = " << result.H.z << "\n";

        return 0;
    }

## Header file structure

Header files should generally follow this structure:

    // SPDX-License-Identifier: Apache-2.0
    #pragma once

    #include ...

    namespace cdfmm {

    //------------------------------------------------------------------------------
    // Constants
    //------------------------------------------------------------------------------

    //------------------------------------------------------------------------------
    // Public types
    //------------------------------------------------------------------------------

    //------------------------------------------------------------------------------
    // Public functions
    //------------------------------------------------------------------------------

    } // namespace cdfmm

## Source file structure

Source files should generally follow this structure:

    // SPDX-License-Identifier: Apache-2.0

    #include ...

    namespace cdfmm {

    //------------------------------------------------------------------------------
    // Helper functions
    //------------------------------------------------------------------------------

    //------------------------------------------------------------------------------
    // Public operator implementations
    //------------------------------------------------------------------------------

    } // namespace cdfmm

## Naming conventions

Use consistent terminology:

- source position
- target position
- dipole moment
- multipole coefficient
- local coefficient
- expansion centre
- derivative tensor
- near-field correction
- far-field contribution

Use variable names that mirror the mathematics:

- `alpha`, `beta`, `gamma`, `eta` for multi-indices
- `M` for multipole coefficients
- `L` for local coefficients
- `R` for target-centre minus source-centre
- `d` for translation vector
- `dx` for point displacement relative to an expansion centre
- `r` for target-source displacement in direct P2P
- `H` for magnetic field
- `phi` for scalar potential

Prefer British English in prose and comments.

Prefer `centre` over `center` in variable names.

## Testing requirements

Every new numerical operator should have tests.

Prefer simple analytic tests first, then convergence tests.

Required early tests:

- `MultiIndexSet` sizes for several expansion orders.
- Direct P2P axial dipole sanity test.
- Direct P2P transverse dipole sanity test.
- Output mode tests for field-only, potential-only, and both.
- Laplace derivative sanity tests.
- P2M + M2L + L2P convergence against direct P2P.

Run before finalising changes:

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCDFMM_BUILD_TESTS=ON -DCDFMM_BUILD_PYTHON=ON
    cmake --build build -j
    ctest --test-dir build --output-on-failure
    python -m pytest python_tests -v

## Python interface

The Python interface is provided using `pybind11` and should import as:

    import cdfmm

It exposes direct dipole operators, geometry records, both expansion bases,
uniform and adaptive topology inspection, complete static FMM evaluation, and
plan/timing/memory diagnostics. Keep additions focused on testing and
experimentation rather than promising an unvalidated high-level API.

## Repository workflow

Keep commits focused and reviewable. Use the checked-in CMake presets where
possible; the standard release verification is:

    cmake --fresh --preset release
    cmake --build --preset release -j
    ctest --preset release
    python -m pytest python_tests -v

Use the `cuda`, `notebooks`, `magtense`, or benchmark presets for their
documented optional configurations. Do not assume that an optional CUDA
device, oneMKL runtime, MagTense environment, or FMM3D dependency is present;
record such skips precisely. Preserve the current static-plan architecture
unless an architectural change is explicitly requested.

When in doubt, prefer a small correct implementation with tests over a large
untested implementation.
