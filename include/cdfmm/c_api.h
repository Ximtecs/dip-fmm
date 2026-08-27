// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) && defined(CDFMM_C_EXPORTS)
#define CDFMM_C_API __declspec(dllexport)
#elif defined(_WIN32)
#define CDFMM_C_API __declspec(dllimport)
#else
#define CDFMM_C_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define CDFMM_ABI_VERSION 1u

typedef struct cdfmm_plan cdfmm_plan;

enum cdfmm_status {
    CDFMM_SUCCESS = 0,
    CDFMM_ERROR_INVALID_ARGUMENT = 1,
    CDFMM_ERROR_UNSUPPORTED = 2,
    CDFMM_ERROR_RUNTIME = 3,
    CDFMM_ERROR_CUDA_UNAVAILABLE = 4,
    CDFMM_ERROR_UNKNOWN = 5
};

enum cdfmm_precision {
    CDFMM_PRECISION_FLOAT32 = 0,
    CDFMM_PRECISION_FLOAT64 = 1
};

enum cdfmm_expansion_basis {
    CDFMM_BASIS_SPHERICAL = 0,
    CDFMM_BASIS_CARTESIAN = 1
};

enum cdfmm_execution_backend {
    CDFMM_BACKEND_AUTO = 0,
    CDFMM_BACKEND_CPU_REFERENCE = 1,
    CDFMM_BACKEND_CPU_STATIC = 2,
    CDFMM_BACKEND_CUDA_PARTIAL = 3,
    CDFMM_BACKEND_CUDA_FULL = 4
};

enum cdfmm_static_matrix_backend {
    CDFMM_STATIC_MATRIX_PORTABLE = 0,
    CDFMM_STATIC_MATRIX_ONE_MKL = 1
};

/** @brief Stable subset of plan construction options exposed to C callers. */
typedef struct cdfmm_options {
    uint32_t struct_size;
    int expansion_order;
    int tree_depth;
    int precision;
    int expansion_basis;
    int execution_backend;
    int static_matrix_backend;
} cdfmm_options;

/** @brief Small stable snapshot of persistent-plan diagnostics. */
typedef struct cdfmm_plan_stats {
    uint32_t struct_size;
    size_t source_count;
    size_t target_count;
    int expansion_order;
    size_t coefficient_count;
    size_t host_persistent_bytes;
    size_t device_persistent_bytes;
} cdfmm_plan_stats;

CDFMM_C_API uint32_t cdfmm_abi_version(void);
CDFMM_C_API const char* cdfmm_get_last_error(void);
CDFMM_C_API void cdfmm_default_options(cdfmm_options* options);
/** @return One when this library was built with the oneMKL backend. */
CDFMM_C_API int cdfmm_one_mkl_available(void);

CDFMM_C_API int cdfmm_plan_create_points(
    size_t source_count, const double* source_x, const double* source_y,
    const double* source_z, size_t target_count, const double* target_x,
    const double* target_y, const double* target_z,
    const int32_t* target_source_identity, const cdfmm_options* options,
    cdfmm_plan** plan);

CDFMM_C_API int cdfmm_plan_create_same_points(
    size_t count, const double* x, const double* y, const double* z,
    const cdfmm_options* options, cdfmm_plan** plan);

CDFMM_C_API int cdfmm_plan_create_uniform_cuboid_sources(
    size_t source_count, const double* source_x, const double* source_y,
    const double* source_z, size_t target_count, const double* target_x,
    const double* target_y, const double* target_z, double hx, double hy,
    double hz, const int32_t* target_source_identity,
    const cdfmm_options* options, cdfmm_plan** plan);

CDFMM_C_API int cdfmm_plan_create_same_uniform_cuboids(
    size_t count, const double* x, const double* y, const double* z,
    double hx, double hy, double hz, const cdfmm_options* options,
    cdfmm_plan** plan);

/**
 * @brief Creates a fully periodic same-source/same-target uniform-cuboid plan.
 *
 * This additive entry point keeps the version-one options structure unchanged
 * while exposing the periodic cell required by native-language wrappers. The
 * currently supported periodic mode is a three-dimensional cubic cell using
 * the zero-k0 convention.
 */
CDFMM_C_API int cdfmm_plan_create_same_uniform_cuboids_periodic(
    size_t count, const double* x, const double* y, const double* z,
    double hx, double hy, double hz, const double cell_centre[3],
    const double cell_lengths[3], double setup_tolerance,
    const cdfmm_options* options, cdfmm_plan** plan);

CDFMM_C_API int cdfmm_plan_evaluate_f32(
    cdfmm_plan* plan, const float* mx, const float* my, const float* mz,
    float* hx, float* hy, float* hz);
CDFMM_C_API int cdfmm_plan_evaluate_f64(
    cdfmm_plan* plan, const double* mx, const double* my, const double* mz,
    double* hx, double* hy, double* hz);
CDFMM_C_API int cdfmm_plan_get_stats(const cdfmm_plan* plan,
                                     cdfmm_plan_stats* stats);
CDFMM_C_API int cdfmm_plan_get_last_evaluation_seconds(
    const cdfmm_plan* plan, double* seconds);
CDFMM_C_API void cdfmm_plan_destroy(cdfmm_plan* plan);

#ifdef __cplusplus
}
#endif
