! SPDX-License-Identifier: Apache-2.0
program test_fortran_api
    use, intrinsic :: iso_c_binding
    use cdfmm_fortran
    implicit none

    type(cdfmm_plan_t) :: plan, low_plan
    type(cdfmm_options_t) :: options
    type(cdfmm_c_options_t), target :: c_options
    real(c_double) :: x(2) = [0.0_c_double, 0.0_c_double]
    real(c_double) :: y(2) = [0.0_c_double, 0.0_c_double]
    real(c_double) :: z(2) = [0.0_c_double, 1.0_c_double]
    real(c_double), parameter :: cell_size(3) = [0.2_c_double, 0.2_c_double, 0.2_c_double]
    real(c_float) :: mx(2), my(2), mz(2), hx(2), hy(2), hz(2), first_hz(2)
    real(c_float) :: low_hx(2), low_hy(2), low_hz(2)
    real(c_double) :: dmx(2), dmy(2), dmz(2), dhx(2), dhy(2), dhz(2)
    integer(c_int) :: ierr

    options = cdfmm_options()
    if (options%basis /= CDFMM_BASIS_SPHERICAL) error stop "incorrect default basis"
    if (cdfmm_one_mkl_available()) then
        options%static_matrix_backend = CDFMM_STATIC_MATRIX_ONE_MKL
    else
        options%static_matrix_backend = CDFMM_STATIC_MATRIX_PORTABLE
    end if
    call cdfmm_create_uniform_cuboids(plan, x, y, z, cell_size, options, ierr)
    if (ierr /= CDFMM_SUCCESS .or. .not. plan%valid()) error stop cdfmm_last_error()
    call plan%destroy()

    options%order = 3
    options%depth = 1
    options%basis = CDFMM_BASIS_CARTESIAN
    options%backend = CDFMM_BACKEND_CPU_REFERENCE
    call cdfmm_create_uniform_cuboids(plan, x, y, z, cell_size, options, ierr)
    if (ierr /= CDFMM_ERROR_INVALID_ARGUMENT .or. plan%valid()) then
        error stop "CPU reference should reject finite cuboids"
    end if

    options = cdfmm_options()
    options%backend = CDFMM_BACKEND_CUDA_FULL
    call cdfmm_create_uniform_cuboids(plan, x, y, z, cell_size, options, ierr)
    if (ierr == CDFMM_SUCCESS) then
        call plan%destroy()
    else if (ierr /= CDFMM_ERROR_CUDA_UNAVAILABLE) then
        error stop cdfmm_last_error()
    end if

    options = cdfmm_options()
    options%order = 5
    options%depth = 2
    options%basis = CDFMM_BASIS_CARTESIAN
    options%precision = CDFMM_PRECISION_FLOAT32
    options%backend = CDFMM_BACKEND_CPU_STATIC
    call cdfmm_create_uniform_cuboids(plan, x, y, z, cell_size, options, ierr)
    if (ierr /= CDFMM_SUCCESS .or. .not. plan%valid()) error stop cdfmm_last_error()

    mx = 0.0_c_float
    my = 0.0_c_float
    mz = [1.0_c_float, 0.0_c_float]
    call cdfmm_evaluate(plan, mx, my, mz, hx, hy, hz, ierr)
    if (ierr /= CDFMM_SUCCESS) error stop cdfmm_last_error()
    first_hz = hz
    mz(1) = 2.0_c_float
    call cdfmm_evaluate(plan, mx, my, mz, hx, hy, hz, ierr)
    if (ierr /= CDFMM_SUCCESS) error stop cdfmm_last_error()
    if (maxval(abs(hz - 2.0_c_float * first_hz)) > 2.0e-5_c_float) error stop "plan was not reused"

    dmx = 0.0_c_double
    dmy = 0.0_c_double
    dmz = 0.0_c_double
    call cdfmm_evaluate(plan, dmx, dmy, dmz, dhx, dhy, dhz, ierr)
    if (ierr /= CDFMM_ERROR_INVALID_ARGUMENT) error stop "precision mismatch was accepted"
    if (index(cdfmm_last_error(), "FLOAT64") == 0) error stop "precision mismatch message is unclear"

    call cdfmm_default_options(c_options)
    c_options%expansion_order = options%order
    c_options%tree_depth = options%depth
    c_options%expansion_basis = options%basis
    c_options%precision = options%precision
    c_options%execution_backend = options%backend
    call cdfmm_create_same_uniform_cuboids(low_plan, x, y, z, cell_size, c_options, ierr)
    if (ierr /= CDFMM_SUCCESS) error stop cdfmm_last_error()
    call cdfmm_evaluate_f32(low_plan, mx, my, mz, low_hx, low_hy, low_hz, ierr)
    if (maxval(abs(low_hz - hz)) > 1.0e-6_c_float) error stop "high- and low-level results differ"

    options%periodic = .true.
    options%backend = CDFMM_BACKEND_CPU_STATIC
    options%basis = CDFMM_BASIS_SPHERICAL
    options%periodic_cell_lengths = [2.0_c_double, 2.0_c_double, 2.0_c_double]
    call cdfmm_create_uniform_cuboids(plan, x, y, z, cell_size, options, ierr)
    if (ierr /= CDFMM_SUCCESS .or. .not. plan%valid()) error stop cdfmm_last_error()
    call cdfmm_evaluate(plan, mx, my, mz, hx, hy, hz, ierr)
    if (ierr /= CDFMM_SUCCESS) error stop cdfmm_last_error()

    call plan%destroy()
    call plan%destroy()
    call low_plan%destroy()

    options = cdfmm_options()
    options%precision = CDFMM_PRECISION_FLOAT64
    options%basis = CDFMM_BASIS_CARTESIAN
    call cdfmm_create_uniform_cuboids(plan, x, y, z, cell_size, options, ierr)
    if (ierr /= CDFMM_SUCCESS) error stop cdfmm_last_error()
    dmz = [1.0_c_double, 0.0_c_double]
    call cdfmm_evaluate(plan, dmx, dmy, dmz, dhx, dhy, dhz, ierr)
    if (ierr /= CDFMM_SUCCESS) error stop cdfmm_last_error()
    call cdfmm_evaluate(plan, mx, my, mz, hx, hy, hz, ierr)
    if (ierr /= CDFMM_ERROR_INVALID_ARGUMENT) error stop "reverse precision mismatch was accepted"
    if (index(cdfmm_last_error(), "FLOAT32") == 0) error stop "reverse mismatch message is unclear"
    call plan%destroy()

    block
        type(cdfmm_plan_t) :: finalised_plan
        call cdfmm_create_uniform_cuboids(finalised_plan, x, y, z, cell_size, options, ierr)
        if (ierr /= CDFMM_SUCCESS) error stop cdfmm_last_error()
    end block
end program test_fortran_api
