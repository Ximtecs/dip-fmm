! SPDX-License-Identifier: Apache-2.0
program test_fortran_api
    use, intrinsic :: iso_c_binding
    use cdfmm_fortran
    implicit none
    type(cdfmm_plan_t) :: plan
    type(cdfmm_options_t), target :: options
    real(c_double) :: x(2) = [0.0_c_double, 0.0_c_double]
    real(c_double) :: y(2) = [0.0_c_double, 0.0_c_double]
    real(c_double) :: z(2) = [0.0_c_double, 1.0_c_double]
    real(c_double) :: mx(2), my(2), mz(2), hx(2), hy(2), hz(2)
    real(c_double), parameter :: pi = acos(-1.0_c_double)
    integer(c_int) :: ierr

    call cdfmm_default_options(options)
    options%precision = CDFMM_PRECISION_FLOAT64
    options%expansion_basis = CDFMM_BASIS_CARTESIAN
    call cdfmm_create_same_points(plan, x, y, z, options, ierr)
    if (ierr /= CDFMM_SUCCESS) error stop cdfmm_last_error()
    mx = 0.0_c_double
    my = 0.0_c_double
    mz = [1.0_c_double, 0.0_c_double]
    call cdfmm_evaluate_f64(plan, mx, my, mz, hx, hy, hz, ierr)
    if (ierr /= CDFMM_SUCCESS) error stop cdfmm_last_error()
    if (abs(hz(2) - 1.0_c_double / (2.0_c_double*pi)) > 1.0e-12_c_double) &
        error stop "incorrect first C-ABI field"
    mz = [2.0_c_double, 0.0_c_double]
    call cdfmm_evaluate_f64(plan, mx, my, mz, hx, hy, hz, ierr)
    if (abs(hz(2) - 1.0_c_double / pi) > 1.0e-12_c_double) &
        error stop "persistent plan did not update moments"
    call cdfmm_destroy(plan)
end program test_fortran_api
