! SPDX-License-Identifier: Apache-2.0
program magtense_style_demag
    use, intrinsic :: iso_c_binding
    use cdfmm_fortran
    implicit none

    integer, parameter :: n = 8
    type(cdfmm_plan_t) :: plan
    type(cdfmm_options_t), target :: options
    real(c_double) :: x(n), y(n), z(n)
    real(c_float) :: mx(n), my(n), mz(n), hx(n), hy(n), hz(n)
    real(c_double), parameter :: cell_size(3) = [1.0e-9_c_double, &
                                                 1.0e-9_c_double, &
                                                 1.0e-9_c_double]
    real(c_float), parameter :: cell_volume = real( &
        cell_size(1) * cell_size(2) * cell_size(3), c_float)
    real(c_float), parameter :: saturation_magnetisation = 8.0e5_c_float
    real(c_float) :: start_time, finish_time
    integer(c_int) :: ierr
    integer :: index, state

    ! Geometry and the persistent plan are initialised once.
    do index = 1, n
        x(index) = real(mod(index - 1, 2), c_double) * 1.0e-9_c_double
        y(index) = real(mod((index - 1) / 2, 2), c_double) * 1.0e-9_c_double
        z(index) = real((index - 1) / 4, c_double) * 1.0e-9_c_double
    end do
    call cdfmm_default_options(options)
    options%precision = CDFMM_PRECISION_FLOAT32
    ! Change this single option to CDFMM_BASIS_CARTESIAN for a Cartesian plan.
    options%expansion_basis = CDFMM_BASIS_SPHERICAL
    call cdfmm_create_same_uniform_cuboids( &
        plan, x, y, z, cell_size, options, ierr)
    if (ierr /= CDFMM_SUCCESS) error stop cdfmm_last_error()

    call cpu_time(start_time)
    do state = 1, 3
        ! dip-fmm consumes total moment m=V*M=V*Ms*mhat, not magnetisation.
        mx = cell_volume * saturation_magnetisation * cos(real(state, c_float))
        my = cell_volume * saturation_magnetisation * sin(real(state, c_float))
        mz = 0.0_c_float
        call cdfmm_evaluate_f32(plan, mx, my, mz, hx, hy, hz, ierr)
        if (ierr /= CDFMM_SUCCESS) error stop cdfmm_last_error()
        ! Returned H is analytically averaged over each receiving cuboid.
        print '(a,i0,a,3es13.4)', 'state ', state, &
            ' volume-averaged H(1) = ', hx(1), hy(1), hz(1)
    end do
    call cpu_time(finish_time)
    print '(a,es12.4,a)', 'three C-ABI evaluations: ', finish_time-start_time, ' s'
    call cdfmm_destroy(plan)
end program magtense_style_demag
