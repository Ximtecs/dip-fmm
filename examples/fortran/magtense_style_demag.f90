! SPDX-License-Identifier: Apache-2.0
program magtense_style_demag
    use, intrinsic :: iso_c_binding
    use cdfmm_fortran
    implicit none

    integer, parameter :: ncells = 8
    integer, parameter :: nstates = 3
    type(cdfmm_plan_t) :: fmm
    type(cdfmm_options_t) :: options
    real(c_double) :: x(ncells), y(ncells), z(ncells)
    real(c_float) :: mx(ncells), my(ncells), mz(ncells)
    real(c_float) :: hx(ncells), hy(ncells), hz(ncells)
    real(c_double), parameter :: cell_size(3) = [1.0e-9_c_double, 1.0e-9_c_double, 1.0e-9_c_double]
    real(c_float), parameter :: volume = real(product(cell_size), c_float)
    real(c_float), parameter :: Ms = 8.0e5_c_float
    integer(c_int) :: ierr
    integer :: cell, state

    do cell = 1, ncells
        x(cell) = real(mod(cell - 1, 2), c_double) * cell_size(1)
        y(cell) = real(mod((cell - 1) / 2, 2), c_double) * cell_size(2)
        z(cell) = real((cell - 1) / 4, c_double) * cell_size(3)
    end do

    options = cdfmm_options()
    options%order = 6
    options%depth = 2
    options%basis = CDFMM_BASIS_SPHERICAL
    options%precision = CDFMM_PRECISION_FLOAT32
    options%backend = CDFMM_BACKEND_AUTO

    ! Periodicity uses this same constructor when periodic plan support is available.
    ! options%periodic = .true.
    ! options%periodic_cell_center = [0.5d-9, 0.5d-9, 0.5d-9]
    ! options%periodic_cell_lengths = [2.0d-9, 2.0d-9, 2.0d-9]

    call cdfmm_create_uniform_cuboids(fmm, x, y, z, cell_size, options, ierr)
    if (ierr /= CDFMM_SUCCESS) error stop cdfmm_last_error()

    do state = 1, nstates
        ! dip-fmm consumes total moment m=V*Ms*mhat; it does not apply this scaling.
        mx = volume * Ms * cos(real(state, c_float))
        my = volume * Ms * sin(real(state, c_float))
        mz = 0.0_c_float
        call cdfmm_evaluate(fmm, mx, my, mz, hx, hy, hz, ierr)
        if (ierr /= CDFMM_SUCCESS) error stop cdfmm_last_error()
        print '(a,i0,a,3es13.4)', 'state ', state, ' volume-averaged H(1) = ', hx(1), hy(1), hz(1)
    end do

    call fmm%destroy()
end program magtense_style_demag
