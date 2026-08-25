! SPDX-License-Identifier: Apache-2.0
module cdfmm_fortran
    use, intrinsic :: iso_c_binding
    implicit none
    private

    integer(c_int), parameter, public :: CDFMM_SUCCESS = 0_c_int
    integer(c_int), parameter, public :: CDFMM_PRECISION_FLOAT32 = 0_c_int
    integer(c_int), parameter, public :: CDFMM_PRECISION_FLOAT64 = 1_c_int
    integer(c_int), parameter, public :: CDFMM_BASIS_SPHERICAL = 0_c_int
    integer(c_int), parameter, public :: CDFMM_BASIS_CARTESIAN = 1_c_int
    integer(c_int), parameter, public :: CDFMM_BACKEND_AUTO = 0_c_int
    integer(c_int), parameter, public :: CDFMM_BACKEND_CPU_STATIC = 2_c_int
    integer(c_int), parameter, public :: CDFMM_BACKEND_CUDA_FULL = 4_c_int

    type, bind(c), public :: cdfmm_options_t
        integer(c_int32_t) :: struct_size
        integer(c_int) :: expansion_order
        integer(c_int) :: tree_depth
        integer(c_int) :: precision
        integer(c_int) :: expansion_basis
        integer(c_int) :: execution_backend
        integer(c_int) :: static_matrix_backend
    end type

    type, public :: cdfmm_plan_t
        private
        type(c_ptr) :: handle = c_null_ptr
    contains
        procedure :: destroy => cdfmm_plan_destroy_method
        final :: cdfmm_plan_finalise
    end type

    public :: cdfmm_default_options
    public :: cdfmm_create_points, cdfmm_create_same_points
    public :: cdfmm_create_uniform_cuboids
    public :: cdfmm_create_same_uniform_cuboids
    public :: cdfmm_evaluate_f32, cdfmm_evaluate_f64, cdfmm_destroy
    public :: cdfmm_last_error

    interface
        subroutine c_default_options(options) bind(c, name="cdfmm_default_options")
            import :: cdfmm_options_t
            type(cdfmm_options_t), intent(out) :: options
        end subroutine

        function c_create_points(ns, sx, sy, sz, nt, tx, ty, tz, identity, &
                                 options, plan) result(status) &
                                 bind(c, name="cdfmm_plan_create_points")
            import :: c_size_t, c_double, c_int32_t, c_ptr, c_int
            integer(c_size_t), value :: ns, nt
            real(c_double), intent(in) :: sx(*), sy(*), sz(*)
            real(c_double), intent(in) :: tx(*), ty(*), tz(*)
            type(c_ptr), value :: identity, options
            type(c_ptr), intent(out) :: plan
            integer(c_int) :: status
        end function

        function c_create_same(count, x, y, z, options, plan) result(status) &
                               bind(c, name="cdfmm_plan_create_same_points")
            import :: c_size_t, c_double, c_ptr, c_int
            integer(c_size_t), value :: count
            real(c_double), intent(in) :: x(*), y(*), z(*)
            type(c_ptr), value :: options
            type(c_ptr), intent(out) :: plan
            integer(c_int) :: status
        end function

        function c_create_cuboids(ns, sx, sy, sz, nt, tx, ty, tz, hx, hy, &
                                  hz, identity, options, plan) result(status) &
                                  bind(c, name="cdfmm_plan_create_uniform_cuboid_sources")
            import :: c_size_t, c_double, c_ptr, c_int
            integer(c_size_t), value :: ns, nt
            real(c_double), intent(in) :: sx(*), sy(*), sz(*)
            real(c_double), intent(in) :: tx(*), ty(*), tz(*)
            real(c_double), value :: hx, hy, hz
            type(c_ptr), value :: identity, options
            type(c_ptr), intent(out) :: plan
            integer(c_int) :: status
        end function

        function c_create_same_cuboids(count, x, y, z, hx, hy, hz, options, &
                                       plan) result(status) &
                bind(c, name="cdfmm_plan_create_same_uniform_cuboids")
            import :: c_size_t, c_double, c_ptr, c_int
            integer(c_size_t), value :: count
            real(c_double), intent(in) :: x(*), y(*), z(*)
            real(c_double), value :: hx, hy, hz
            type(c_ptr), value :: options
            type(c_ptr), intent(out) :: plan
            integer(c_int) :: status
        end function

        function c_evaluate_f32(plan, mx, my, mz, hx, hy, hz) result(status) &
                                bind(c, name="cdfmm_plan_evaluate_f32")
            import :: c_ptr, c_float, c_int
            type(c_ptr), value :: plan
            real(c_float), intent(in) :: mx(*), my(*), mz(*)
            real(c_float), intent(out) :: hx(*), hy(*), hz(*)
            integer(c_int) :: status
        end function

        function c_evaluate_f64(plan, mx, my, mz, hx, hy, hz) result(status) &
                                bind(c, name="cdfmm_plan_evaluate_f64")
            import :: c_ptr, c_double, c_int
            type(c_ptr), value :: plan
            real(c_double), intent(in) :: mx(*), my(*), mz(*)
            real(c_double), intent(out) :: hx(*), hy(*), hz(*)
            integer(c_int) :: status
        end function

        subroutine c_destroy(plan) bind(c, name="cdfmm_plan_destroy")
            import :: c_ptr
            type(c_ptr), value :: plan
        end subroutine

        function c_last_error() result(message) bind(c, name="cdfmm_get_last_error")
            import :: c_ptr
            type(c_ptr) :: message
        end function
    end interface

contains

    subroutine cdfmm_default_options(options)
        type(cdfmm_options_t), intent(out), target :: options
        call c_default_options(options)
    end subroutine

    subroutine cdfmm_create_same_uniform_cuboids(plan, x, y, z, cell_size, &
                                                  options, ierr)
        type(cdfmm_plan_t), intent(inout) :: plan
        real(c_double), intent(in) :: x(:), y(:), z(:)
        real(c_double), intent(in) :: cell_size(3)
        type(cdfmm_options_t), intent(in), target :: options
        integer(c_int), intent(out) :: ierr
        call cdfmm_destroy(plan)
        ierr = c_create_same_cuboids(size(x, kind=c_size_t), x, y, z, &
            cell_size(1), cell_size(2), cell_size(3), c_loc(options), &
            plan%handle)
    end subroutine

    subroutine cdfmm_create_points(plan, sx, sy, sz, tx, ty, tz, options, ierr, identity)
        type(cdfmm_plan_t), intent(inout) :: plan
        real(c_double), intent(in) :: sx(:), sy(:), sz(:), tx(:), ty(:), tz(:)
        type(cdfmm_options_t), intent(in), target :: options
        integer(c_int), intent(out) :: ierr
        integer(c_int32_t), intent(in), optional, target :: identity(:)
        type(c_ptr) :: identity_pointer
        call cdfmm_destroy(plan)
        identity_pointer = c_null_ptr
        if (present(identity)) identity_pointer = c_loc(identity(1))
        ierr = c_create_points(size(sx, kind=c_size_t), sx, sy, sz, &
            size(tx, kind=c_size_t), tx, ty, tz, identity_pointer, &
            c_loc(options), plan%handle)
    end subroutine

    subroutine cdfmm_create_same_points(plan, x, y, z, options, ierr)
        type(cdfmm_plan_t), intent(inout) :: plan
        real(c_double), intent(in) :: x(:), y(:), z(:)
        type(cdfmm_options_t), intent(in), target :: options
        integer(c_int), intent(out) :: ierr
        call cdfmm_destroy(plan)
        ierr = c_create_same(size(x, kind=c_size_t), x, y, z, &
                             c_loc(options), plan%handle)
    end subroutine

    subroutine cdfmm_create_uniform_cuboids(plan, sx, sy, sz, tx, ty, tz, &
                                            cell_size, options, ierr)
        type(cdfmm_plan_t), intent(inout) :: plan
        real(c_double), intent(in) :: sx(:), sy(:), sz(:), tx(:), ty(:), tz(:)
        real(c_double), intent(in) :: cell_size(3)
        type(cdfmm_options_t), intent(in), target :: options
        integer(c_int), intent(out) :: ierr
        call cdfmm_destroy(plan)
        ierr = c_create_cuboids(size(sx, kind=c_size_t), sx, sy, sz, &
            size(tx, kind=c_size_t), tx, ty, tz, cell_size(1), cell_size(2), &
            cell_size(3), c_null_ptr, c_loc(options), plan%handle)
    end subroutine

    subroutine cdfmm_evaluate_f32(plan, mx, my, mz, hx, hy, hz, ierr)
        type(cdfmm_plan_t), intent(in) :: plan
        real(c_float), intent(in) :: mx(:), my(:), mz(:)
        real(c_float), intent(out) :: hx(:), hy(:), hz(:)
        integer(c_int), intent(out) :: ierr
        ierr = c_evaluate_f32(plan%handle, mx, my, mz, hx, hy, hz)
    end subroutine

    subroutine cdfmm_evaluate_f64(plan, mx, my, mz, hx, hy, hz, ierr)
        type(cdfmm_plan_t), intent(in) :: plan
        real(c_double), intent(in) :: mx(:), my(:), mz(:)
        real(c_double), intent(out) :: hx(:), hy(:), hz(:)
        integer(c_int), intent(out) :: ierr
        ierr = c_evaluate_f64(plan%handle, mx, my, mz, hx, hy, hz)
    end subroutine

    subroutine cdfmm_destroy(plan)
        type(cdfmm_plan_t), intent(inout) :: plan
        if (c_associated(plan%handle)) call c_destroy(plan%handle)
        plan%handle = c_null_ptr
    end subroutine

    subroutine cdfmm_plan_destroy_method(plan)
        class(cdfmm_plan_t), intent(inout) :: plan
        call cdfmm_destroy(plan)
    end subroutine

    subroutine cdfmm_plan_finalise(plan)
        type(cdfmm_plan_t), intent(inout) :: plan
        call cdfmm_destroy(plan)
    end subroutine

    function cdfmm_last_error() result(message)
        character(len=:), allocatable :: message
        character(kind=c_char), pointer :: chars(:)
        integer :: index, length
        call c_f_pointer(c_last_error(), chars, [4096])
        length = 0
        do while (length < size(chars) .and. chars(length + 1) /= c_null_char)
            length = length + 1
        end do
        allocate(character(len=length) :: message)
        do index = 1, len(message)
            message(index:index) = chars(index)
        end do
    end function

end module cdfmm_fortran
