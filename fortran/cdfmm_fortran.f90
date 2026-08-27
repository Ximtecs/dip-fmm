! SPDX-License-Identifier: Apache-2.0
module cdfmm_fortran
    use, intrinsic :: iso_c_binding
    implicit none
    private

    integer(c_int), parameter, public :: CDFMM_SUCCESS = 0_c_int
    integer(c_int), parameter, public :: CDFMM_ERROR_INVALID_ARGUMENT = 1_c_int
    integer(c_int), parameter, public :: CDFMM_ERROR_UNSUPPORTED = 2_c_int
    integer(c_int), parameter, public :: CDFMM_ERROR_RUNTIME = 3_c_int
    integer(c_int), parameter, public :: CDFMM_ERROR_CUDA_UNAVAILABLE = 4_c_int
    integer(c_int), parameter, public :: CDFMM_ERROR_UNKNOWN = 5_c_int
    integer(c_int), parameter, public :: CDFMM_PRECISION_FLOAT32 = 0_c_int
    integer(c_int), parameter, public :: CDFMM_PRECISION_FLOAT64 = 1_c_int
    integer(c_int), parameter, public :: CDFMM_BASIS_SPHERICAL = 0_c_int
    integer(c_int), parameter, public :: CDFMM_BASIS_CARTESIAN = 1_c_int
    integer(c_int), parameter, public :: CDFMM_BACKEND_AUTO = 0_c_int
    integer(c_int), parameter, public :: CDFMM_BACKEND_CPU_REFERENCE = 1_c_int
    integer(c_int), parameter, public :: CDFMM_BACKEND_CPU_STATIC = 2_c_int
    integer(c_int), parameter, public :: CDFMM_BACKEND_CUDA_PARTIAL = 3_c_int
    integer(c_int), parameter, public :: CDFMM_BACKEND_CUDA_FULL = 4_c_int
    integer(c_int), parameter, public :: CDFMM_STATIC_MATRIX_PORTABLE = 0_c_int
    integer(c_int), parameter, public :: CDFMM_STATIC_MATRIX_ONE_MKL = 1_c_int

    !> Native options for the recommended Fortran interface.
    type, public :: cdfmm_options_t
        integer :: order = 4
        integer :: depth = 1
        integer :: basis = CDFMM_BASIS_SPHERICAL
        integer :: precision = CDFMM_PRECISION_FLOAT32
        integer :: backend = CDFMM_BACKEND_AUTO
        integer :: static_matrix_backend = CDFMM_STATIC_MATRIX_PORTABLE
        logical :: periodic = .false.
        real(c_double) :: periodic_cell_center(3) = 0.0_c_double
        real(c_double) :: periodic_cell_lengths(3) = 0.0_c_double
        real(c_double) :: periodic_tolerance = 1.0e-12_c_double
    end type

    !> Exact C ABI options layout for advanced and backwards-compatible use.
    type, bind(c), public :: cdfmm_c_options_t
        integer(c_int32_t) :: struct_size
        integer(c_int) :: expansion_order
        integer(c_int) :: tree_depth
        integer(c_int) :: precision
        integer(c_int) :: expansion_basis
        integer(c_int) :: execution_backend
        integer(c_int) :: static_matrix_backend
    end type

    !> Owning wrapper for a persistent C FMM plan.
    type, public :: cdfmm_plan_t
        private
        type(c_ptr) :: handle = c_null_ptr
        integer(c_int) :: precision = -1_c_int
        integer(c_size_t) :: count = 0_c_size_t
    contains
        procedure :: destroy => cdfmm_plan_destroy_method
        procedure :: valid => cdfmm_plan_valid
        final :: cdfmm_plan_finalise
    end type

    character(len=:), allocatable, save :: wrapper_error

    public :: cdfmm_options, cdfmm_default_options
    public :: cdfmm_one_mkl_available
    public :: cdfmm_create_points, cdfmm_create_same_points
    public :: cdfmm_create_uniform_cuboids, cdfmm_create_same_uniform_cuboids
    public :: cdfmm_evaluate, cdfmm_evaluate_f32, cdfmm_evaluate_f64
    public :: cdfmm_destroy, cdfmm_last_error

    interface cdfmm_create_uniform_cuboids
        module procedure cdfmm_create_uniform_cuboids_native
        module procedure cdfmm_create_uniform_cuboids_c
    end interface

    interface cdfmm_evaluate
        module procedure cdfmm_evaluate_f32
        module procedure cdfmm_evaluate_f64
    end interface

    interface
        subroutine c_default_options(options) bind(c, name="cdfmm_default_options")
            import :: cdfmm_c_options_t
            type(cdfmm_c_options_t), intent(out) :: options
        end subroutine
        function c_one_mkl_available() result(available) bind(c, name="cdfmm_one_mkl_available")
            import :: c_int
            integer(c_int) :: available
        end function
        function c_create_points(ns, sx, sy, sz, nt, tx, ty, tz, identity, options, plan) result(status) &
                bind(c, name="cdfmm_plan_create_points")
            import :: c_size_t, c_double, c_ptr, c_int
            integer(c_size_t), value :: ns, nt
            real(c_double), intent(in) :: sx(*), sy(*), sz(*), tx(*), ty(*), tz(*)
            type(c_ptr), value :: identity, options
            type(c_ptr), intent(out) :: plan
            integer(c_int) :: status
        end function
        function c_create_cuboids(ns, sx, sy, sz, nt, tx, ty, tz, hx, hy, hz, identity, options, plan) &
                result(status) bind(c, name="cdfmm_plan_create_uniform_cuboid_sources")
            import :: c_size_t, c_double, c_ptr, c_int
            integer(c_size_t), value :: ns, nt
            real(c_double), intent(in) :: sx(*), sy(*), sz(*), tx(*), ty(*), tz(*)
            real(c_double), value :: hx, hy, hz
            type(c_ptr), value :: identity, options
            type(c_ptr), intent(out) :: plan
            integer(c_int) :: status
        end function
        function c_create_same_cuboids(count, x, y, z, hx, hy, hz, options, plan) result(status) &
                bind(c, name="cdfmm_plan_create_same_uniform_cuboids")
            import :: c_size_t, c_double, c_ptr, c_int
            integer(c_size_t), value :: count
            real(c_double), intent(in) :: x(*), y(*), z(*)
            real(c_double), value :: hx, hy, hz
            type(c_ptr), value :: options
            type(c_ptr), intent(out) :: plan
            integer(c_int) :: status
        end function
        function c_create_same_cuboids_periodic(count, x, y, z, hx, hy, hz, cell_center, cell_lengths, &
                setup_tolerance, options, plan) result(status) &
                bind(c, name="cdfmm_plan_create_same_uniform_cuboids_periodic")
            import :: c_size_t, c_double, c_ptr, c_int
            integer(c_size_t), value :: count
            real(c_double), intent(in) :: x(*), y(*), z(*)
            real(c_double), value :: hx, hy, hz
            real(c_double), intent(in) :: cell_center(3), cell_lengths(3)
            real(c_double), value :: setup_tolerance
            type(c_ptr), value :: options
            type(c_ptr), intent(out) :: plan
            integer(c_int) :: status
        end function
        function c_evaluate_f32(plan, mx, my, mz, hx, hy, hz) result(status) bind(c, name="cdfmm_plan_evaluate_f32")
            import :: c_ptr, c_float, c_int
            type(c_ptr), value :: plan
            real(c_float), intent(in) :: mx(*), my(*), mz(*)
            real(c_float), intent(out) :: hx(*), hy(*), hz(*)
            integer(c_int) :: status
        end function
        function c_evaluate_f64(plan, mx, my, mz, hx, hy, hz) result(status) bind(c, name="cdfmm_plan_evaluate_f64")
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

    !> Returns options initialised from the C library's canonical defaults.
    function cdfmm_options() result(options)
        type(cdfmm_options_t) :: options
        type(cdfmm_c_options_t) :: c_options
        call c_default_options(c_options)
        options%order = c_options%expansion_order
        options%depth = c_options%tree_depth
        options%basis = c_options%expansion_basis
        options%precision = c_options%precision
        options%backend = c_options%execution_backend
        options%static_matrix_backend = c_options%static_matrix_backend
    end function

    !> Reports whether the linked C++ library contains the oneMKL executor.
    logical function cdfmm_one_mkl_available()
        cdfmm_one_mkl_available = c_one_mkl_available() /= 0_c_int
    end function

    !> Initialises the advanced C ABI options structure.
    subroutine cdfmm_default_options(options)
        type(cdfmm_c_options_t), intent(out) :: options
        call c_default_options(options)
    end subroutine

    subroutine to_c_options(options, c_options)
        type(cdfmm_options_t), intent(in) :: options
        type(cdfmm_c_options_t), intent(out), target :: c_options
        call c_default_options(c_options)
        c_options%expansion_order = int(options%order, c_int)
        c_options%tree_depth = int(options%depth, c_int)
        c_options%precision = int(options%precision, c_int)
        c_options%expansion_basis = int(options%basis, c_int)
        c_options%execution_backend = int(options%backend, c_int)
        c_options%static_matrix_backend = int(options%static_matrix_backend, c_int)
    end subroutine

    !> Creates the common same-source/same-target uniform-cuboid plan.
    subroutine cdfmm_create_uniform_cuboids_native(plan, x, y, z, cell_size, options, ierr)
        type(cdfmm_plan_t), intent(inout) :: plan
        real(c_double), intent(in), target :: x(:), y(:), z(:)
        real(c_double), intent(in) :: cell_size(3)
        type(cdfmm_options_t), intent(in) :: options
        integer(c_int), intent(out) :: ierr
        type(cdfmm_c_options_t), target :: c_options

        call clear_wrapper_error()
        call cdfmm_destroy(plan)
        if (size(y) /= size(x) .or. size(z) /= size(x)) then
            call set_wrapper_error("x, y, and z must have equal sizes")
            ierr = CDFMM_ERROR_INVALID_ARGUMENT
            return
        end if
        call to_c_options(options, c_options)
        if (options%periodic) then
            ierr = c_create_same_cuboids_periodic(size(x, kind=c_size_t), x, y, z, cell_size(1), cell_size(2), &
                cell_size(3), options%periodic_cell_center, options%periodic_cell_lengths, &
                options%periodic_tolerance, c_loc(c_options), plan%handle)
        else
            ierr = c_create_same_cuboids(size(x, kind=c_size_t), x, y, z, cell_size(1), cell_size(2), cell_size(3), &
                                         c_loc(c_options), plan%handle)
        end if
        if (ierr == CDFMM_SUCCESS) then
            plan%precision = c_options%precision
            plan%count = size(x, kind=c_size_t)
        end if
    end subroutine

    !> Low-level constructor retaining separate source and target arrays.
    subroutine cdfmm_create_uniform_cuboids_c(plan, sx, sy, sz, tx, ty, tz, cell_size, options, ierr)
        type(cdfmm_plan_t), intent(inout) :: plan
        real(c_double), intent(in), target :: sx(:), sy(:), sz(:), tx(:), ty(:), tz(:)
        real(c_double), intent(in) :: cell_size(3)
        type(cdfmm_c_options_t), intent(in), target :: options
        integer(c_int), intent(out) :: ierr
        call clear_wrapper_error()
        call cdfmm_destroy(plan)
        ierr = c_create_cuboids(size(sx, kind=c_size_t), sx, sy, sz, size(tx, kind=c_size_t), tx, ty, tz, &
            cell_size(1), cell_size(2), cell_size(3), c_null_ptr, c_loc(options), plan%handle)
        if (ierr == CDFMM_SUCCESS) then
            plan%precision = options%precision
            plan%count = size(tx, kind=c_size_t)
        end if
    end subroutine

    subroutine cdfmm_create_same_uniform_cuboids(plan, x, y, z, cell_size, options, ierr)
        type(cdfmm_plan_t), intent(inout) :: plan
        real(c_double), intent(in), target :: x(:), y(:), z(:)
        real(c_double), intent(in) :: cell_size(3)
        type(cdfmm_c_options_t), intent(in), target :: options
        integer(c_int), intent(out) :: ierr
        call clear_wrapper_error()
        call cdfmm_destroy(plan)
        ierr = c_create_same_cuboids(size(x, kind=c_size_t), x, y, z, cell_size(1), cell_size(2), cell_size(3), &
                                     c_loc(options), plan%handle)
        if (ierr == CDFMM_SUCCESS) then
            plan%precision = options%precision
            plan%count = size(x, kind=c_size_t)
        end if
    end subroutine

    subroutine cdfmm_create_points(plan, sx, sy, sz, tx, ty, tz, options, ierr, identity)
        type(cdfmm_plan_t), intent(inout) :: plan
        real(c_double), intent(in), target :: sx(:), sy(:), sz(:), tx(:), ty(:), tz(:)
        type(cdfmm_c_options_t), intent(in), target :: options
        integer(c_int), intent(out) :: ierr
        integer(c_int32_t), intent(in), optional, target :: identity(:)
        type(c_ptr) :: identity_pointer
        call clear_wrapper_error()
        call cdfmm_destroy(plan)
        identity_pointer = c_null_ptr
        if (present(identity)) identity_pointer = c_loc(identity(1))
        ierr = c_create_points(size(sx, kind=c_size_t), sx, sy, sz, size(tx, kind=c_size_t), tx, ty, tz, &
                               identity_pointer, c_loc(options), plan%handle)
        if (ierr == CDFMM_SUCCESS) then
            plan%precision = options%precision
            plan%count = size(tx, kind=c_size_t)
        end if
    end subroutine

    subroutine cdfmm_create_same_points(plan, x, y, z, options, ierr)
        type(cdfmm_plan_t), intent(inout) :: plan
        real(c_double), intent(in), target :: x(:), y(:), z(:)
        type(cdfmm_c_options_t), intent(in), target :: options
        integer(c_int), intent(out) :: ierr
        integer(c_int32_t), allocatable, target :: identity(:)
        integer :: index
        allocate(identity(size(x)))
        identity = [(int(index - 1, c_int32_t), index = 1, size(x))]
        call cdfmm_create_points(plan, x, y, z, x, y, z, options, ierr, identity)
    end subroutine

    !> Evaluates a FLOAT32 plan without copying or converting moment arrays.
    subroutine cdfmm_evaluate_f32(plan, mx, my, mz, hx, hy, hz, ierr)
        type(cdfmm_plan_t), intent(in) :: plan
        real(c_float), intent(in), contiguous :: mx(:), my(:), mz(:)
        real(c_float), intent(out), contiguous :: hx(:), hy(:), hz(:)
        integer(c_int), intent(out) :: ierr
        call clear_wrapper_error()
        if (plan%precision /= CDFMM_PRECISION_FLOAT32) then
            call set_wrapper_error("FLOAT32 arrays require a FLOAT32 plan")
            ierr = CDFMM_ERROR_INVALID_ARGUMENT
            return
        end if
        if (any([size(mx, kind=c_size_t), size(my, kind=c_size_t), size(mz, kind=c_size_t), &
                 size(hx, kind=c_size_t), size(hy, kind=c_size_t), size(hz, kind=c_size_t)] /= plan%count)) then
            call set_wrapper_error("moment and field arrays must match the plan size")
            ierr = CDFMM_ERROR_INVALID_ARGUMENT
            return
        end if
        ierr = c_evaluate_f32(plan%handle, mx, my, mz, hx, hy, hz)
    end subroutine

    !> Evaluates a FLOAT64 plan without copying or converting moment arrays.
    subroutine cdfmm_evaluate_f64(plan, mx, my, mz, hx, hy, hz, ierr)
        type(cdfmm_plan_t), intent(in) :: plan
        real(c_double), intent(in), contiguous :: mx(:), my(:), mz(:)
        real(c_double), intent(out), contiguous :: hx(:), hy(:), hz(:)
        integer(c_int), intent(out) :: ierr
        call clear_wrapper_error()
        if (plan%precision /= CDFMM_PRECISION_FLOAT64) then
            call set_wrapper_error("FLOAT64 arrays require a FLOAT64 plan")
            ierr = CDFMM_ERROR_INVALID_ARGUMENT
            return
        end if
        if (any([size(mx, kind=c_size_t), size(my, kind=c_size_t), size(mz, kind=c_size_t), &
                 size(hx, kind=c_size_t), size(hy, kind=c_size_t), size(hz, kind=c_size_t)] /= plan%count)) then
            call set_wrapper_error("moment and field arrays must match the plan size")
            ierr = CDFMM_ERROR_INVALID_ARGUMENT
            return
        end if
        ierr = c_evaluate_f64(plan%handle, mx, my, mz, hx, hy, hz)
    end subroutine

    subroutine cdfmm_destroy(plan)
        type(cdfmm_plan_t), intent(inout) :: plan
        if (c_associated(plan%handle)) call c_destroy(plan%handle)
        plan%handle = c_null_ptr
        plan%precision = -1_c_int
        plan%count = 0_c_size_t
    end subroutine

    subroutine cdfmm_plan_destroy_method(plan)
        class(cdfmm_plan_t), intent(inout) :: plan
        if (c_associated(plan%handle)) call c_destroy(plan%handle)
        plan%handle = c_null_ptr
        plan%precision = -1_c_int
        plan%count = 0_c_size_t
    end subroutine

    logical function cdfmm_plan_valid(plan)
        class(cdfmm_plan_t), intent(in) :: plan
        cdfmm_plan_valid = c_associated(plan%handle)
    end function

    subroutine cdfmm_plan_finalise(plan)
        type(cdfmm_plan_t), intent(inout) :: plan
        call cdfmm_destroy(plan)
    end subroutine

    subroutine clear_wrapper_error()
        if (allocated(wrapper_error)) deallocate(wrapper_error)
    end subroutine

    subroutine set_wrapper_error(message)
        character(len=*), intent(in) :: message
        wrapper_error = message
    end subroutine

    function cdfmm_last_error() result(message)
        character(len=:), allocatable :: message
        character(kind=c_char), pointer :: chars(:)
        integer :: index, length
        if (allocated(wrapper_error)) then
            message = wrapper_error
            return
        end if
        call c_f_pointer(c_last_error(), chars, [4096])
        length = 0
        do while (length < size(chars) .and. chars(length + 1) /= c_null_char)
            length = length + 1
        end do
        allocate(character(len=length) :: message)
        do index = 1, length
            message(index:index) = chars(index)
        end do
    end function

end module cdfmm_fortran
