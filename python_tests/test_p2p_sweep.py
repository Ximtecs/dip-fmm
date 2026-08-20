from types import SimpleNamespace

from benchmarks.run_p2p_sweep import (
    add_case_columns,
    build_cases,
    estimate_uniform_interactions,
    generate_figures,
    parse_benchmark_output,
    SweepCase,
)


SAMPLE_OUTPUT = """\
case,particles=4096,depth=3,occupied_leaves=512,mean_occupancy=8,max_occupancy=8,unique_occupancies=1,uniform=yes,particle_interactions=681472,leaf_interactions=10648,threads=8
implementation,evaluation_s,interactions_per_s,speedup,tensor_bytes,index_bytes,row_metadata_bytes,leaf_metadata_bytes,scratch_bytes,total_bytes,checksum
canonical-aos,0.001,681472000,1,100,20,4,0,0,124,-2
onemkl-bsr3,0.0008,851840000,1.25,150,20,4,0,0,174,-2
cuda_implementation,setup_s,h2d_s,kernel_s,d2h_s,device_total_s,host_total_s,kernel_speedup,total_speedup,interactions_per_kernel_s,tensor_bytes,index_bytes,row_metadata_bytes,leaf_metadata_bytes,identity_bytes,scratch_bytes,threads_per_block,persistent_device_bytes,checksum
cuda-canonical-aos,0.01,0.00001,0.0002,0.00001,0.00022,0.00024,1,1,3407360000,100,20,4,0,8,0,256,200,-2
cuda-cusparse-bsr3,0.002,0.00001,0.00005,0.00001,0.00007,0.00009,4,3.142857,13629440000,150,20,4,0,0,0,0,260,-2
"""


def test_default_particle_sweep_uses_every_size_at_every_depth():
    options = SimpleNamespace(
        particles=[2**exponent for exponent in range(12, 18)],
        depths=[2, 3, 4, 5],
        irregular=False,
    )

    cases = build_cases(options)

    assert {case.particles for case in cases} == set(options.particles)
    assert {case.depth for case in cases} == set(options.depths)
    assert len(cases) == 24


def test_uniform_interaction_estimate_matches_dense_leaf_geometry():
    case = SweepCase(depth=3, particles=4096, irregular=False)

    assert estimate_uniform_interactions(case) == 681472
    assert estimate_uniform_interactions(
        SweepCase(depth=3, particles=64, irregular=False)
    ) == 484


def test_parser_and_figures_preserve_cpu_mkl_and_cuda_rows(tmp_path):
    geometry, cpu_rows, cuda_rows = parse_benchmark_output(SAMPLE_OUTPUT)
    case = SweepCase(depth=3, particles=4096, irregular=False)
    cpu_rows = add_case_columns(cpu_rows, case, geometry)
    cuda_rows = add_case_columns(cuda_rows, case, geometry)

    assert {row["implementation"] for row in cpu_rows} == {
        "canonical-aos",
        "onemkl-bsr3",
    }
    assert {row["cuda_implementation"] for row in cuda_rows} == {
        "cuda-canonical-aos",
        "cuda-cusparse-bsr3",
    }

    figures = generate_figures(cpu_rows, cuda_rows, tmp_path)

    assert len(figures) == 4
    assert all(figure.is_file() and figure.stat().st_size > 0 for figure in figures)
