import numpy as np

import cdfmm


def test_geometry_records_and_independent_model_controls_are_bound():
    prism = cdfmm.RectangularPrism(0.2, 0.3, 0.4)
    assert prism.hx == 0.2
    assert prism.volume == 0.024

    tetrahedron = cdfmm.Tetrahedron(
        np.asarray([
            [-0.25, -0.25, -0.25],
            [0.75, -0.25, -0.25],
            [-0.25, 0.75, -0.25],
            [-0.25, -0.25, 0.75],
        ])
    )
    assert np.isclose(tetrahedron.volume, 1.0 / 6.0)

    options = cdfmm.UniformFmmOptions()
    options.source_geometry = cdfmm.SourceGeometry.RECTANGULAR_PRISM
    options.target_geometry = cdfmm.TargetGeometry.TETRAHEDRON
    options.source_sizes = [prism]
    options.target_tetrahedra = [tetrahedron]
    options.near_field_source_model = cdfmm.SourceModel.EXACT_GEOMETRY
    options.near_field_target_model = cdfmm.TargetModel.POINT
    options.far_field_source_model = cdfmm.SourceModel.POINT_DIPOLE
    options.far_field_target_model = cdfmm.TargetModel.EXACT_GEOMETRY

    assert options.near_field_source_model == cdfmm.SourceModel.EXACT_GEOMETRY
    assert options.near_field_target_model == cdfmm.TargetModel.POINT
    assert options.far_field_source_model == cdfmm.SourceModel.POINT_DIPOLE
    assert options.far_field_target_model == cdfmm.TargetModel.EXACT_GEOMETRY
    assert not hasattr(options, "use_" + "cuboid_p2m")
    assert not hasattr(options, "use_" + "cuboid_l2p")
    assert not hasattr(cdfmm, "Cuboid" + "Size")
