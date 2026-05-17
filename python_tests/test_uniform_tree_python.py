import numpy as np
import pytest

import cdfmm


def test_uniform_tree_node_counts_and_arrays():
    sources = np.array(
        [[-0.9, -0.9, -0.9], [0.9, 0.9, 0.9], [0.1, -0.2, 0.3], [0.0, 0.0, 0.0]],
        dtype=float,
    )

    for level, expected in [(0, 1), (1, 9), (2, 73)]:
        options = cdfmm.UniformTreeOptions()
        options.max_level = level
        tree = cdfmm.UniformTree(sources.tolist(), options)
        assert len(tree.nodes) == expected

    options = cdfmm.UniformTreeOptions()
    options.max_level = 2
    tree = cdfmm.UniformTree(sources.tolist(), options)
    sorted_sources = tree.sorted_source_positions()
    assert sorted_sources.shape == (len(sources), 3)
    assert sorted(tree.source_permutation) == list(range(len(sources)))
    assert len(tree.leaf_indices()) == 8 ** options.max_level
    assert any(node.source_count > 0 for node in tree.nodes)

    node = tree.nodes[-1]
    assert isinstance(node.morton_index, int)
    assert isinstance(node.list1, list)
    assert isinstance(node.list2, list)


def test_uniform_tree_with_targets_and_no_padding_factor_keyword():
    sources = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], dtype=float)
    targets = np.array([[0.2, 0.1, -0.1], [-0.4, 0.8, 0.3]], dtype=float)

    options = cdfmm.UniformTreeOptions()
    options.max_level = 2
    tree = cdfmm.UniformTree(sources.tolist(), targets.tolist(), options)
    assert tree.sorted_target_positions().shape == (len(targets), 3)
    assert sorted(tree.target_permutation) == list(range(len(targets)))

    with pytest.raises(TypeError):
        cdfmm.UniformTree(sources.tolist(), max_level=2, padding_factor=1.0)
