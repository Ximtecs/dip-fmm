import matplotlib.pyplot as plt
import numpy as np

import cdfmm


def main():
    rng = np.random.default_rng(1234)
    sources = rng.uniform(-1.0, 1.0, size=(32, 3))

    options = cdfmm.UniformTreeOptions()
    options.max_level = 2
    tree = cdfmm.UniformTree(sources.tolist(), options)

    sorted_sources = tree.sorted_source_positions()

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(sorted_sources[:, 0], sorted_sources[:, 1], sorted_sources[:, 2], c="tab:blue", s=24)

    for leaf_index in tree.leaf_indices():
        node = tree.nodes[leaf_index]
        if node.source_count == 0:
            continue
        ax.text(node.centre.x, node.centre.y, node.centre.z, str(node.morton_index), fontsize=7)

    ax.set_title("Uniform Morton-sorted tree leaf centres")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
