# Getting started

## C++ direct evaluation

The umbrella header exposes the public interface.  This example evaluates one
target from two dipoles and requests both supported quantities:

```cpp
#include <iostream>
#include <vector>

#include "cdfmm/cdfmm.hpp"

int main()
{
    const cdfmm::Vec3 target_position{0.0, 0.0, 2.0};
    const std::vector<cdfmm::Vec3> source_positions{
        {0.0, 0.0, 0.0},
        {0.25, 0.0, 0.0}
    };
    const std::vector<cdfmm::Vec3> dipole_moments{
        {0.0, 0.0, 1.0},
        {1.0, 0.0, 0.0}
    };

    const cdfmm::PotentialField result = cdfmm::p2p_dipole_sum(
        target_position,
        source_positions,
        dipole_moments,
        cdfmm::OutputFlags::Both
    );

    std::cout << "phi = " << result.phi << "\n";
    std::cout << "H = (" << result.H.x << ", " << result.H.y << ", "
              << result.H.z << ")\n";
}
```

When targets coincide with sources, pass the corresponding `self_index` to
`p2p_dipole_sum`; otherwise the singular self-pair is evaluated.

## Python direct evaluation

```python
import numpy as np
import cdfmm

sources = np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]])
moments = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])

result = cdfmm.p2p_dipole_sum(
    [0.0, 0.0, 2.0], sources, moments, output="both"
)
print(result["phi"], result["H"])
```

The Python interface also exposes `UniformTree` for geometry inspection.  It
does not yet expose the Cartesian expansion operators or a complete FMM solve.
