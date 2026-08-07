# Laplace derivative generation

M2L and M2P require raw derivatives $D_\alpha G(R)$ at orders beyond those
stored in the source expansion.  `laplace_derivatives_raw` obtains them through
truncated Cartesian Taylor algebra rather than finite differences.

A `TaylorJet` stores

$$f(r_0+h)=\sum_{|\alpha|\le p}c_\alpha h^\alpha,
\qquad c_\alpha=\frac{D_\alpha f(r_0)}{\alpha!}.$$

Coordinate jets represent $x=r_x+h_x$, $y=r_y+h_y$, and $z=r_z+h_z$.
The implementation composes

$$\rho^2=x^2+y^2+z^2,\qquad G=\frac{1}{4\pi}(\rho^2)^{-1/2}.$$

Products use the multi-index Cauchy product.  The inverse square root is solved
coefficient by coefficient from $y^2\rho^2=1$ in increasing total degree, so
each unknown depends only on coefficients already found.  Finally the jet
coefficient is multiplied by $\alpha!$ to return the raw derivative expected
by the operators.

This approach is analytic/algebraic to the requested truncation order.  It
avoids finite-difference step-size selection and cancellation noise, while also
avoiding a hand-maintained table of high-order Cartesian derivatives.  The
evaluation point must be away from the kernel singularity at $R=0$.
