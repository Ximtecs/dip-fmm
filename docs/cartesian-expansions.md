# Cartesian expansions and coefficient storage

For a multi-index $\alpha=(\alpha_x,\alpha_y,\alpha_z)$, define
$|\alpha|=\alpha_x+\alpha_y+\alpha_z$, $\alpha!=\alpha_x!\alpha_y!\alpha_z!$,
and $r^\alpha=r_x^{\alpha_x}r_y^{\alpha_y}r_z^{\alpha_z}$.

`MultiIndexSet(p)` contains every non-negative $\alpha$ with $|\alpha|\le p$.
Its size is

$$N_p=\binom{p+3}{3}=\frac{(p+1)(p+2)(p+3)}{6}.$$

Entries are grouped by total degree.  At fixed degree the implementation loops
over $\alpha_x$ and then $\alpha_y$ in ascending order; $\alpha_z$ is the
remaining degree.  `index(alpha)` maps a mathematical index to linear storage,
and `operator[](i)` performs the reverse lookup.  `CoeffVector` uses precisely
this basis order for both multipole and local coefficients.

Factorial-normalised monomials $r^\alpha/\alpha!$ make Taylor translations and
jet products free of explicit multinomial factors.  Multipole coefficients are
paired with **raw** derivatives $D_\alpha G$, whereas local coefficients are
raw potential derivatives paired with these normalised monomials.

For dipoles, $M_{(0,0,0)}=0$: P2M requires a term
$\alpha-e_k$ for a dipole component $m_k$, and no coordinate direction is
valid when $\alpha=0$.  This expresses the absence of net monopole charge; it
does not mean local expansions omit their degree-zero potential coefficient.
