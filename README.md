# Koshino's effective model of Magic-Angle Twisted Bilayer Graphene

A reproduction of Koshino _et al_. (2018) model and hopping integrals (eff_hopping_ver2.dat), both available as an open acess [PRX article](doi.org/10.1103/PhysRevX.8.031087).

This model describes the MATBG low-energy band structure through a list of hopping integrals $t\left( r_{ij} \right)$ between maximally localized Wannier orbitals. These orbitals have a three-peak form at AA spots and are centered at nonequivalent BA (orbital 1) and AB (orbital 2) spots in the emergent moiré pattern, which creates the alternating honeycomb lattice that is used to calculate the band structure.

Orbitals for valley $\xi = +$ are the complex conjugate of $\xi = -$. Due to symmetries under rotation, the orbitals are $p_\xi \equiv p_x + i\xi p_y$ $\left( \xi = \pm \right)$. The effective model is then constructed by considering hoppings between $\left( p_x, p_y \right)$ orbitals.

Although it is not specified in the original article, columns in the numerical table eff_hopping_ver2.dat mean, respectively: indexes (m,n) of the real-space lattice vector of TBG, initial and final orbitals (i,j) of a hopping, real and imaginary components of a $\xi = +$ valley hopping integral $t$. Do not mistake these (m,n) vector indexes with the (m,n) used to calculate commensurate angles, particularly here 1.05° (m=32,n=31).
