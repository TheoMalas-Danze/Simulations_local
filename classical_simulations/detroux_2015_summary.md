# Reference summary — Detroux, Renson, Masset & Kerschen (2015)
"The harmonic balance method for bifurcation analysis of large-scale nonlinear mechanical systems"
*Comput. Methods Appl. Mech. Engrg.* 296 (2015) 18–38

Purpose of this note: capture the formalism and notation of the paper.

---

## 1. The problem class the paper solves

Second-order mechanical system with $n$ DOFs:

$$
M\ddot x + C\dot x + Kx + f_{nl}(x,\dot x) = f_{ext}(\omega,t)
$$

The solution is periodic in time, with (sub)harmonic structure $\omega/\nu$.

## 2. Fourier / harmonic-balance representation (Sec. 2.1)

The solution is truncated to $N_H$ harmonics. The paper orders the unknowns as

$$
z = \big(c_0^x{}^T,\ s_1^x{}^T,\ c_1^x{}^T,\ \dots,\ s_{N_H}^x{}^T,\ c_{N_H}^x{}^T\big)^T \in \mathbb R^{n(2N_H+1)}
$$

The basis vector is $Q(t) = \big(1/\sqrt2,\ \sin(\omega t/\nu),\ \cos(\omega t/\nu),\dots\big)$.

- $x(t) = (Q(t)\otimes I_n)\,z$
- $\dot x(t) = (Q(t)\nabla \otimes I_n)\, z$ and $\ddot x(t) = (Q(t)\nabla^2\otimes I_n)\,z$
- $\nabla$ is block diagonal with rotation generators $\nabla_k = \begin{pmatrix}0&-k\omega/\nu\\k\omega/\nu&0\end{pmatrix}$, acting on $(s_k, c_k)$.

A Galerkin projection onto the trigonometric basis (using the orthogonality of $Q$) removes the time dependence. It gives
the frequency-domain residual:

$$
h(z,\omega) \equiv A(\omega) z - b(z) = 0, \qquad
A = \nabla^2\otimes M + \nabla\otimes C + I_{2N_H+1}\otimes K.
$$

Here $b(z)$ gathers the Fourier coefficients of the external and nonlinear forces, $f_{ext} - f_{nl}$.

## 3. AFT / operator $\Gamma$ (Sec. 2.2)

The paper evaluates the Fourier coefficients of the nonlinear force without an analytic formula. It uses the
**alternating frequency–time (AFT)** scheme, through the collocation operator $\Gamma(\omega)$:

$$
\tilde x = \Gamma(\omega)\, z, \qquad z = \Gamma(\omega)^+ \tilde x, \qquad
b(z) = \Gamma(\omega)^+ \tilde f, \quad \tilde f = f\big(x(t_j), \dot x(t_j), \dots\big)\ \text{(time samples).}
$$

$\Gamma$ has size $nN\times n(2N_H+1)$ and is built from the samples of $Q(t_j)\otimes I_n$ at the $N$ instants $t_j$
of one period. $\Gamma$ is tall with full column rank, so its pseudo-inverse is $\Gamma^+ = (\Gamma^T\Gamma)^{-1}\Gamma^T$.

The Jacobian of the nonlinear map is (Eq. 30):

$$
h_z = \frac{\partial h}{\partial z} = A - \Gamma^+ \frac{\partial \tilde f}{\partial \tilde x}\,\Gamma
$$

Solvers: the paper mentions Newton–Raphson and the Powell hybrid method (MINPACK `hybr`).

## 4. Continuation (Sec. 2.3)

The paper uses pseudo-arclength continuation, parametrized by $\omega$. Each step is a tangent prediction followed by a
Moore–Penrose correction, as in MatCont.

## 5. Hill's method / stability (Sec. 3.1)

Perturb the periodic solution as $p(t) = x^*(t) + e^{\lambda t}s(t)$. Expand $s(t)$ in the same Fourier basis,
with unknown coefficients $u$, and apply Galerkin. This gives **the quadratic eigenvalue problem**

$$
\big(\Delta_2\lambda^2 + \Delta_1\lambda + h_z\big)u = 0,
$$

$$
\Delta_1 = \nabla\otimes 2M + I_{2N_H+1}\otimes C, \qquad \Delta_2 = I_{2N_H+1}\otimes M.
$$

It is recast as a linear generalized eigenproblem of double size:

$$
B_1 = \begin{pmatrix}\Delta_1 & h_z\\ -I & 0\end{pmatrix}, \qquad
B_2 = -\begin{pmatrix}\Delta_2 & 0\\ 0 & I\end{pmatrix}, \qquad
B = B_2^{-1}B_1.
$$

The problem has $(2N_H+1)2n$ eigenvalues, but only $2n$ of them are genuine Floquet exponents: **the $2n$ with the smallest
$|\mathrm{Im}(\lambda)|$** (Moore's criterion). The others are spurious, aliased copies. The solution is stable if and
only if all genuine Floquet exponents have $\mathrm{Re}(\lambda)\le 0$.

## 6. Bifurcation detection & tracking (Sec. 3.2–3.3)

- **Fold:** detected by $\phi_F = |h_z|$ (rank deficiency), or by a sign change of a tangent component.
- **Neimark–Sacker (NS):** detected by the bialternate-product test function $\phi_{NS} = |\tilde B^\odot|$, evaluated on
  the diagonal matrix of genuine Floquet exponents.
- **Determinants:** the bordering technique turns each determinant evaluation into a bordered linear solve.
- **Tracking:** the condition $g=0$ is appended to the HB system, and the bifurcation is continued in a 2-parameter space
  using eigenvalue-derivative formulas (Van der Aa et al.).

## 7. Physical demonstration (Sec. 4)

The method is validated on a 37-DOF satellite finite-element model with a nonlinear vibration isolator. Results are
cross-checked against Newmark time integration and against a Floquet computation from the monodromy matrix. This shows
that Hill's method with modest $N_H$ gives accurate stability information without ever forming or propagating a
monodromy matrix. The paper finds $N_H=5$ sufficient for 1% accuracy.
