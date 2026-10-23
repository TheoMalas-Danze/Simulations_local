# Why the instability threshold rises with κ_b (Fig. 7)

Analysis of 19–20/10/2026, answering report v2 ("Fig. 7 slope, \TMD{Explanation?}").
Code and outputs: cells 43–52 of
`classical_simulations/studies/hill_orbits/Hill_method_doubling_corrected.ipynb`.

**Setup.**
- **Orbit and threshold:** period-2 satellite at ε_d/g₂ = 10.12 (ε_d/2g₂ = 5.06). The threshold g₂* is the smallest g₂ with max |μ| > 1 (Hill's method, N_H = 15).
- **κ_b convention:** GKSL, so κ_b^GKSL = 2 × `kappa_b` of the notebook, and κ′_b = κ_b/2ω_b. The sweep covers κ_b/2π = 2–500 MHz.

## What sets the slope

**Quadratic law.** g₂*/2π ≈ 75.8 + 7.9×10⁻⁵ (κ_b/2π)² MHz, with κ_b/2π in MHz. This fits all 30 points within 0.4 MHz, and the local exponent is 2.0–2.2 over the whole range. The "linear at large κ_b" in the article comes from fitting only the last 6 points, which gives the tangent of a parabola.

**Mechanism.** With κ_b → 0, the period-2 orbit becomes unstable at about 76.2 MHz through a quartet of multipliers μ, μ\*, 1/μ, 1/μ\* leaving the unit circle. This is a Hamiltonian–Hopf (Krein) collision, the lossless counterpart of the Neimark–Sacker bifurcation. Its growth rate follows σ_H² ≈ 0.19 ns⁻²/MHz × (g₂ − g₂,₀).

**Effect of loss.** With damping, the orbit stays stable until σ_H reaches about 0.47–0.62 κ_b, which is close to the buffer's amplitude decay rate κ_b/2. Combined with the square-root growth, this gives

  g₂* − g₂,₀ ≈ w² κ_b² / a,  with w = σ_H(g₂*)/κ_b and σ_H² = a (g₂ − g₂,₀),

so Δg₂ ∝ κ_b². In the plot of σ_H against g₂ (cell 49), the points κ_b/2 at g₂*(κ_b) lie on the σ_H(g₂) curve at small κ_b and drift about 20% below it at large κ_b.

## Related results (same notebook)

- **Existence hypothesis rejected.** The satellite existence line 2κ′_b|y| = ε′_d is not the threshold.
  - Along the threshold, r = 2κ′_b|y|/ε′_d rises from 0.005 to 0.78 instead of staying at 1.
  - The existence line lies 12–30 MHz below the threshold and scales as g₂ ∝ √κ_b.
- **Multipliers in the complex plane (cell 51).**
  - With loss, the Hamiltonian–Hopf collision is unfolded into a Neimark–Sacker bifurcation: one pair crosses |μ| = 1 at arg μ ≈ ±(0.20–0.24)π.
  - The crossing is not on the real axis. On the real axis there is only the other pair, for κ_b/2π ≳ 300 MHz, which stays inside the unit circle.

## Suggested text for Sec. "Increasing the single photon loss"

> The instability threshold increases quadratically with κ_b. Without loss, the period-2 satellite loses stability through a Krein collision of its Floquet multipliers, with a growth rate σ ∝ √(g₂ − g₂,₀). The buffer loss damps this growth at a rate of order κ_b/2, so that the threshold moves by Δg₂ ∝ κ_b².

# Hamiltonian–Hopf (Krein) collision and its link to Neimark–Sacker

> "Mechanism: with κ_b → 0, the period-2 orbit becomes unstable at about 76.2 MHz through a quartet of multipliers μ, μ*, 1/μ, 1/μ* leaving the unit circle. This is a Hamiltonian–Hopf (Krein) collision, the lossless counterpart of the Neimark–Sacker bifurcation. Its growth rate follows σ_H² ≈ 0.19 ns⁻² per MHz × (g₂ − g₂,₀)."

The paragraph says: **switch off the dissipation and see what's left of the instability.** Without loss, the dynamics is Hamiltonian, and Hamiltonian systems can't have a Neimark–Sacker bifurcation in the usual sense. Their analogue is the Hamiltonian–Hopf bifurcation. The Neimark–Sacker you see with loss is that Hamiltonian instability, partially "paid for" by the damping.

## 1. Floquet multipliers of a Hamiltonian orbit come in quartets

For κ_b = 0, the monodromy matrix M (the linearized flow over one period of the period‑2 orbit) is symplectic, $M^T J M = J$. Its spectrum is then invariant under

$$\mu \to \mu^*, \qquad \mu \to 1/\mu, \qquad \mu \to 1/\mu^* .$$

The first symmetry holds because M is real. The second is Liouville/symplecticity: volumes are preserved, so for every expanding direction there is a contracting one.

A stable (elliptic) orbit has all multipliers on the unit circle, $\mu = e^{\pm i\theta}$. For such a μ, the partner 1/μ* is μ itself, so the pair is "self‑sufficient." A single pair therefore cannot leave the unit circle on its own: if μ moved outward, its partner 1/μ* would have to move inward at the same point, and a pair can't split like that continuously. There are only two ways out:

- **Through ±1.** At −1, the pair $e^{\pm i\theta}$ meets itself, then splits along the real axis into (μ, 1/μ) with μ < −1. This is the Hamiltonian period‑doubling, the counterpart of the first bifurcation. At +1 it is the tangent/pitchfork case.
- **Two different pairs collide on the unit circle** at some $e^{i\theta_0} \neq \pm 1$ and then move off as a quartet μ, μ*, 1/μ, 1/μ*, with one complex pair outside the circle and one inside. This is the **Hamiltonian–Hopf bifurcation**.

## 2. Krein signature: why some collisions destabilize and others don't

Not every collision of two pairs on the unit circle causes an instability. Each multiplier on the unit circle with eigenvector v carries a **Krein signature**, the sign of the Hermitian form

$$\kappa(v) = \operatorname{sign}\big( i\, v^\dagger J v \big) \quad(\text{up to convention}).$$

The Krein–Gelfand–Lidskii theorem says:

- If two colliding eigenvalues have the **same** signature, they pass through each other and stay on the circle. The collision is robust.
- If they have **opposite** signatures, they can split off the circle. This is the instability.

That is why the paragraph calls it a "Krein collision."

There is a translation that should feel natural from superconducting circuits. Linearize around the orbit and quantize the fluctuations. The Krein signature is then the sign of the **Bogoliubov norm** $|u|^2 - |v|^2$, or equivalently whether the mode carries positive or negative energy in the rotating frame. A collision of opposite signatures is a positive‑energy mode becoming resonant with a negative‑energy mode. The coupling between them acts as a **two‑mode squeezing (parametric amplification) term**, $a b + a^\dagger b^\dagger$, rather than a beam‑splitter term. The fluctuations then grow exponentially while the total "energy" stays conserved. Same‑signature modes only hybridize, like a beam splitter (avoided crossing), and stay stable.

So the lossless period‑2 orbit becomes unstable because, at g₂ ≈ g₂,₀ (≈ 76.2 MHz), two of its fluctuation modes with opposite Krein signature come into resonance and start parametrically amplifying each other. The growth has a nonzero oscillation frequency, because θ₀ ≠ 0 or π.

## 3. Why σ_H² ∝ (g₂ − g₂,₀): the square‑root law

Write the Floquet exponents as λ = ln μ / T, where T is the period of the period‑2 orbit. Near the collision, the 2×2 reduced problem looks like the two‑mode model above. The two frequencies are detuned by δ(g₂), which crosses zero at the collision, and the eigenvalues go like

$$\lambda \simeq i\Omega \pm \sqrt{|c|^2 - \delta^2/4}\quad\text{(schematically)}.$$

In the generic unfolding, the quantity under the square root changes sign linearly in the parameter. So past the threshold,

$$\lambda = \pm\sigma_H \pm i\Omega, \qquad \sigma_H^2 \simeq a\,(g_2 - g_{2,0}),$$

with a ≈ 0.19 ns⁻²/MHz here. For example, 1 MHz past threshold gives σ_H ≈ √0.19 ≈ 0.44 ns⁻¹.

This is the key difference from a dissipative Hopf or Neimark–Sacker bifurcation, where the real part grows **linearly** in the distance to threshold. In the Hamiltonian case it grows like a **square root**, because instability comes from two eigenvalues merging (a branch point), not from one eigenvalue crossing.

## 4. Connecting back to Neimark–Sacker with loss

With κ_b > 0, M is no longer symplectic, and $\det M = \exp(-\oint \nabla\!\cdot\! f\, dt) < 1$. To leading order in the damping, the whole Hamiltonian spectrum is contracted radially. Each multiplier becomes roughly $\mu\, e^{-\gamma T}$, where γ is the damping rate felt by that Floquet mode. If only mode b is damped, γ is roughly κ_b/2 weighted by the mode's participation in b.

Consequences:

- Before the collision, all multipliers sit slightly **inside** the unit circle, so the orbit is stable.
- After the collision, the quartet has radii $e^{(\pm\sigma_H - \gamma)T}$. The inner pair just goes deeper inside. The outer pair only crosses the unit circle once **σ_H > γ**.
- That crossing is a single complex‑conjugate pair leaving the unit circle at angle ±ΩT. That is exactly the definition of a **Neimark–Sacker bifurcation**. It creates a torus with a new incommensurate frequency Ω.

Combined with the square‑root law, this predicts where the dissipative threshold sits:

$$g_{2,\text{NS}} \simeq g_{2,0} + \frac{\gamma^2}{a}.$$

The Neimark–Sacker threshold therefore moves away from 76.2 MHz **quadratically** in the loss. This is presumably why the κ_b → 0 limit and the coefficient a were extracted: together they let you predict the NS point for any small κ_b.

## A practical check

At the NS point found with loss, compute the Krein signatures of the two unit‑circle pairs just before g₂,₀ in the lossless system. They should be opposite. Then check that ln|μ_NS| / T, plotted against √(g₂ − g₂,₀) as κ_b → 0, falls on a straight line of slope √a.