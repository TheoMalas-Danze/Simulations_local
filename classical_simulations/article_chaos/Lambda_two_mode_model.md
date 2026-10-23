# Floquet exponent near the Neimark–Sacker point: two-mode model

Written 20/10/2026, for report v2, point 10.
The numbers come from `classical_simulations/poster_chicago_2026/floquet_rescaling_kappa_b.ipynb`.

**Goal.** Derive

$$\Lambda = -\frac{\kappa_b}{4} + \mathrm{Re}\sqrt{a\,(g_2 - g_{2,0}) + iG\kappa_b + e\kappa_b^2}\,,$$

the Floquet exponent of the pair of multipliers that destabilises the period-2 satellite. Then explain the factor $i$, the threshold law, the finite offset at $\kappa_b\to0$, and the rescaling.

**Conventions.**
- $\kappa_b$ is the GKSL rate of the article. The buffer amplitude decays as $\dot b = -(\kappa_b/2)b$, and in the flux equations $\ddot\phi_b = \dots - \kappa_b\dot\phi_b$. In the code, the damping term is $-2\kappa\dot\phi_b$ with $\kappa = \kappa_b/2$.
- The orbit is the period-2 satellite at $\varepsilon_d = 10.12\,g_2$, with period $T = 4\pi/\omega_a$.
- Floquet exponents $\lambda$ are in ns⁻¹, so the multipliers are $\mu = e^{\lambda T}$, and $\Lambda = \ln|\mu_{NS}|/T = \mathrm{Re}\,\lambda_{NS}$.

---

## 1. Setting

Linearise the equations of motion around the period-2 orbit. The perturbation $\delta\mathbf y = (\delta x_a, \delta x_b, \delta\dot x_a, \delta\dot x_b)$ obeys $\delta\dot{\mathbf y} = J(t)\,\delta\mathbf y$, with $J$ periodic of period $T$. There are four Floquet exponents.

Near the instability, the four exponents form two complex pairs, $\lambda_1, \lambda_1^*$ and $\lambda_2, \lambda_2^*$. Their angles are $\arg\mu \approx \pm 0.2\pi$, far from the real axis.

The instability is a Neimark–Sacker bifurcation: one complex pair crosses $|\mu| = 1$. For $\kappa_b\to0$, it becomes a Hamiltonian–Hopf (Krein) collision of the two pairs on the unit circle.

## 2. Reduction to a 2×2 problem

Keep one member of each pair, the one with $\mathrm{Im}\,\lambda > 0$. Floquet theory writes the dynamics restricted to these two modes as $\dot{\mathbf u} = R\,\mathbf u$, with a constant $2\times2$ complex matrix $R$. Its eigenvalues are $\lambda_{1,2}$:

$$\lambda_{1,2} = \frac{\mathrm{tr}\,R}{2} \pm \sqrt{D},\qquad D = \frac{(R_{11} - R_{22})^2}{4} + R_{12}R_{21} = \left(\frac{\lambda_1 - \lambda_2}{2}\right)^2.$$

The other two exponents are the complex conjugates, so this reduction contains **all four** exponents.

Two remarks:
- $D$ is a symmetric function of $\lambda_1$ and $\lambda_2$, so it is **analytic** in $(g_2, \kappa_b)$. This holds even at the collision, where $\lambda_{1,2}$ themselves have a square-root branch point. All the non-analytic behaviour of $\Lambda$ comes from the square root.
- $D$ can be computed directly from the Hill exponents, with no model: $D = ((\lambda_1 - \lambda_2)/2)^2$.

The unstable exponent is therefore

$$\Lambda = \mathrm{Re}\,\frac{\mathrm{tr}\,R}{2} + \mathrm{Re}\sqrt D.$$

## 3. The mean damping, $-\kappa_b/4$ (Liouville)

The trace of $J$ is the divergence of the flow, $\mathrm{tr}\,J = -\kappa_b$, which is constant. By Liouville's formula, the four exponents add up to

$$\lambda_1 + \lambda_1^* + \lambda_2 + \lambda_2^* = 2\,\mathrm{Re}(\lambda_1 + \lambda_2) = \frac{1}{T}\int_0^T \mathrm{tr}\,J\,dt = -\kappa_b .$$

Hence

$$\mathrm{Re}\,\frac{\mathrm{tr}\,R}{2} = -\frac{\kappa_b}{4}\quad\text{exactly}.$$

Equivalently, the moduli of the two pairs satisfy $|\mu_1||\mu_2| = e^{-\kappa_b T/2}$. The numerical check holds to $10^{-4}$ ns⁻¹.

What remains is to describe $D$.

## 4. The lossless case: Krein signature

Without loss, the linearised flow is symplectic. Restricted to two modes, the generator has the form

$$R_0 = i\,\eta H,\qquad H = H^\dagger,\qquad \eta = \mathrm{diag}(s_1, s_2),$$

where $s_j = \pm1$ is the Krein signature of mode $j$: the sign of its energy, or of its Bogoliubov norm $|u|^2 - |v|^2$.

**Same signatures** ($\eta = \mathbb 1$). $R_0 = iH$ is anti-Hermitian, so the exponents are purely imaginary for any $H$. The two modes can only hybridise, as with a beam-splitter coupling: this is an avoided crossing, and there is no instability.

**Opposite signatures** ($\eta = \mathrm{diag}(1, -1)$). Take $H = \begin{pmatrix}\alpha & \beta\\ \beta^* & \delta\end{pmatrix}$. Then

$$R_0 = i\begin{pmatrix}\alpha & \beta\\ -\beta^* & -\delta\end{pmatrix},\qquad \lambda = i\,\frac{\alpha - \delta}{2} \pm\sqrt{D_0},\qquad D_0 = |\beta|^2 - \frac{\Delta^2}{4},\quad \Delta = \alpha + \delta .$$

Here $\Delta$ is the detuning between the two modes, whose frequencies are $\alpha$ and $-\delta$. **$D_0$ is real**:
- For $D_0 < 0$, the exponents stay on the imaginary axis, with a frequency splitting $2\sqrt{|D_0|}$. The orbit is stable.
- For $D_0 > 0$, the exponents get real parts $\pm\sqrt{D_0}$. This is the Hamiltonian–Hopf instability, a two-mode-squeezing coupling between a positive-energy and a negative-energy mode.

$D_0$ changes sign linearly at the collision point $g_{2,0}$:

$$D_0 \simeq a\,(g_2 - g_{2,0}),\qquad \text{i.e.}\quad \sigma_H^2 = a\,(g_2 - g_{2,0})\ \text{above the collision}.$$

This square-root law is the signature of an eigenvalue collision. A dissipative Hopf bifurcation is different: there, the real part grows linearly.

*Numbers:* $g_{2,0}/2\pi = 76.302$ MHz and $a = 0.210$ ns⁻²/MHz. Here $a$ is the slope of $\sigma_H^2$ at the collision, and $g_2$ is in MHz (i.e. $g_2/2\pi$).

## 5. Adding the loss: why $G$ comes with a factor $i$

Add the buffer damping, projected on the two modes. It is a real matrix:

$$R = i\eta H - \kappa_b\Gamma,\qquad \Gamma = \begin{pmatrix}\gamma_1 & \gamma_{12}\\ \gamma_{21} & \gamma_2\end{pmatrix}.$$

Every entry of $R$ has the form

$$i\times(\text{conservative, real}) + \kappa_b\times(\text{dissipative, real}).$$

$D$ is quadratic in the entries. Sorting its terms by the number of factors of $i$:

| term of $D$ | origin | type |
|---|---|---|
| $\kappa_b^0$ | conservative × conservative, $i\cdot i = -1$ | real: $D_0 = a(g_2 - g_{2,0})$ |
| $\kappa_b^1$ | conservative × dissipative, $i\cdot 1$ | **imaginary**: $iG\kappa_b$ |
| $\kappa_b^2$ | dissipative × dissipative | real: $e\kappa_b^2$ |

Explicitly, with $\beta$ chosen real by a phase convention:

$$D = \underbrace{\beta^2 - \frac{\Delta^2}{4}}_{D_0}\; -\; i\kappa_b\underbrace{\left[\frac{\Delta\,(\gamma_1 - \gamma_2)}{2} + \beta(\gamma_{21} - \gamma_{12})\right]}_{-G}\; +\; \kappa_b^2\left[\frac{(\gamma_1 - \gamma_2)^2}{4} + \gamma_{12}\gamma_{21}\right].$$

At the collision, $\Delta \approx 2\beta$, so $G \approx -\beta\,[(\gamma_1 - \gamma_2) + (\gamma_{21} - \gamma_{12})]$.

**$G$ measures how differently the two colliding modes are damped.** Only the buffer is lossy, and the two modes have different weights on it, so $G\neq0$. If the loss acted identically on both modes ($\Gamma\propto\mathbb 1$), $G$ would vanish.

A real term of order $\kappa_b$ would shift the collision point linearly in $\kappa_b$. None is seen: the threshold shift is purely quadratic.

### The general reason: time reversal

The counting of $i$'s is the first-order version of an exact symmetry. The equations

$$M\ddot{\mathbf X} + C\dot{\mathbf X} + K\mathbf X + \mathbf f_{nl}(\mathbf X) = \mathbf F\cos 4\tau$$

are mapped onto themselves with $C\to-C$ by $\mathbf Y(\tau) = \mathbf X(-\tau)$, because the drive $\cos 4\tau$ is even.

So the orbit at $-\kappa_b$ is the time-reversed orbit at $\kappa_b$. Its monodromy matrix is the inverse, up to a similarity, so the exponents change sign: $\lambda(-\kappa_b) = -\lambda(\kappa_b)$. The upper-half-plane members at $-\kappa_b$ are therefore $-\lambda_j^*(\kappa_b)$, and

$$D(g_2, -\kappa_b) = D(g_2, \kappa_b)^* .$$

Write $D = \sum_{m,n} d_{mn}\,(g_2 - g_{2,0})^m\kappa_b^n$. Then $d_{mn}$ is **real for even $n$ and imaginary for odd $n$, to all orders**.

This includes the deformation of the orbit with $\kappa_b$ and the $\kappa_b$-dependence of $H$ and $\Gamma$. These change the values of $G$ and $e$, but not the structure.

*Check:* at $\kappa_b = 0$, $|\mathrm{Im}\,D| < 10^{-13}$ ns⁻². In fits of $D(g_{2,0}, \kappa_b)$, the forbidden coefficients go to zero as the degree increases.

## 6. The formula

Combining Secs. 3–5, to second order in $\kappa_b$ and first order in $g_2 - g_{2,0}$:

$$\boxed{\Lambda = -\frac{\kappa_b}{4} + \mathrm{Re}\sqrt{a\,(g_2 - g_{2,0}) + iG\kappa_b + e\kappa_b^2}}$$

- The $-\kappa_b/4$ is exact (Liouville).
- The form of the root (real, imaginary, real) is exact (time reversal).
- The only approximation is the truncation of $D$.

## 7. Threshold: the $\kappa_b^2$ law and the finite offset

Use $\mathrm{Re}\sqrt{u + iv} = \sqrt{(|u + iv| + u)/2}$ with $u = a(g_2 - g_{2,0}) + e\kappa_b^2$, $v = G\kappa_b$ and $r = \kappa_b/4$. The condition $\Lambda = 0$ reads $\mathrm{Re}\sqrt{u + iv} = r$:

$$|u + iv| = 2r^2 - u \;\Rightarrow\; u^2 + v^2 = 4r^4 - 4r^2u + u^2 \;\Rightarrow\; u = r^2 - \frac{v^2}{4r^2}.$$

With $r = \kappa_b/4$, $v^2/4r^2 = 4G^2$, so

$$\boxed{a\,(g_2^* - g_{2,0}) = -4G^2 + \left(\frac{1}{16} - e\right)\kappa_b^2}$$

This gives:
- **A $\kappa_b^2$ law** for the threshold, as observed: the local exponent is 2.0–2.2 over 2–500 MHz.
- **A finite offset as $\kappa_b\to0$**, $g_2^*(0^+) - g_{2,0} = -4G^2/a$, independent of $\kappa_b$. Both the damping $r$ and the imaginary term $v$ are proportional to $\kappa_b$, so their ratio survives the limit.
  - An arbitrarily small loss destabilises the orbit slightly *before* the conservative Krein collision. At $\kappa_b = 0$ exactly the window closes, so the limit is singular.
  - This is the dissipation-induced instability of Hamiltonian–Hopf points: Ziegler's paradox (Ziegler 1952, Bottema 1956), reviewed in Kirillov (2013).
  - Physically: just below the collision the two modes are nearly degenerate ($|D_0|$ small). The unequal damping mixes them, and the square root of a nearly imaginary number has a real part that beats the mean damping.

### Exact limit $\kappa_b\to0$

To first order in $\kappa_b$, keep the full functions $D_0(g_2)$ and $G(g_2) = \mathrm{Im}\,\partial_{\kappa_b}D$. With $D_0 < 0$:

$$\mathrm{Re}\sqrt{D_0 + iG\kappa_b} \simeq \frac{G\kappa_b}{2\sqrt{|D_0|}} = \frac{\kappa_b}{4}\quad\Longrightarrow\quad D_0(g_2^*) + 4G(g_2^*)^2 = 0 .$$

This uses only the lossless spectrum and its first-order change with loss, and involves no fit. It predicts $g_2^*(0^+) - g_{2,0} = -0.1546$ MHz. The Hill bisection gives $-0.1545$ MHz.

## 8. Rescaling

Divide by $\kappa_b$ and set $x = a(g_2 - g_{2,0})/\kappa_b^2$ and $\gamma = G/\kappa_b$:

$$\frac{\Lambda}{\kappa_b} = -\frac14 + \mathrm{Re}\sqrt{x + e + i\gamma}.$$

The curves collapse when $\gamma \ll 1$, onto the master curve

$$\frac{\Lambda}{\kappa_b} = \sqrt{x + e} - \frac14 \xrightarrow{\;x\gg1\;} \sqrt x - \frac14 .$$

The shift $-e$ in $x$ is the real $\kappa_b^2$ term of $D$.

**Better variable.** Measure $g_2$ from $g_2^*(0^+)$: $x' = a(g_2 - g_2^*(0^+))/\kappa_b^2$. Since $a(g_2^*(0^+) - g_{2,0}) = -4G^2$,

$$\frac{\Lambda}{\kappa_b} = -\frac14 + \mathrm{Re}\sqrt{x' + e - 4\gamma^2 + i\gamma},$$

and the threshold condition of Sec. 7 gives $x'^* = 1/16 - e$ **for every** $\kappa_b$. In this variable the zero crossings collapse exactly (within the model). Only the shape just above them still depends on $\gamma$.

## 9. Numbers (ε_d = 10.12 g₂, N_H = 15)

| quantity | value | source |
|---|---|---|
| $g_{2,0}/2\pi$ | 76.302 MHz | lossless bisection |
| $a$ | 0.210 ns⁻²/MHz | slope of $\sigma_H^2$ at the collision |
| $g_2^*(0^+)/2\pi$ | 76.147 MHz | thresholds at $\kappa_b/2\pi$ = 0.25–48 MHz (N_H = 15 and 20 agree to 10⁻³ MHz) |
| $g_2^* - g_2^*(0^+)$ | 1.54 MHz × $(\kappa_b\,[\mathrm{ns}^{-1}])^2$ | fit on $\kappa_b/2\pi \le 48$ MHz |
| $G$ | 0.090 ns⁻² (thresholds), 0.096 ns⁻² (direct, at $g_{2,0}$) | |
| $e$ | −0.26 (thresholds), −0.22 to −0.33 (direct, depends on the fit degree) | |
| $x^*$ in the collapse | 0.32–0.40 for $\kappa_b/2\pi$ = 48–384 MHz | model: $1/16 - e = 0.32$ |
| $\arg\mu_{NS}$ at the threshold | ≈ 0.21π for every $\kappa_b$ | |

The thresholds give a smaller $G$ than the direct value because $G$ depends on $g_2$ (0.0945 at $g_2^*(0^+)$) and $D_0$ is not linear. The order-2 fit on the thresholds absorbs both into an effective $G$.

## 10. Validity and higher orders

$\Lambda = -\kappa_b/4 + \mathrm{Re}\sqrt D$ is exact, so the approximation lies entirely in the truncation of $D$.

The expansion of $D$ around $(g_{2,0}, 0)$ converges only up to the nearest point where $D$ is not analytic. That point is where one of the two pairs reaches the real axis, because the reduction to two upper-half-plane exponents then no longer exists.
- At $\kappa_b = 0$ this is the period doubling where the period-2 orbit is born (multiplier $+1$), 1.55 MHz below $g_{2,0}$.
- With loss, this line moves up in $g_2$, just below the threshold.

Consequences, from truncations of order $N = 2$–6 (weights: 2 for $g_2 - g_{2,0}$, 1 for $\kappa_b$):
- For $\kappa_b/2\pi \lesssim 100$ MHz, the threshold error falls from about $3\times10^{-2}$ to $2\times10^{-3}$ MHz, slowly and not monotonically.
- At 197 and 384 MHz, higher orders make it worse. These thresholds lie 2.4 and 11 MHz above $g_{2,0}$, outside the domain of convergence.

To describe large $\kappa_b$, one would expand around a point closer to the threshold of interest rather than add orders around $g_{2,0}$.

## References

- H. Ziegler, *Die Stabilitätskriterien der Elastomechanik*, Ing.-Arch. 20, 49 (1952): the destabilising effect of small damping.
- O. Bottema, *The Routh–Hurwitz condition for the biquadratic equation*, Indag. Math. 18, 403 (1956): the singular stability boundary (Whitney umbrella).
- R. S. MacKay, *Movement of eigenvalues of Hamiltonian equilibria under non-Hamiltonian perturbation*, Phys. Lett. A 155, 266 (1991).
- O. N. Kirillov, *Nonconservative Stability Problems of Modern Physics*, De Gruyter (2013): dissipation-induced instabilities and Krein collisions.
