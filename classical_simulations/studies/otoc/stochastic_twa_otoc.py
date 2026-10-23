"""Classical counterpart of the dissipative quantum OTOC: truncated Wigner with the vacuum noise of the buffer loss.

The quantum OTOC of OTOCS_poster_init_coherent evolves x_a under the adjoint Lindbladian, x_a(t) = Phi_t^dag(x_a). Its
classical counterpart is the noise-averaged observable X(z0, t) = E_xi[x_a^rot(t) | z0], so
    C_cl(t) = < (d X(z0, t) / d x_a(0))^2 >_{z0 ~ W},
with the derivative of the noise AVERAGE, squared, then averaged over the Wigner function W of |a0>|b0>. The
noise-free TWA of otoc_chaotic_start_points.py (mean of the squared pathwise derivative) keeps growing; this one
saturates and decays, as the quantum OTOC does.

Noise: the map EOM (equation_1ODE!, damping -2 kappa_code phi_b' = -kappa_b phi_b') plus the force sqrt(kappa_b) w_b dW
on phi_b'', which relaxes the damped buffer to the vacuum (<phi_b^2> = 2 <(Re b)^2> = 1/2; checked: 0.49).

Derivative: central difference (x(x0 + h) - x(x0 - h)) / 2h along the same noise path (common random numbers), with
x0 = x_a(0) = phi_a(0). The pathwise tangent derivative cannot be used: it grows as e^{lambda t} on every path, so its
noise average is a cancellation of exponentially large terms that no affordable number of paths resolves. The finite
difference stays bounded (|x_a| is), and h of a fraction of the vacuum width (std of x_a: 1/sqrt(2)) is the resolution
of the quantum OTOC anyway; H_LIST checks the dependence on h.

Estimator: for each z0, the mean D and the variance s^2 of the difference quotient over N_XI noise paths;
D^2 - s^2 / N_XI is an unbiased estimate of (E_xi[...])^2. The biased D^2 is stored too.

Integrator: RK4 for the drift, plus the noise increment at the end of each step (strong order 1 for additive noise),
dt = tau / STEPS_PER_PERIOD, vectorised over 2 N_Z0 N_XI trajectories (log C matches DOP853 to 1e-3 without noise).

Run: python3 stochastic_twa_otoc.py   (writes otoc_stochastic_twa_22_10_2026.json)
"""
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from otoc_chaotic_start_points import (E_J, KAPPA_B, KAPPA_CODE, N_PERIODS_Q, N_SUB, OUT_DIR, PHI_QUANTUM, TAU, W_A, W_B,
                                       Z_A, Z_B, g2_of)

N_Z0 = 32               # Wigner samples of the initial state
N_XI = 512              # noise paths per Wigner sample
H_LIST = (0.05, 0.15, 0.4)
STEPS_PER_PERIOD = 200  # dt = tau / 200 ~ 6e-4 ns, w_b dt = 0.03


def make_drift(eps_p, ratio):
    """equation_1ODE!, vectorised: u has shape (4, n)."""
    eps_d = ratio * g2_of(eps_p)
    pump = 2 * E_J * math.sin(eps_p)
    k_a, k_b = math.sqrt(Z_A) / PHI_QUANTUM, math.sqrt(Z_B) / PHI_QUANTUM
    sa, sb = math.sqrt(Z_A) * pump, math.sqrt(Z_B) * pump

    def drift(t, u):
        pa, pb, va, vb = u
        coupling = (np.cos(k_a * pa + k_b * pb) - 1) / PHI_QUANTUM
        return np.stack([va, vb,
                         -W_A * (W_A * pa - sa * coupling),
                         -W_B * (W_B * pb + eps_d * math.cos(W_B * t) * math.sqrt(2) - sb * coupling) - 2 * KAPPA_CODE * vb])
    return drift


def otoc_curve(eps_p, ratio, a0, b0, h, seed, n_xi=N_XI, steps_per_period=STEPS_PER_PERIOD, n_periods=N_PERIODS_Q,
               n_z0=N_Z0):
    """C_cl on t = j tau / N_SUB, j = 0 .. n_periods N_SUB.

    Returns (C, C_biased, C_mean_sq): the noise-averaged derivative squared, unbiased (nan if n_xi = 1) and biased, and
    the mean of the squared derivative over the noise paths and the Wigner samples (noise average after squaring).
    """
    rng = np.random.default_rng(seed)
    drift = make_drift(eps_p, ratio)
    da = (rng.standard_normal(n_z0) + 1j * rng.standard_normal(n_z0)) / 2   # Wigner samples, std 1/2 on Re and Im
    db = (rng.standard_normal(n_z0) + 1j * rng.standard_normal(n_z0)) / 2
    a, b = np.repeat(a0 + da, n_xi), np.repeat(b0 + db, n_xi)
    u = np.stack([math.sqrt(2) * a.real, math.sqrt(2) * b.real, math.sqrt(2) * W_A * a.imag, math.sqrt(2) * W_B * b.imag])
    u = np.concatenate([u, u], axis=1)              # [x0 + h | x0 - h], same noise on both halves
    n = a.size
    u[0, :n] += h
    u[0, n:] -= h

    dt = TAU / steps_per_period
    sigma = math.sqrt(KAPPA_B) * W_B * math.sqrt(dt)
    save_every = steps_per_period // N_SUB
    C, C_biased, C_mean_sq = [], [], []

    def record(t, u):
        x = u[0] * math.cos(W_A * t) - u[2] * math.sin(W_A * t) / W_A
        d = ((x[:n] - x[n:]) / (2 * h)).reshape(n_z0, n_xi)
        mean = d.mean(axis=1)
        C.append(float(np.mean(mean**2 - d.var(axis=1, ddof=1) / n_xi)) if n_xi > 1 else float("nan"))
        C_biased.append(float(np.mean(mean**2)))
        C_mean_sq.append(float(np.mean(d**2)))

    t = 0.0
    record(t, u)
    for step in range(1, n_periods * steps_per_period + 1):
        k1 = drift(t, u)
        k2 = drift(t + dt / 2, u + dt / 2 * k1)
        k3 = drift(t + dt / 2, u + dt / 2 * k2)
        k4 = drift(t + dt, u + dt * k3)
        u = u + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        xi = sigma * rng.standard_normal(n)
        u[3, :n] += xi
        u[3, n:] += xi
        t = step * dt
        if step % save_every == 0:
            record(t, u)
    return C, C_biased, C_mean_sq


def run(args):
    p, h = args
    seed = int(round(p["epsilon_0"] * 100))   # same Wigner samples and noise for every h
    C, C_biased, _ = otoc_curve(p["epsilon_0"], p["ratio"], complex(*p["a0"]), complex(*p["b0"]), h, seed)
    return p["epsilon_0"], h, C, C_biased


def main():
    t0 = time.time()
    with open(OUT_DIR + "otoc/otoc_chaotic_start_points_22_10_2026.json") as f:
        points = json.load(f)["points"]
    with ProcessPoolExecutor(max_workers=6) as pool:
        res = list(pool.map(run, [(p, h) for p in points for h in H_LIST]))
    out = []
    for p in points:
        curves = {f"{h:g}": (C, Cb) for e, h, C, Cb in res if e == p["epsilon_0"]}
        out.append({"epsilon_0": p["epsilon_0"], "ratio": p["ratio"], "g2_MHz": p["g2_MHz"], "a0": p["a0"], "b0": p["b0"],
                    "tsave": (np.arange(N_PERIODS_Q * N_SUB + 1) * (TAU / N_SUB)).tolist(),
                    "C": {h: c[0] for h, c in curves.items()}, "C_biased": {h: c[1] for h, c in curves.items()}})
    path = OUT_DIR + "otoc/otoc_stochastic_twa_22_10_2026.json"
    with open(path, "w") as f:
        json.dump({
            "description": "Classical counterpart of the dissipative OTOC (stochastic_twa_otoc.py): "
                           "C = <(d E_xi[x_a^rot(t)] / d x_a(0))^2> over the Wigner function of |a0>|b0>, E_xi over the "
                           "vacuum noise of the buffer loss (force sqrt(kappa_b) w_b dW on phi_b''), derivative as a "
                           "central difference of half-width h on common noise paths. C[h]: unbiased (mean^2 - var/N_XI), "
                           f"C_biased[h]: mean^2. N_Z0 = {N_Z0}, N_XI = {N_XI}, dt = tau/{STEPS_PER_PERIOD}, states from "
                           "otoc_chaotic_start_points_22_10_2026.json.",
            "kappa_b": KAPPA_B, "n_z0": N_Z0, "n_xi": N_XI, "h_list": list(H_LIST), "points": out}, f)
    print(f"wrote {path} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
