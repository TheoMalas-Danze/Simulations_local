"""Initial states on the classical attractor for the quantum OTOC, and the classical references from the same states.

For each (eps_p, ratio): integrate the map EOM (equation_1ODE! of config/equations.jl) from the cat lobe a = i|alpha|,
b = 0 over the map's transient Ttr = 15/kappa (code), then on, period by period, to the first whole number of drive
periods n tau where |a|^2 <= MAX_A_SQ and |b|^2 <= MAX_B_SQ (the quantum truncation n_a = 65, n_b = 33 then holds the
coherent state; chaotic excursions reach |a|^2 ~ 50). The drive phase there is 0 as at t = 0 of the quantum solve. The
state, (a0, b0) in the lab frame, is the centre of the coherent state |a0>|b0> of the quantum run (section "From the classical attractor" of Quantum_codes/OTOCS_poster_init_coherent.ipynb).
From that state:
  lambda_long: half the slope of log C_cl over 30/kappa (code), the map's averaging window, and over each half of it,
               to flag transient chaos (the second half then drops to ~0 or below);
  lambda_10ns: the same over the first 10 ns only, the reference for the quantum window when the chaos is transient
               (above ~100 MHz it ends after 15-40 ns from these states);
  logC_point, logC_twa: log C_cl on the quantum time grid (n_sub points per drive period over n_periods periods), for a
               single trajectory and averaged over the Wigner function of |a0>|b0> (vacuum noise, none from the loss).
C_cl(t) = (d x_a^rot(t) / d x_a(0))^2 is the classical OTOC (Poisson bracket {x_a(t), p_a(0)}), from the tangent
dynamics. Rates and kappa_b are in the GKSL convention of the quantum notebook, kappa_b = 2 pi / 10.4.

State: u = [phi_a, phi_b, phi_a', phi_b'] = sqrt(2) [Re a, Re b, w_a Im a, w_b Im b] (get_init of the Julia code), so
x_a^rot(t) = sqrt(2) Re(a e^{i w_a t}) = phi_a cos(w_a t) - phi_a' sin(w_a t) / w_a.

Run: python3 otoc_chaotic_start_points.py   (about 2 min on 6 processes)
"""
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.integrate import solve_ivp

OUT_DIR = "/Users/tmalas/Desktop/Dev/Simulations_local/classical_simulations/data/"

W_A = 25.338776456203686
W_B = 2 * W_A
PHI_QUANTUM = 1 / 2
PHI_A, PHI_B = 0.11, 0.204
Z_A, Z_B = 2 * PHI_QUANTUM**2 * PHI_A**2, 2 * PHI_QUANTUM**2 * PHI_B**2
E_J = 37.12 * 2 * math.pi * 1.5
KAPPA_CODE = math.pi / 10.4     # Julia convention (-2 kappa phi_b'), i.e. kappa_b / 2
KAPPA_B = 2 * KAPPA_CODE        # GKSL rate, the one the OTOC times are normalised by
TAU = 2 * math.pi / W_B         # drive period
N_SUB = 8
RTOL = ATOL = 1e-11

N_PERIODS_Q = 24                # quantum window: 24 drive periods, kappa_b T ~ 1.8
T_TR = 15 / KAPPA_CODE
T_LONG = 30 / KAPPA_CODE
T_SHORT = 10.0                  # ns
N_TWA = 64
MAX_A_SQ, MAX_B_SQ = 15.0, 3.0
POINTS = [(eps_p, 10.0) for eps_p in (0.6, 0.69, 0.75, 0.81, 0.87, 0.93)]


def g2_of(eps_p):
    return math.sin(eps_p) * E_J * PHI_A**2 * PHI_B


def make_rhs(eps_p, ratio):
    """equation_1ODE! (flux form, kappa_a = 0, static pump) and its tangent dynamics, on y = [u, delta u]."""
    eps_d = ratio * g2_of(eps_p)
    pump = 2 * E_J * math.sin(eps_p)          # 2 E_J sin(eps_0 cos(w_p t)), w_p = 0
    k_a, k_b = math.sqrt(Z_A) / PHI_QUANTUM, math.sqrt(Z_B) / PHI_QUANTUM
    sa, sb = math.sqrt(Z_A) * pump, math.sqrt(Z_B) * pump

    def rhs(t, y):
        pa, pb, va, vb, dpa, dpb, dva, dvb = y
        theta = k_a * pa + k_b * pb
        coupling = (math.cos(theta) - 1) / PHI_QUANTUM
        dcoupling = -math.sin(theta) * (k_a * dpa + k_b * dpb) / PHI_QUANTUM
        return [va, vb,
                -W_A * (W_A * pa - sa * coupling),
                -W_B * (W_B * pb + eps_d * math.cos(W_B * t) * math.sqrt(2) - sb * coupling) - 2 * KAPPA_CODE * vb,
                dva, dvb,
                -W_A * (W_A * dpa - sa * dcoupling),
                -W_B * (W_B * dpb - sb * dcoupling) - 2 * KAPPA_CODE * dvb]
    return rhs


def state_of(a, b):
    return [math.sqrt(2) * a.real, math.sqrt(2) * b.real, math.sqrt(2) * W_A * a.imag, math.sqrt(2) * W_B * b.imag]


def amplitudes_of(u):
    return (u[0] + 1j * u[2] / W_A) / math.sqrt(2), (u[1] + 1j * u[3] / W_B) / math.sqrt(2)


def evolve(rhs, u0, t0, t1):
    """State only (zero deviation), from t0 to t1."""
    sol = solve_ivp(rhs, (t0, t1), list(u0) + [0.0] * 4, method="DOP853", rtol=RTOL, atol=ATOL)
    return sol.y[:4, -1]


def log_otoc(rhs, a0, b0, n_periods):
    """log C_cl(t_j), t_j = j tau / N_SUB, j = 0 .. n_periods N_SUB, from (a0, b0); deviation along x_a(0) = phi_a(0).

    One solve per drive period, so the deviation can be renormalised between periods (it is linear in itself).
    """
    y = np.array(state_of(a0, b0) + [1.0, 0.0, 0.0, 0.0])
    ts = np.arange(n_periods * N_SUB + 1) * (TAU / N_SUB)
    log_c = np.empty(len(ts))
    log_scale = 0.0

    def log_c_at(t, dev):
        dx = dev[0] * math.cos(W_A * t) - dev[2] * math.sin(W_A * t) / W_A
        return 2 * (math.log(abs(dx)) + log_scale)

    log_c[0] = log_c_at(0.0, y[4:])
    for n in range(n_periods):
        t_eval = ts[n * N_SUB:(n + 1) * N_SUB + 1]
        sol = solve_ivp(rhs, (t_eval[0], t_eval[-1]), y, method="DOP853", t_eval=t_eval, rtol=RTOL, atol=ATOL)
        for j in range(1, N_SUB + 1):
            log_c[n * N_SUB + j] = log_c_at(sol.t[j], sol.y[4:, j])
        y = sol.y[:, -1].copy()
        norm = np.linalg.norm(y[4:])
        if norm > 1e50:   # keep the linear deviation finite over long windows
            y[4:] /= norm
            log_scale += math.log(norm)
    return ts, log_c


def half_slope(ts, log_c):
    return np.polyfit(ts, log_c, 1)[0] / 2


def reference(args):
    eps_p, ratio, seed = args
    rhs = make_rhs(eps_p, ratio)
    n_tr = round(T_TR / TAU)
    u = evolve(rhs, state_of(1j * math.sqrt(ratio / 2), 0j), 0.0, n_tr * TAU)
    a0, b0 = amplitudes_of(u)
    while abs(a0)**2 > MAX_A_SQ or abs(b0)**2 > MAX_B_SQ:   # next drive period, until the truncation holds
        u = evolve(rhs, u, n_tr * TAU, (n_tr + 1) * TAU)
        n_tr += 1
        a0, b0 = amplitudes_of(u)

    ts_l, log_c_l = log_otoc(rhs, a0, b0, round(T_LONG / TAU))
    h = len(ts_l) // 2
    lam_long = half_slope(ts_l, log_c_l)
    lam_first, lam_second = half_slope(ts_l[:h], log_c_l[:h]), half_slope(ts_l[h - 1:], log_c_l[h - 1:])
    k = ts_l <= T_SHORT
    lam_10ns = half_slope(ts_l[k], log_c_l[k])

    ts, log_c_point = log_otoc(rhs, a0, b0, N_PERIODS_Q)
    rng = np.random.default_rng(seed)
    logs = np.array([log_otoc(rhs, a0 + complex(*rng.standard_normal(2)) / 2,
                              b0 + complex(*rng.standard_normal(2)) / 2, N_PERIODS_Q)[1] for _ in range(N_TWA)])
    mx = logs.max(axis=0)
    log_c_twa = mx + np.log(np.mean(np.exp(logs - mx), axis=0))   # log of the mean, without overflow

    return {
        "epsilon_0": eps_p, "ratio": ratio, "g2_MHz": g2_of(eps_p) / (2 * math.pi) * 1e3,
        "n_transient_periods": n_tr,
        "a0": [a0.real, a0.imag], "b0": [b0.real, b0.imag],
        "abs_a0_sq": abs(a0)**2, "abs_b0_sq": abs(b0)**2,
        "lambda_long_per_ns": lam_long, "lambda_long_first_half_per_ns": lam_first,
        "lambda_long_second_half_per_ns": lam_second, "lambda_long_over_kappa_b": lam_long / KAPPA_B,
        "lambda_10ns_per_ns": lam_10ns, "lambda_10ns_over_kappa_b": lam_10ns / KAPPA_B,
        "tsave": ts.tolist(), "logC_point": log_c_point.tolist(), "logC_twa": log_c_twa.tolist(),
    }


def main():
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=len(POINTS)) as pool:
        out = list(pool.map(reference, [(eps_p, ratio, k + 1) for k, (eps_p, ratio) in enumerate(POINTS)]))
    for r in out:
        print(f"eps_p = {r['epsilon_0']} ({r['g2_MHz']:.1f} MHz), ratio = {r['ratio']}, n = {r['n_transient_periods']}: |a0|^2 = {r['abs_a0_sq']:.2f}, "
              f"|b0|^2 = {r['abs_b0_sq']:.2f}, lambda_long = {r['lambda_long_per_ns']:.3f}/ns "
              f"(halves {r['lambda_long_first_half_per_ns']:.3f}, {r['lambda_long_second_half_per_ns']:.3f}), "
              f"first 10 ns {r['lambda_10ns_per_ns']:.3f}/ns")
    path = OUT_DIR + "otoc/otoc_chaotic_start_points_22_10_2026.json"
    with open(path, "w") as f:
        json.dump({
            "description": f"Initial states (a0, b0) on the classical attractor for the quantum OTOC "
                           "(otoc_chaotic_start_points.py): map EOM from a = i alpha, b = 0 over Ttr = 15/kappa_code, "
                           f"then to the first drive period (n_transient_periods) with |a0|^2 <= {MAX_A_SQ:g} and |b0|^2 <= {MAX_B_SQ:g}. Classical references from the same state: "
                           "lambda_long (half slope of log C_cl over 30/kappa_code; lambda_10ns over the first 10 ns), logC_point and logC_twa "
                           f"({N_TWA} Wigner samples of |a0>|b0>) on tsave (n_sub = {N_SUB} per drive period, "
                           f"{N_PERIODS_Q} periods). kappa_b = 2pi/10.4 (GKSL).",
            "kappa_b": KAPPA_B, "n_sub": N_SUB, "n_periods": N_PERIODS_Q, "points": out}, f)
    print(f"wrote {path} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
