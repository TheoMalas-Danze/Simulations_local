"""Truncated-Wigner counterpart of the quantum OTOC heatmap (OTOCS_poster_init_coherent), on the same grid and window.

Quantum: C(t) from |i alpha>|0>, alpha = sqrt(eps_d / 2 g2), on t = j tau / 8 over 3 drive periods (kappa_b T = 0.225),
rate = half the slope of a linear fit of log C against kappa_b t (lyap_best_init_coherent_15_10_2026.json for
eps_d / g2 = 2..9, otoc_largecat_na65_nb33_init_coherent_21_10_2026.json for 10..14).
Classical: the same fit of the truncated-Wigner OTOC C_cl = < (d x_a^rot(t) / d x_a(0))^2 >, the mean over the Wigner
function of |i alpha>|0> and over the vacuum noise of the buffer loss of the squared derivative along each path
(stochastic_twa_otoc.otoc_curve, C_mean_sq: one noise path per Wigner sample, central difference with H = 1e-4, i.e.
the tangent derivative). On this window, averaging the derivative over the noise BEFORE squaring underestimates the
quantum rates (0.2 against 5.5 kappa_b at eps_p = 0.93, eps_d = 10 g2); that ordering only matters for the late-time
decay (stochastic_twa_otoc.py). The mean of squares is dominated by its largest samples: N_Z0 = 16384 converges the
rates to ~0.3 kappa_b (1024 / 4096 / 16384 samples checked at eps_d = 10 g2).

Run: python3 twa_otoc_heatmap.py   (writes twa_otoc_heatmap_22_10_2026.json, about 1 min on 10 processes)
"""
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from otoc_chaotic_start_points import KAPPA_B, N_SUB, OUT_DIR, TAU
from stochastic_twa_otoc import otoc_curve

N_PERIODS = 3
N_Z0 = 16384
N_XI = 1
H = 1e-4
RATIOS = list(range(2, 15))


def rate(C):
    """As fit_log_otoc of the quantum notebook: half the slope of log C against kappa_b t."""
    t = np.arange(N_PERIODS * N_SUB + 1) * (TAU / N_SUB) * KAPPA_B
    C = np.asarray(C)
    if np.any(C <= 0):
        return float("nan")
    return float(np.polyfit(t, np.log(C), 1)[0] / 2)


def run(args):
    i, j, eps_p, ratio = args
    _, _, C = otoc_curve(eps_p, float(ratio), 1j * math.sqrt(ratio / 2), 0j, H, seed=1000 * i + j,
                         n_xi=N_XI, n_periods=N_PERIODS, n_z0=N_Z0)
    return i, j, rate(C), C


def main():
    t0 = time.time()
    with open(OUT_DIR + "otoc/lyap_best_init_coherent_15_10_2026.json") as f:
        eps_list = [round(e, 2) for e in json.load(f)["epsilon_0"]]
    tasks = [(i, j, e, r) for i, e in enumerate(eps_list) for j, r in enumerate(RATIOS)]
    lyap = np.full((len(eps_list), len(RATIOS)), np.nan)
    curves = [[None] * len(RATIOS) for _ in eps_list]
    with ProcessPoolExecutor(max_workers=10) as pool:
        for i, j, r, C in pool.map(run, tasks, chunksize=4):
            lyap[i, j], curves[i][j] = r, C
    path = OUT_DIR + "otoc/twa_otoc_heatmap_22_10_2026.json"
    with open(path, "w") as f:
        json.dump({
            "description": "Truncated-Wigner OTOC rate (twa_otoc_heatmap.py): C_cl = <(d x_a^rot(t) / d x_a(0))^2> over "
                           f"the Wigner function of |i alpha>|0> ({N_Z0} samples) and the vacuum noise of the buffer loss "
                           f"(one path per sample), derivative by central difference h = {H:g}, {N_PERIODS} drive periods "
                           "(kappa_b T = 0.225). lyap = half slope of log C vs kappa_b t, as the quantum OTOC. "
                           "Arrays are [i_epsilon_0][j_ratio]; C on t = j tau / 8.",
            "epsilon_0": eps_list, "ratio": RATIOS, "h": H, "n_z0": N_Z0, "n_xi": N_XI, "n_periods": N_PERIODS,
            "kappa_b": KAPPA_B, "lyap": lyap.tolist(), "C": curves}, f)
    print(f"wrote {path} in {time.time() - t0:.0f} s; {np.isnan(lyap).sum()} nan")


if __name__ == "__main__":
    main()
