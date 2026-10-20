"""
Fine Hill sweeps around the period doubling (eps_d = 10.12 g2), for the insets of plot_floquet_multipliers.ipynb.

The sweeps of Hill_method_corrected.ipynb (period-1 orbit) and Hill_method_doubling_corrected.ipynb (period-2 orbit)
step by about 0.3 MHz in g2. That is wider than the window, about 0.25 MHz, where the critical pair of multipliers is
real (collision on the real axis, then period doubling). This script reruns both sweeps with a step of 0.02 MHz in g2
around the period doubling, with the same code (cells read from the notebooks) and the same settings.

Outputs (same format as the floquet_*_period_doubling_12_10_2026.json files):
    analysis/output/chicago_poster_2026/floquet_before_period_doubling_zoom_16_10_2026.json
    analysis/output/chicago_poster_2026/floquet_after_period_doubling_zoom_16_10_2026.json
"""
import json
import os
import numpy as np
from scipy.special import jv
from scipy.sparse import lil_matrix
from scipy.linalg import block_diag, eig as scipy_eig
from scipy.optimize import root, brentq

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = "/Users/tmalas/Desktop/Dev/Simulations_local/classical_simulations/analysis/output/chicago_poster_2026/"
EPS_D_RATIO = 10.12


def load_definitions(notebook):
    """Run the parameter and definition cells of a Hill notebook (located by their content) in this module."""
    cells = json.load(open(os.path.join(HERE, notebook)))["cells"]
    markers = ("E_J = 37.12", "def build_L", "def build_Gamma", "def residual", "def select_genuine", "def match_eigs")
    for marker in markers:
        src = next("".join(c["source"]) for c in cells if c["cell_type"] == "code" and marker in "".join(c["source"]))
        exec(src, globals())


def g2_MHz(eps_p):
    return g2_of(eps_p)/2/np.pi*1e3


def eps_p_of(g2_target):
    return brentq(lambda e: g2_MHz(e) - g2_target, 0.3, 1.2)


def solve_at(z, eps_p, ratio=EPS_D_RATIO):
    """Orbit at (eps_p, eps_d = ratio g2) from the guess z. Sets the globals used by f_nl_time / db_dz."""
    global beta_a_p, beta_b_p, B
    beta_a_p = 2*E_J*np.sin(eps_p)*varphi_a**2/w_a
    beta_b_p = 2*E_J*np.sin(eps_p)*varphi_b**2/w_b
    B = np.array([beta_a_p, beta_b_p])
    b_ext = build_b_ext(N_H, ratio*g2_of(eps_p)*varphi_b/w_b, nu)
    return powell_hybrid_solve(z, L, b_ext, Gamma, Gamma_inverse, n, N, verbose=False)


def sweep(z, eps_p_list):
    """Continuation along eps_p_list, Floquet exponents sorted by match_eigs as in the notebooks."""
    eigvals_list, prev = [], None
    for eps_p in eps_p_list:
        z = solve_at(z, eps_p)
        ev, vec = Floq_coef(z, L, M, C, N_H, n)
        if prev is not None:
            perm = match_eigs(prev[0], prev[1], ev, vec, T=2*np.pi*nu/omega)
            ev, vec = ev[perm], vec[:, perm]
        eigvals_list.append(ev)
        prev = (ev, vec)
    eigvals_array = np.array(eigvals_list)
    print("max Liouville residual:", np.max([abs(liouville_residual(ev, M, C)) for ev in eigvals_array]))
    return eigvals_array


def save(path, description, eps_p_list, eigvals_array, T):
    mu = np.exp(eigvals_array*T)
    data = {
        "description": description,
        "eps_d_over_g2": EPS_D_RATIO,
        "g2_definition": "E_J sin(eps_p) phi_a^2 phi_b",
        "kappa_b": kappa_b,
        "N_H": N_H,
        "T": T,
        "eps_p": list(eps_p_list),
        "g2_MHz": list(g2_MHz(np.array(eps_p_list))),
        "exponents_real": np.real(eigvals_array).tolist(),
        "exponents_imag": np.imag(eigvals_array).tolist(),
        "multipliers_real": np.real(mu).tolist(),
        "multipliers_imag": np.imag(mu).tolist(),
    }
    with open(path, "w") as f:
        json.dump(data, f)
    print("saved", path)


# ---------------------------------------------------------------- period-1 orbit (Hill_method_corrected.ipynb)
load_definitions("Hill_method_corrected.ipynb")
n, N_H, omega, nu = 2, 16, 2.0, 2
N = 900*N_H
tau_j = np.linspace(0, 100*np.pi, N)
varphi_a, varphi_b = np.sqrt(Z_a)/np.sqrt(2)/phi_quantum, np.sqrt(Z_b)/np.sqrt(2)/phi_quantum
M = np.array([[1.0, 0.0], [0.0, 0.25]])
C = np.array([[0.0, 0.0], [0.0, kappa_b/w_b]])
L = build_L(M, C, np.eye(2), N_H, omega, nu)
Gamma = build_Gamma(tau_j, N_H, n, omega, nu)
Gamma_inverse = np.linalg.pinv(Gamma)

z = np.array([-1.64935222e-01, -2.83634057e-01, -3.61112171e-02, -1.04869837e-16,
        6.90729610e-01,  8.65452455e-17,  2.06183395e-04, -5.86034682e-02,
       -2.12934625e-03,  5.60620385e-01, -1.42972809e-03, -1.59629647e-02,
        9.08371762e-03,  9.99380000e-02, -6.45296832e-04, -5.58158012e-03,
        3.04362567e-03,  2.61630898e-02])  # seed of the sweep of Hill_method_corrected.ipynb
z = np.pad(z, (0, n*(2*N_H+1)-len(z)))
# the seed was converged with the former g2 (J1): continue it in eps_d at eps_p = 0.5, as in the notebook
for ratio in np.linspace(EPS_D_RATIO*g2_J1_of(0.5)/g2_of(0.5), EPS_D_RATIO, 25):
    z = solve_at(z, 0.5, ratio)
eps_p_before = np.linspace(eps_p_of(74.0), eps_p_of(76.6), 131)   # ascending g2, as in the notebook
for eps_p in np.linspace(0.5, eps_p_before[0], 15):
    z = solve_at(z, eps_p)
ev_before = sweep(z, eps_p_before)
save(OUT_DIR + "floquet_before_period_doubling_zoom_16_10_2026.json",
     "Floquet exponents and multipliers of the period-1 (satellite) orbit around the period doubling, "
     "from floquet_zoom_period_doubling.py (code and settings of Hill_method_corrected.ipynb)",
     eps_p_before, ev_before, 2*np.pi/(omega/2))

# ---------------------------------------------------------------- period-2 orbit (Hill_method_doubling_corrected.ipynb)
load_definitions("Hill_method_doubling_corrected.ipynb")
n, N_H, omega, nu = 2, 12, 4.0, 4
N = 900*N_H
tau_j = np.linspace(0, 100*np.pi, N)
varphi_a, varphi_b = np.sqrt(Z_a)/np.sqrt(2)/phi_quantum, np.sqrt(Z_b)/np.sqrt(2)/phi_quantum
M = np.array([[1/4, 0.0], [0.0, 1/16]])
C = np.array([[0.0, 0.0], [0.0, 0.5*kappa_b/w_b]])
L = build_L(M, C, np.eye(2), N_H, omega, nu)
Gamma = build_Gamma(tau_j, N_H, n, omega, nu)
Gamma_inverse = np.linalg.pinv(Gamma)

z = np.pad(z_0_init, (0, n*(2*N_H+1)-len(z_0_init)))   # period-2 orbit at eps_p = 1.2 (former g2)
for ratio in np.linspace(EPS_D_RATIO*g2_J1_of(1.2)/g2_of(1.2), EPS_D_RATIO, 25):
    z = solve_at(z, 1.2, ratio)
eps_p_after = np.linspace(eps_p_of(77.0), eps_p_of(74.0), 151)    # descending g2, as in the notebook
for eps_p in np.linspace(1.2, eps_p_after[0], 40):
    z = solve_at(z, eps_p)
ev_after = sweep(z, eps_p_after)
save(OUT_DIR + "floquet_after_period_doubling_zoom_16_10_2026.json",
     "Floquet exponents and multipliers of the period-2 (satellite) orbit around the period doubling, "
     "from floquet_zoom_period_doubling.py (code and settings of Hill_method_doubling_corrected.ipynb)",
     eps_p_after, ev_after, 2*np.pi/(omega/4))
