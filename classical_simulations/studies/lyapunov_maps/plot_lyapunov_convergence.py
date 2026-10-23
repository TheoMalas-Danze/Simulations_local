# Figure for the transient-chaos appendix: which points of the vacuum-start Lyapunov map change sign
# when the discarded transient is extended (T_tr x 4) or the averaging time is lengthened (T x 8).
# Data: lyapunov_convergence.jl (13/10/2026). Times in the article convention (kappa_b/2pi = 96 MHz).
import json
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

sys.path.insert(0, "/Users/tmalas/Desktop/Dev/Simulations_local/Code_python")
from Plot_setting import *  # noqa: E402,F403

OUT = "/Users/tmalas/Desktop/Dev/Simulations_local/classical_simulations/data/"
FIG = "/Users/tmalas/Desktop/Dev/Simulations_local/classical_simulations/figures/lyapunov_maps/transient_lyapunov_convergence_13_10_2026.pdf"
ARTICLE = "/Users/tmalas/Desktop/Dev/Simulations_local/classical_simulations/article_chaos/figures_article/"


def load(name):
    d = json.load(open(OUT + f"lyapunov_maps/lyapunov_convergence_{name}_13_10_2026.json"))
    return np.array(d["lambda"], dtype=float).T, np.array(d["eps_p"]), np.array(d["ratio"])  # (eps_p, ratio)


lam_ref, eps_p, ratio = load("base")
E_J = 37.12 * 2 * np.pi * 1.5
g2 = E_J * np.sin(eps_p) * 0.11**2 * 0.204 / 2 / np.pi * 1e3   # MHz, g2 = E_J sin(eps_p) phi_a^2 phi_b
alpha_sq = ratio / 2                                            # eps_d / 2 g2

g2_hill = np.array(json.load(open(OUT + "hill_orbits/g2_list_12_10_2026_Hill_unstable.json")), dtype=float)
alpha_sq_hill = np.array(json.load(open(OUT + "hill_orbits/eps_d_ratio_list_12_10_2026_Hill_unstable.json"))) / 2

# 0 regular in both, 1 chaotic in both, 2 chaotic -> regular, 3 regular -> chaotic
colors = ["#e4edf3", "#e09a9a", "#0d4c74", "#b91515"]   # checked: worst CVD pair dE_OK = 14.6
labels = ["regular in both", "chaotic in both", "chaotic → regular", "regular → chaotic"]
cmap = ListedColormap(colors)
norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)


def classify(lam):
    a, b = lam_ref > 0, lam > 0
    return np.select([~a & ~b, a & b, a & ~b], [0, 1, 2], default=3)


panels = [("Ttr4", r"(a) $T_\mathrm{tr}=120/\kappa_b$"), ("T8", r"(b) $T=480/\kappa_b$")]
fig, axes = plt.subplots(1, 2, figsize=(width_in, 2.5), sharey=True, constrained_layout=True)
for ax, (name, title) in zip(axes, panels):
    lam, _, _ = load(name)
    cls = classify(lam)
    ax.pcolormesh(g2, alpha_sq, cls.T, cmap=cmap, norm=norm, shading="nearest", rasterized=True)
    ax.plot(g2_hill, alpha_sq_hill, color="k", lw=1)
    ax.axvline(100, color="0.35", lw=0.8, ls="--")
    ax.plot(98.8, 2.6, marker="*", ms=8, mfc="white", mec="k", mew=0.8)   # parameters of the transient-chaos figure
    ax.set_xlim(60, g2[-1])
    ax.grid(False)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel(r"$g_2/2\pi$ (MHz)", fontsize=8)
    ax.tick_params(labelsize=7)
    n_c2r, n_r2c = (cls == 2).sum(), (cls == 3).sum()
    print(f"{name}: chaotic->regular {n_c2r}, regular->chaotic {n_r2c}")
axes[0].set_ylabel(r"$\varepsilon_d/2g_2$", fontsize=8)
fig.legend(handles=[Patch(facecolor=c, edgecolor="0.6", lw=0.4, label=l) for c, l in zip(colors, labels)],
           loc="outside lower center", ncol=2, fontsize=7, frameon=False)
fig.savefig(FIG, dpi=300)
shutil.copy(FIG, ARTICLE)
print("saved", FIG)
