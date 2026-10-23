"""catchaos against saved outputs of the notebooks it replaces (about 2 min).
Every value below was produced by the notebook named in the test, with its own copy of the code."""
import numpy as np

from catchaos.params import g2_max, MHz, G0
from catchaos.paths import DATA
from catchaos.jsonio import load_json
from catchaos.hill import (monodromy, hill_multipliers, instability, strobe, embed, odd_mask, natural,
                           branch_on_grid, pd_null_vector, arclength, odd_content)
from catchaos.ats import HB, HB2
from catchaos.seeds import satellite, cat_seed


def close_sets(a, b, atol):
    # multisets of complex numbers equal within atol (greedy matching)
    b = list(b)
    for x in a:
        j = int(np.argmin([abs(x - y) for y in b]))
        if abs(x - b[j]) > atol:
            return False
        b.pop(j)
    return True


def test_period2_branches_eps_d10():
    # Hill_period2_branch_II.ipynb: stored HB coefficients solve HB2(32), and give the stored multipliers and types
    P = load_json(DATA/"torus_crisis"/"period2_branches_eps_d10_23_10_2026.json")
    hb2 = HB2(32, r=10.0)
    for name in ("branch_I", "branch_II", "omega_half_node", "omega_half_saddle"):
        B = P[name]
        for i in range(0, len(B["z"]), max(1, len(B["z"])//4)):
            k = i*B["z_every"]
            z = np.array(B["z"][i])
            hb2.set_s(B["g2_MHz"][k]/g2_max)
            assert np.linalg.norm(hb2.residual(z)) < 1e-8, (name, k)
            mu = monodromy(hb2, z)
            mu_ref = np.array(B["mu_re"][k]) + 1j*np.array(B["mu_im"][k])
            assert close_sets(mu, mu_ref, 1e-6*max(1, np.abs(mu_ref).max())), (name, k)
            assert instability(mu) == B["kind"][k], (name, k)


def test_bubble_edges_eps_d10():
    # Hill_period2_branch_II.ipynb section 1: period doublings of the satellite at eps_d = 10 g2 (bisection on mu = -1)
    P = load_json(DATA/"torus_crisis"/"period2_branches_eps_d10_23_10_2026.json")
    hb = HB(16, r=10.0)
    z = satellite(hb, 60/g2_max, 10.0)

    def margin(g, z):
        hb.set_s(g/g2_max)
        z = hb.newton(z)
        mu = monodromy(hb, z)
        re = mu[np.abs(mu.imag) < 1e-9].real
        return (re.min() if len(re) else 0.0) + 1, z

    z, _ = natural(hb, z, lambda g: hb.set_s(g/g2_max), np.arange(60.5, 75.01, 0.5), solver=hb.newton)
    for lo, hi, ref in ((75.0, 76.0, P["g2_pd_lower_MHz"]), (103.5, 104.0, P["g2_pd_upper_MHz"])):
        if lo > 100:
            z, _ = natural(hb, z, lambda g: hb.set_s(g/g2_max), np.arange(75.5, 103.51, 0.5), solver=hb.newton)
        v_lo, zz = margin(lo, z)
        for _ in range(30):
            mid = (lo + hi)/2
            v, zz = margin(mid, zz)
            if (v > 0) == (v_lo > 0):
                lo, v_lo = mid, v
            else:
                hi = mid
        assert abs((lo + hi)/2 - ref) < 1e-5, ((lo + hi)/2, ref)


def test_branch_II_switching():
    # Hill_period2_branch_II.ipynb section 2: branch II starts from the odd null vector, a saddle above the upper edge
    P = load_json(DATA/"torus_crisis"/"period2_branches_eps_d10_23_10_2026.json")
    g_pd = P["g2_pd_upper_MHz"]
    hb, hb2 = HB(16, r=10.0), HB2(32, r=10.0)
    z = satellite(hb, 60/g2_max, 10.0)
    z, _ = natural(hb, z, lambda g: hb.set_s(g/g2_max), list(np.arange(60.5, 103.51, 0.5)) + [g_pd], solver=hb.newton)
    hb2.set_s(g_pd/g2_max)
    z2 = hb2.newton(embed(z, 16, 32))
    v, S = pd_null_vector(hb2, z2, odd_mask(32))
    assert S[-1] < 1e-6*S[-2]
    H, end = arclength(hb2, lambda p: hb2.set_s(p*100/g2_max), np.append(z2, g_pd/100), np.append(v, 0.0),
                       n_max=40, stop=lambda y, it: "odd vanish" if it > 20 and odd_content(y[:-1], 32) < 1e-5 else None)
    ref = P["branch_II"]
    g, odd = H[-1, -1]*100, odd_content(H[-1, :-1], 32)
    assert g > g_pd
    assert abs(odd - np.interp(g, ref["g2_MHz"], ref["odd"])) < 1e-4
    hb2.set_s(g/g2_max)
    assert instability(monodromy(hb2, H[-1, :-1])) == "saddle"


def _top_satellite(hb, r, s_top):
    # Hill_satellite_seed_grid.ipynb: 60 MHz satellite at 10.12, up to the top of the map, then along the top edge to r
    s_seed = 60/g2_max
    z = satellite(hb, s_seed, 10.12)
    z, _ = natural(hb, z, lambda s: hb.set_s(s, 10.12), np.linspace(s_seed, s_top, 25)[1:])
    rs = np.round(np.arange(10.1, r - 1e-9, -0.1), 1) if r < 10.12 else np.round(np.arange(10.2, r + 1e-9, 0.1), 1)
    r_prev = 10.12
    for rr in rs:
        z, _ = natural(hb, z, lambda x: hb.set_s(s_top, x), np.linspace(r_prev, rr, 4)[1:])
        r_prev = rr
    return z


def test_satellite_seed_grid():
    G = load_json(DATA/"hill_orbits"/"satellite_seed_hill_22_10_2026.json")
    s_grid = np.sin(np.array(G["eps_p_list"]))
    ratios = G["eps_d_over_g2"]
    hb = HB(16)
    for r in (10.1, 6.0):
        j = ratios.index(r)
        z = _top_satellite(hb, r, s_grid[-1])
        zs, s_fold, _ = branch_on_grid(hb, lambda s: hb.set_s(s, r), z, s_grid[-1], s_grid, direction=-1)
        assert abs(MHz(s_fold*G0) - G["g2_fold_MHz"][j]) < 1e-3, (r, MHz(s_fold*G0), G["g2_fold_MHz"][j])
        for i in np.where(np.isfinite(zs[:, 0]))[0]:
            assert np.allclose(strobe(zs[i], 16), G["state"][j][i], rtol=0, atol=1e-8), (r, i)
        assert sum(v is not None for v in G["state"][j]) == np.isfinite(zs[:, 0]).sum()


def test_cat_seed_grid():
    G = load_json(DATA/"hill_orbits"/"cat_seed_hill_22_10_2026.json")
    eps_p = np.array(G["eps_p_list"])
    s_grid = np.sin(eps_p)
    i_half = int(np.argmin(np.abs(eps_p - 0.5)))
    r = 10.1
    j = G["eps_d_over_g2"].index(r)
    hb = HB(16)
    z = cat_seed(hb)
    z, _ = natural(hb, z, lambda x: hb.set_s(s_grid[i_half], x), np.linspace(10.12, r, 3)[1:])
    z, _ = natural(hb, z, lambda s: hb.set_s(s, r), s_grid[i_half - 1::-1])
    zs, s_fold, _ = branch_on_grid(hb, lambda s: hb.set_s(s, r), z, s_grid[0], s_grid, direction=1)
    assert abs(MHz(s_fold*G0) - G["g2_fold_MHz"][j]) < 1e-3, (MHz(s_fold*G0), G["g2_fold_MHz"][j])
    for i in np.where(np.isfinite(zs[:, 0]))[0]:
        y0, ypi = strobe(zs[i], 16, 0.0), strobe(zs[i], 16, np.pi)
        y_plus = ypi if ypi[2] > y0[2] else y0
        assert np.allclose(y_plus, G["state_plus"][j][i], rtol=0, atol=1e-8), i


def test_hill_floquet_before_period_doubling():
    # Hill_method_corrected.ipynb (older AFT over 50 periods + Hill's method): multipliers of the satellite at 10.12
    F = load_json(DATA/"hill_orbits"/"floquet_before_period_doubling_12_10_2026.json")
    hb = HB(16, r=10.12)
    z = satellite(hb, 60/g2_max, 10.12)
    s_now = 60/g2_max
    for k in (10, 40, 70):
        s = np.sin(F["eps_p"][k])
        z, _ = natural(hb, z, lambda x: hb.set_s(x), np.linspace(s_now, s, 20)[1:])
        s_now = s
        mu = hill_multipliers(hb, z)
        mu_ref = np.array(F["multipliers_real"][k]) + 1j*np.array(F["multipliers_imag"][k])
        assert close_sets(mu, mu_ref, 1e-6), (k, mu, mu_ref)
