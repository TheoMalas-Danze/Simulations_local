"""Branches of the ATS model followed in p = g2/(ps MHz) at fixed eps_d/g2 = r (Hill_period2_branch_II, crisis_collision, story_A).
Same algorithms and stopping rules as the notebooks' copies, so the results are bit-identical."""
import numpy as np

from .params import g2_max
from .ats import HB, HB2, rotating_frame, section
from .hill import extended, arclength, monodromy, instability, odd_content

PS = 100.0                                              # p = g2 / (100 MHz)


def g2_setter(hb, r, ps=PS):
    # setp(p) for continuation in p = g2/(ps MHz)
    return lambda p: hb.set_s(p*ps/g2_max, r)


def extended_g2(hb, r, ps=PS):
    # residual F(y) and Jacobian dF(y) of y = (z, p)
    return extended(hb, g2_setter(hb, r, ps))


def arclength_g2(hb, y, t, r, ds0=2e-3, ds_max=3e-3, g_lo=60.0, g_hi=140.0, n_max=5000, stop_odd=True, ps=PS):
    """Pseudo-arclength from y = (z, p) along t; stops when g2 leaves [g_lo, g_hi] or (stop_odd) the odd harmonics vanish again.
    Returns (points (z, p), reason)."""
    def stop(y, it):
        g = y[-1]*ps
        if not g_lo < g < g_hi:
            return f"leaves [{g_lo:g}, {g_hi:g}] MHz"
        if stop_odd and it > 20 and odd_content(y[:-1], hb.N_H) < 1e-5:
            return f"odd harmonics vanish at {g:.3f} MHz"
    H, end = arclength(hb, g2_setter(hb, r, ps), y, t, ds0=ds0, ds_max=ds_max, ds_min=1e-8, n_max=n_max, stop=stop)
    if end.startswith("step too small"):
        end = f"step too small at {H[-1, -1]*ps:.2f} MHz"
    return H, end


def describe(H, N_H, r, period=2, ps=PS):
    """Multipliers (sorted by |mu|, decreasing), type, odd content, section points (orbit and partner) and <|a_R|^2>
    at every point of a branch H (rows (z, p))."""
    hb_ = HB2(N_H, r=r) if period == 2 else HB(N_H, r=r)
    tt = np.linspace(0, 2*np.pi*period, 300*period, endpoint=False)
    out = dict(g2=H[:, -1]*ps, mu=[], kind=[], odd=[], sec=[], sec_partner=[], a2=[])
    for y in H:
        hb_.set_s(y[-1]*ps/g2_max)
        mu = monodromy(hb_, y[:-1])
        mu = mu[np.argsort(-np.abs(mu))]
        s0, s1 = section(y[:-1], N_H, period)
        out["mu"].append(mu)
        out["kind"].append(instability(mu))
        out["odd"].append(odd_content(y[:-1], N_H) if period == 2 else 0.0)
        out["sec"].append(s0)
        out["sec_partner"].append(s1)
        out["a2"].append(np.mean(np.abs(rotating_frame(y[:-1], N_H, tt, period)[0])**2))
    return {k: np.array(v) for k, v in out.items()}
