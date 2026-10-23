"""Continuation of harmonic-balance branches in one parameter p.

`setp(p)` sets the parameter on the HB object (e.g. lambda p: hb.set_s(p, r)). Unknowns y = (z, p).
Pseudo-arclength: predictor along the tangent (null vector of [dR/dz, dR/dp], from the SVD), corrector with the
constraint t.(y - y_prev) = ds.
"""
import numpy as np
from scipy.optimize import root


def extended(hb, setp, h=1e-7):
    # residual F(y) and its Jacobian dF(y) = [dR/dz, dR/dp] (dR/dp by forward difference)
    def F(y):
        setp(y[-1])
        return hb.residual(y[:-1])

    def dF(y):
        setp(y[-1])
        Jz, r_ = hb.jacobian(y[:-1]), hb.residual(y[:-1])
        setp(y[-1] + h)
        return np.column_stack([Jz, (hb.residual(y[:-1]) - r_)/h])
    return F, dF


def tangent(dF, y, t_prev=None, direction=None):
    # unit tangent; oriented along t_prev, or (first step) so that sign(dp) = direction
    t = np.linalg.svd(dF(y))[2][-1]
    if (t_prev is not None and t @ t_prev < 0) or (t_prev is None and direction is not None and np.sign(t[-1]) != direction):
        t = -t
    return t


def natural(hb, z, setp, values, solver=None):
    # natural continuation through the parameter values; returns the last solution and all of them
    solver = solver or hb.solve
    out = []
    for p in values:
        setp(p)
        z = solver(z)
        out.append(z.copy())
    return z, np.array(out)


def _corrector_newton(F, dF, y0, t, ds, tol=1e-11, maxit=20):
    w = y0 + ds*t
    for _ in range(maxit):
        G = np.append(F(w), t @ (w - y0) - ds)
        if np.linalg.norm(G) < tol:
            return w
        w = w - np.linalg.solve(np.vstack([dF(w), t]), G)
    if np.linalg.norm(np.append(F(w), t @ (w - y0) - ds)) < 1e-9:
        return w
    raise RuntimeError("corrector did not converge")


def _corrector_hybr(F, dF, y0, t, ds, tol=1e-12):
    sol = root(lambda w: np.append(F(w), t @ (w - y0) - ds), y0 + ds*t,
               jac=lambda w: np.vstack([dF(w), t]), method="hybr", tol=tol)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.x


def arclength(hb, setp, y, t, ds0=2e-3, ds_max=3e-3, ds_min=1e-8, n_max=5000, stop=None, corrector="newton"):
    """Pseudo-arclength from y = (z, p) along the tangent t. Failed steps are halved, accepted ones grow by 1.3.
    stop(y, it) -> None or a string (reason to stop). Returns (array of the points (z, p), reason)."""
    F, dF = extended(hb, setp)
    correct = _corrector_newton if corrector == "newton" else _corrector_hybr
    ds, hist, end = ds0, [y.copy()], "n_max reached"
    for it in range(n_max):
        try:
            yn = correct(F, dF, y, t, ds)
        except (RuntimeError, np.linalg.LinAlgError):
            ds /= 2
            if ds < ds_min:
                end = f"step too small at p = {y[-1]:.6g}"
                break
            continue
        t = tangent(dF, yn, t)
        y = yn
        hist.append(y.copy())
        ds = min(1.3*ds, ds_max)
        reason = stop(y, it) if stop else None
        if reason:
            end = reason
            break
    return np.array(hist), end


def continue_to_fold(hb, setp, z, p, ds=0.002, nmax=4000, direction=-1, ds_min=1e-7):
    """Arclength from (z, p) in the direction of p given by `direction` until p turns back.
    Returns the p of the fold (vertex of a parabola through the last three points; NaN if none) and the points."""
    F, dF = extended(hb, setp)
    y = np.append(z, p)
    t = tangent(dF, y, direction=direction)
    hist = [y.copy()]
    for _ in range(nmax):
        try:
            y_new = _corrector_hybr(F, dF, y, t, ds)
        except RuntimeError:
            ds /= 2
            if ds < ds_min:
                break
            continue
        y = y_new
        t_new = tangent(dF, y, t)
        hist.append(y.copy())
        if np.sign(t_new[-1]) != np.sign(t[-1]):
            c = np.polyfit([-1, 0, 1], np.array(hist)[-3:, -1], 2)
            return np.polyval(c, -c[1]/(2*c[0])), np.array(hist)
        t = t_new
    return np.nan, np.array(hist)


def branch_on_grid(hb, setp, z, p, p_grid, direction=-1, ds0=0.003, ds_max=0.01, ds_fold=1e-6):
    """Pseudo-arclength from (z, p) in the direction of p given by `direction`, up to the fold, sampling the branch at
    every value of p_grid it crosses (solved at fixed p from the linear interpolation of the step; must stay within 1%).
    A step that passes the fold (p turns back) is rejected and halved, so the branch is never continued past the fold.
    Returns (z on p_grid, NaN rows where the branch is not reached; p of the fold, NaN if p_grid is left first; steps)."""
    F, dF = extended(hb, setp)
    zs = np.full((len(p_grid), len(z)), np.nan)
    zs[np.isclose(p_grid, p, rtol=0, atol=1e-12)] = z
    y = np.append(z, p)
    t = tangent(dF, y, direction=direction)
    ds, steps = ds0, 0
    while True:
        try:
            y_new = _corrector_hybr(F, dF, y, t, ds)
            t_new = tangent(dF, y_new, t)
            ok = np.sign(t_new[-1]) == direction and np.sign(y_new[-1] - y[-1]) == direction
        except RuntimeError:
            ok = False
        if not ok:                                   # failed, or passed the fold
            ds /= 2
            if ds < ds_fold:
                return zs, y[-1], steps
            continue
        steps += 1
        lo, hi = sorted((y[-1], y_new[-1]))
        crossed = (p_grid >= lo) & (p_grid < hi) if direction < 0 else (p_grid > lo) & (p_grid <= hi)
        for g in np.where(crossed)[0]:
            w = (y[-1] - p_grid[g])/(y[-1] - y_new[-1])
            guess = y[:-1] + w*(y_new[:-1] - y[:-1])
            setp(p_grid[g])
            zs[g] = hb.solve_robust(guess)
            assert np.linalg.norm(zs[g] - guess) < 1e-2*np.linalg.norm(guess), "grid solve left the branch"
        y, t = y_new, t_new
        ds = min(1.3*ds, ds_max)
        if (direction < 0 and y[-1] < p_grid.min()) or (direction > 0 and y[-1] > p_grid.max()):
            return zs, np.nan, steps


def pd_null_vector(hb2, z2, odd):
    # direction of the period-2 branch at a period doubling: null vector of the odd block of the period-2 Jacobian
    J = hb2.jacobian(z2)
    S, Vt = np.linalg.svd(J[np.ix_(odd, odd)])[1:]
    v = np.zeros(len(z2))
    v[odd] = Vt[-1]
    return v, S
