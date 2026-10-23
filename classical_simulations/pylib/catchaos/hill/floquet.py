"""Floquet stability of a harmonic-balance orbit: monodromy matrix and Hill's method.

Monodromy: integrates M dX'' + C dX' + (K + d f_nl/dX(X(tau))) dX = 0 over one period 2 pi * hb.period (in tau).
Hill: quadratic eigenproblem lambda^2 Delta2 + lambda Delta1 + J(z) = 0 in the basis time tau_d, keeping the 2n genuine
exponents (one per family, Detroux et al. 2015), folded into |Im lambda| <= 1/2. Multipliers mu = exp(2 pi lambda).
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import block_diag, eig as scipy_eig
from scipy.optimize import linear_sum_assignment

from .orbit import orbit


def monodromy_matrix(hb, z, rtol=1e-10, atol=1e-12):
    n, p = hb.n, hb.period
    Minv = np.linalg.inv(hb.M)

    def rhs(t, Y):
        X = np.array([x[0] for x, _ in orbit(z, hb.N_H, np.atleast_1d(t)/p, n)])[:, None]
        A = np.zeros((2*n, 2*n))
        A[:n, n:] = np.eye(n)
        A[n:, :n] = -Minv @ (hb.K + hb.dfnl(X)[0])
        A[n:, n:] = -Minv @ hb.C                     # hb.C is in tau units for any period
        return (A @ Y.reshape(2*n, 2*n)).reshape(-1)
    sol = solve_ivp(rhs, (0, 2*np.pi*p), np.eye(2*n).reshape(-1), method="DOP853", rtol=rtol, atol=atol)
    return sol.y[:, -1].reshape(2*n, 2*n)


def monodromy(hb, z, **kw):
    # the 2n Floquet multipliers over one period of the orbit
    return np.linalg.eigvals(monodromy_matrix(hb, z, **kw))


def instability(mu):
    # type of the leading multiplier
    m = mu[np.argmax(np.abs(mu))]
    if abs(m) < 1:
        return "stable"
    if abs(m.imag) > 1e-6:
        return "NS"
    return "PD" if m.real < 0 else "saddle"


def unstable_set(mu):
    # multipliers outside the unit circle: "+" real > 1, "-" real < -1, "c" complex pair (counted once)
    out = [("c" if abs(m.imag) > 1e-6 else ("+" if m.real > 0 else "-")) for m in mu if abs(m) > 1 and m.imag >= -1e-6]
    return "".join(sorted(out)) or "stable"


def transitions(D):
    # changes of the set of unstable multipliers along a branch D (keys "g2", "mu");
    # g2 of the crossing interpolated on the multiplier that crosses |mu| = 1
    res, sets = [], [unstable_set(m) for m in D["mu"]]
    for i in range(1, len(sets)):
        if sets[i] != sets[i - 1]:
            a, b = np.sort(np.abs(D["mu"][i - 1])), np.sort(np.abs(D["mu"][i]))
            k = np.argmax(np.abs((a > 1).astype(int) - (b > 1).astype(int))) if np.any((a > 1) != (b > 1)) else None
            g = D["g2"][i - 1] + (1 - a[k])*(D["g2"][i] - D["g2"][i - 1])/(b[k] - a[k]) if k is not None else D["g2"][i]
            res.append((g, sets[i - 1], sets[i]))
    return res


# Hill's method

def complex_harmonics(U, N_H, n):
    # real coefficient vectors (columns of U) -> complex harmonics A[k + N_H, mode, column]
    A = np.zeros((2*N_H + 1, n, U.shape[1]), dtype=complex)
    A[N_H] = U[:n, :]/np.sqrt(2)
    for k in range(1, N_H + 1):
        s = U[(2*k - 1)*n:(2*k)*n, :]
        c = U[(2*k)*n:(2*k + 1)*n, :]
        A[N_H + k] = (c - 1j*s)/2
        A[N_H - k] = (c + 1j*s)/2
    return A


def harmonic_centroid(A, N_H):
    w = np.sum(np.abs(A)**2, axis=1)
    k = np.arange(-N_H, N_H + 1)[:, None]
    return np.sum(k*w, axis=0)/np.sum(w, axis=0)


def shifted_overlap(a, b, m, N_H):
    if m > 0:
        a_s, b_s = a[m:], b[:-m]
    elif m < 0:
        a_s, b_s = a[:m], b[-m:]
    else:
        a_s, b_s = a, b
    return np.abs(np.vdot(a_s, b_s))/(np.linalg.norm(a)*np.linalg.norm(b) + 1e-300)


def select_genuine(eigvals, eigvecs, N_H, n, tol=1e-3, overlap_min=0.9):
    # keep one exponent per family lambda + i m (best-centred eigenvector first), fold into |Im| <= 1/2
    U = eigvecs[eigvecs.shape[0]//2:, :]
    A = complex_harmonics(U, N_H, n)
    centroid = harmonic_centroid(A, N_H)
    order = np.lexsort((np.abs(np.imag(eigvals)), np.abs(centroid)))
    chosen = []
    for j in order:
        lam = eigvals[j]
        if not np.isfinite(lam):
            continue
        same_family = False
        for c in chosen:
            d = lam - eigvals[c]
            m = int(np.round(np.imag(d)))
            if (m != 0 and abs(np.real(d)) < tol and abs(np.imag(d) - m) < tol
                    and shifted_overlap(A[:, :, c], A[:, :, j], m, N_H) > overlap_min):
                same_family = True
                break
        if not same_family:
            chosen.append(j)
        if len(chosen) == 2*n:
            break
    chosen = np.array(chosen)
    lam = eigvals[chosen]
    lam_folded = lam - 1j*np.round(np.imag(lam))
    lam_folded = np.where(np.abs(np.imag(lam_folded) + 0.5) < 1e-6, lam_folded + 1j, lam_folded)
    return lam_folded, eigvecs[:, chosen]


def hill_exponents(hb, z, vectors=False):
    # genuine Floquet exponents in units of 1/tau_d (basis time): mu = exp(2 pi lambda), rate in 1/ns = lambda w_a/period
    N_H, n = hb.N_H, hb.n
    Mb, Cb, d = hb.M/hb.period**2, hb.C/hb.period, (2*N_H + 1)*n
    Delta_1 = block_diag(*([Cb]*(2*N_H + 1)))
    for k in range(1, N_H + 1):
        i, j = 2*k - 1, 2*k
        Delta_1[i*n:(i+1)*n, j*n:(j+1)*n] = -2*k*Mb
        Delta_1[j*n:(j+1)*n, i*n:(i+1)*n] = 2*k*Mb
    Delta_2 = block_diag(*([Mb]*(2*N_H + 1)))
    B_1 = np.block([[Delta_1, hb.jacobian(z)], [-np.eye(d), np.zeros((d, d))]])
    B_2 = -np.block([[Delta_2, np.zeros((d, d))], [np.zeros((d, d)), np.eye(d)]])
    lam, vec = select_genuine(*scipy_eig(B_1, B_2), N_H, n)
    return (lam, vec) if vectors else lam


def hill_multipliers(hb, z):
    return np.exp(2*np.pi*hill_exponents(hb, z))


def liouville_residual(hb, lam):
    # Liouville/Abel: the real parts of the 2n exponents (basis units) sum to -tr(M^-1 C) in the basis; should be ~0
    return np.sum(np.real(lam)) + np.trace(np.linalg.solve(hb.M/hb.period**2, hb.C/hb.period))


def match_eigs(prev_eigvals, prev_eigvecs, new_eigvals, new_eigvecs, alpha=1.0, beta=5.0, T=2*np.pi, n_dof=2):
    """Permutation perm such that new_eigvals[perm] continues prev_eigvals along a sweep.
    Cost: distance between multipliers exp(lambda T) + (1 - eigenvector overlap maximised over harmonic shifts -1, 0, 1),
    so the i m ambiguity of the exponents does not swap branches after a period doubling."""
    n = len(prev_eigvals)
    cost = np.zeros((n, n))
    n_rows = prev_eigvecs.shape[0]//2               # v = [lambda u ; u] -> bottom half is u
    N_H_v = (n_rows//n_dof - 1)//2
    A_prev = complex_harmonics(prev_eigvecs[n_rows:, :], N_H_v, n_dof)
    A_new = complex_harmonics(new_eigvecs[n_rows:, :], N_H_v, n_dof)
    mu_prev, mu_new = np.exp(prev_eigvals*T), np.exp(new_eigvals*T)
    mu_scale = np.max(np.abs(mu_prev)) + 1e-12
    for i in range(n):
        for j in range(n):
            overlap = max(shifted_overlap(A_prev[:, :, i], A_new[:, :, j], m, N_H_v) for m in (-1, 0, 1))
            cost[i, j] = alpha*np.abs(mu_prev[i] - mu_new[j])/mu_scale + beta*(1.0 - overlap)
    return linear_sum_assignment(cost)[1]
