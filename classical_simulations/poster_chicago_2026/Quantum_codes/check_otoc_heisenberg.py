"""Check of the Heisenberg-picture method of OTOCS_poster_init_cat.ipynb (otoc_point) on a small truncation.

For any operator A, <psi|A_H(t)|psi> must equal Tr[A rho(t)], with rho(t) from a forward Schrodinger
integration of the GKSL equation. A_H is computed as in otoc_point: interaction frame of H_s, H' = -H_drive,
L' = (b e^{-i Delta u})^dag, L'^dag L' passed explicitly, generator at s - u with the drive phase shifted by r,
one solve per offset r saved at u = r + n tau. The naive forward adjoint equation dX/dt = L_t^dag(X) is
computed too, to show that the check detects a wrong time ordering. Finally, the full operator x_H(t) is
checked on random state pairs: <phi|x_H(t)|chi> = Tr[x Phi(t,0)(|chi><phi|)].

Usage: python check.py eps_0 ratio [time]   ("time" only times one Schrodinger integration)
"""
import sys
import time
from math import factorial

import numpy as np
from scipy.integrate import solve_ivp

# poster parameters (article convention: jump sqrt(kappa) b, kappa/2pi = 96 MHz)
w_a = 25.338776456203686; w_b = 2*w_a
phi_a, phi_b = 0.11, 0.204
E_J = 37.12*2*np.pi*1.5
kappa = 2*np.pi/10.4
eps_0 = float(sys.argv[1]); ratio = float(sys.argv[2])
only_time = len(sys.argv) > 3 and sys.argv[3] == "time"
n_a, n_b = 10, 5
n_sub, n_periods = 8, 3
tau = 2*np.pi/w_b
opts = dict(method="DOP853", rtol=1e-10, atol=1e-12)

g2 = np.sin(eps_0)*E_J*phi_a**2*phi_b
eps_d = ratio*g2
da = np.diag(np.sqrt(np.arange(1, n_a)), 1); db = np.diag(np.sqrt(np.arange(1, n_b)), 1)
a = np.kron(da, np.eye(n_b)); b = np.kron(np.eye(n_a), db)
ad, bd = a.conj().T, b.conj().T
xphi = phi_a*(a+ad) + phi_b*(b+bd)
ev, U = np.linalg.eigh(xphi)
sinm = U @ np.diag(np.sin(ev)) @ U.conj().T
H_s = w_a*ad@a + w_b*bd@b - 2*E_J*np.sin(eps_0)*(sinm - xphi)
H_s = 0.5*(H_s + H_s.conj().T)
Hd_op = b + bd
L = np.sqrt(kappa)*b; LdL = L.conj().T@L
d = n_a*n_b

alpha = np.sqrt(ratio/2)
coh = lambda z: np.array([z**k/np.sqrt(float(factorial(k))) for k in range(n_a)], dtype=complex)*np.exp(-abs(z)**2/2)
cat = coh(1j*alpha) + coh(-1j*alpha); cat /= np.linalg.norm(cat)
psi = np.kron(cat, np.eye(n_b)[0]).astype(complex)
t_grid = tau/n_sub*np.arange(n_periods*n_sub + 1)

def forward(r0, t_eval):
    """Schrodinger evolution r' = -i[H(t), r] + D[L] r of any matrix r0 (not necessarily a density matrix)."""
    def rhs(t, y):
        r = y.reshape(d, d); H = H_s + eps_d*np.cos(w_b*t)*Hd_op
        return (-1j*(H@r - r@H) + L@r@L.conj().T - 0.5*(LdL@r + r@LdL)).ravel()
    s = solve_ivp(rhs, (0, t_eval[-1]), r0.ravel(), t_eval=t_eval, **opts)
    return [s.y[:, k].reshape(d, d) for k in range(len(t_eval))], s.nfev

t0 = time.time()
rhos, nfev = forward(np.outer(psi, psi.conj()), t_grid)
print(f"eps_0 = {eps_0}, ratio = {ratio}, n_a = {n_a}, n_b = {n_b}: one Schrodinger integration over "
      f"{n_periods} periods: {time.time()-t0:.1f} s ({nfev} RHS evaluations)", flush=True)
if only_time:
    sys.exit()

# ---------- the method of otoc_point
lam, V = np.linalg.eigh(H_s); Vd = V.conj().T
bar = lambda op: Vd@op@V
Delta = lam[:, None] - lam[None, :]
frame = lambda u: np.exp(-1j*Delta*u)
bdpb_bar, b_bar, bdb_bar = bar(Hd_op), bar(b), bar(bd@b)

def heisenberg_code(A):
    A_bar = bar(A); out = {}
    for k in range(n_sub):
        r = k*tau/n_sub
        us = r + tau*np.arange(n_periods + 1 if k == 0 else n_periods)
        def rhs(u, y, r=r):
            Y = y.reshape(d, d)
            Hp = -eps_d*np.cos(w_b*(r - u))*bdpb_bar*frame(u)
            Lp = (np.sqrt(kappa)*b_bar*frame(u)).conj().T
            LdLp = kappa*bdb_bar*frame(u)
            return (-1j*(Hp@Y - Y@Hp) + Lp@Y@Lp.conj().T - 0.5*(LdLp@Y + Y@LdLp)).ravel()
        s = solve_ivp(rhs, (0, us[-1]), A_bar.ravel().astype(complex), t_eval=us, **opts)
        for j, u in enumerate(us):
            out[round(u/tau*n_sub)] = V@(s.y[:, j].reshape(d, d)*np.exp(1j*Delta*u))@Vd   # a_H(t), Fock basis
    return [out[j] for j in range(len(t_grid))]

def heisenberg_naive(A):
    def rhs(t, y):
        X = y.reshape(d, d); H = H_s + eps_d*np.cos(w_b*t)*Hd_op
        return (1j*(H@X - X@H) + L.conj().T@X@L - 0.5*(LdL@X + X@LdL)).ravel()
    s = solve_ivp(rhs, (0, t_grid[-1]), A.astype(complex).ravel(), t_eval=t_grid, **opts)
    return [s.y[:, k].reshape(d, d) for k in range(len(t_grid))]

x = (a + ad)/np.sqrt(2); p = 1j*(ad - a)/np.sqrt(2)
x_code = None
for name, A in (("a", a), ("a^dag a", ad@a), ("x_a", x)):
    t0 = time.time()
    ref = np.array([np.trace(A@r) for r in rhos])
    H_code = heisenberg_code(A)
    hc = np.array([psi.conj()@X@psi for X in H_code])
    hn = np.array([psi.conj()@X@psi for X in heisenberg_naive(A)])
    if name == "x_a":
        x_code = H_code
    scale = np.max(np.abs(ref))
    print(f"<{name}>: max|ref| = {scale:.3g}, code method: max rel. dev. {np.max(np.abs(hc-ref))/scale:.1e}, "
          f"naive forward: max rel. dev. {np.max(np.abs(hn-ref))/scale:.1e}   ({time.time()-t0:.0f} s)", flush=True)

# ---------- full operator x_H(t) on random pairs, and the OTOC
rng = np.random.default_rng(1)
devs = []
for _ in range(5):
    phi_s = rng.normal(size=d) + 1j*rng.normal(size=d); phi_s /= np.linalg.norm(phi_s)
    chi_s = rng.normal(size=d) + 1j*rng.normal(size=d); chi_s /= np.linalg.norm(chi_s)
    evolved, _ = forward(np.outer(chi_s, phi_s.conj()), t_grid)
    ref = np.array([np.trace(x@r) for r in evolved])
    got = np.array([phi_s.conj()@X@chi_s for X in x_code])
    devs.append(np.max(np.abs(got-ref))/np.max(np.abs(ref)))
print(f"<phi|x_H(t)|chi> on 5 random pairs: max rel. dev. {max(devs):.1e}")
otoc = np.array([np.linalg.norm((X@p - p@X)@psi)**2 for X in x_code])
print("OTOC (lab-frame x_a) at t = 0, tau, 2 tau, 3 tau:", np.round(otoc[::n_sub], 4))
