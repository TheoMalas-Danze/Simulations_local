"""Frozen copy of the notebook code that catchaos replaces (crisis_collision.ipynb cell 3, 23/10/2026), as a reference.
Do not edit: the tests check that catchaos reproduces it."""
import numpy as np
from scipy.optimize import root
from scipy.integrate import solve_ivp

w_a = 25.338776456203686
w_b = 2*w_a
phi_a, phi_b = 0.11, 0.204
E_J = 37.12*2*np.pi*1.5
kappa_b = np.pi/10.4
G0 = E_J*phi_a**2*phi_b
R = 10.0
n = 2
M = np.diag([1.0, 0.25])
K = np.eye(2)


class HB:
    def __init__(self, N_H, kfac=1.0, N=None):
        self.N_H, self.N = N_H, N or 64*N_H
        self.C = np.diag([0.0, kfac*kappa_b/w_b])
        self.L = self.build_L(M, self.C)
        t = np.linspace(0, 2*np.pi, self.N, endpoint=False)
        k = np.arange(1, N_H + 1)
        Phi = np.empty((self.N, 2*N_H + 1))
        Phi[:, 0] = 1/np.sqrt(2)
        Phi[:, 1::2] = np.sin(np.outer(t, k))
        Phi[:, 2::2] = np.cos(np.outer(t, k))
        self.Gamma = np.kron(Phi, np.eye(n))
        self.Gamma_pinv = (2/self.N)*self.Gamma.T
        self.drive_harmonic = 2

    def build_L(self, M_, C_):
        L = np.zeros((n*(2*self.N_H + 1),)*2)
        L[:n, :n] = K
        for k in range(1, self.N_H + 1):
            A, Bk = K - k**2*M_, k*C_
            s_, c_ = 2*k - 1, 2*k
            L[s_*n:(s_+1)*n, s_*n:(s_+1)*n] = A
            L[s_*n:(s_+1)*n, c_*n:(c_+1)*n] = -Bk
            L[c_*n:(c_+1)*n, s_*n:(s_+1)*n] = Bk
            L[c_*n:(c_+1)*n, c_*n:(c_+1)*n] = A
        return L

    def set_s(self, s, r=R):
        self.B = 2*E_J*s*np.array([phi_a**2/w_a, phi_b**2/w_b])
        self.b_ext = np.zeros(n*(2*self.N_H + 1))
        self.b_ext[2*self.drive_harmonic*n + 1] = -np.sqrt(2)*r*s*G0*phi_b/w_b

    def X(self, z):
        return (self.Gamma @ z).reshape(self.N, n).T

    def residual(self, z):
        X = self.X(z)
        Fnl = -np.sqrt(2)*self.B[:, None]*(np.cos(np.sqrt(2)*(X[0] + X[1])) - 1)[None, :]
        return self.L @ z + self.Gamma_pinv @ Fnl.T.reshape(-1) - self.b_ext

    def jacobian(self, z):
        X = self.X(z)
        sn = np.sin(np.sqrt(2)*(X[0] + X[1]))
        JG = np.einsum('j,ab,jbm->jam', 2*sn, np.outer(self.B, [1, 1]),
                       self.Gamma.reshape(self.N, n, -1)).reshape(n*self.N, -1)
        return self.L + self.Gamma_pinv @ JG

    def solve(self, z0, tol=1e-12):
        sol = root(self.residual, z0, jac=self.jacobian, method="hybr", tol=tol, options={"maxfev": 100*(len(z0) + 1)})
        if not sol.success:
            raise RuntimeError(sol.message)
        return sol.x


class HB2(HB):
    def __init__(self, N_H, kfac=1.0, N=None):
        super().__init__(N_H, kfac, N)
        self.L = self.build_L(M/4, self.C/2)
        self.drive_harmonic = 4


def orbit(z, N_H, t):
    k = np.arange(1, N_H + 1)
    out = []
    for zz in (z[::2], z[1::2]):
        s_, c_ = zz[1::2], zz[2::2]
        x = zz[0]/np.sqrt(2) + np.sin(np.outer(t, k)) @ s_ + np.cos(np.outer(t, k)) @ c_
        v = np.cos(np.outer(t, k)) @ (k*s_) - np.sin(np.outer(t, k)) @ (k*c_)
        out.append((x, v))
    return out


def monodromy(hb, z, period=1):
    Minv, base = np.linalg.inv(M), np.outer(hb.B, [1, 1])
    def rhs(t, Y):
        (x_a, _), (x_b, _) = orbit(z, hb.N_H, np.atleast_1d(t)/period)
        A = np.zeros((4, 4)); A[:2, 2:] = np.eye(2)
        A[2:, :2] = -Minv @ (K + 2*np.sin(np.sqrt(2)*(x_a[0] + x_b[0]))*base); A[2:, 2:] = -Minv @ hb.C
        return (A @ Y.reshape(4, 4)).reshape(-1)
    sol = solve_ivp(rhs, (0, 2*np.pi*period), np.eye(4).reshape(-1), method="DOP853", rtol=1e-10, atol=1e-12)
    return np.linalg.eigvals(sol.y[:, -1].reshape(4, 4))


def full_rhs(s, kfac=1.0, r=R):
    beta = 2*E_J*s*np.array([phi_a**2/w_a, phi_b**2/w_b])
    eps_d_p, kp = r*s*G0*phi_b/w_b, kfac*kappa_b/w_b
    def f(t, Y):
        x_a, x_b, v_a, v_b = Y
        g = np.cos(np.sqrt(2)*(x_a + x_b)) - 1
        return [v_a, v_b, -x_a + np.sqrt(2)*beta[0]*g, 4*(-x_b - np.sqrt(2)*eps_d_p*np.cos(2*t) + np.sqrt(2)*beta[1]*g - kp*v_b)]
    return f


# ---- Hill_period2_branch_II.ipynb cells 3 (rest) and 4, frozen 23/10/2026
N_H, N_H2 = 16, 32

def MHz(g):
    return g/2/np.pi*1e3
g2_max = MHz(G0)
yb_grid = np.linspace(1e-3, 3.0, 3000)

def strobe(z, N_H, tau=0.0):
    # period-1 orbit: state [x_a, x_b, v_a, v_b] at time tau
    (x_a, v_a), (x_b, v_b) = orbit(z, N_H, np.array([tau]))
    return np.array([x_a[0], x_b[0], v_a[0], v_b[0]])


def rotating_frame(z, N_H, tau, period=1):
    # a_R = i a e^{i tau}, b_R = b e^{2 i tau} (limit_cycles.ipynb); tau in units of omega_a t
    (x_a, v_a), (x_b, v_b) = orbit(z, N_H, tau/period)
    v_a, v_b = v_a/period, v_b/period                                   # d/dtau
    return 1j*(x_a + 1j*v_a)/(np.sqrt(2)*phi_a)*np.exp(1j*tau), (x_b + 1j*v_b/2)/(np.sqrt(2)*phi_b)*np.exp(2j*tau)


def odd_content(z, N_H):
    # odd harmonics of the memory in the period-2 basis: zero on a period-1 orbit
    za = z[::2]
    return np.sqrt(sum(za[2*k - 1]**2 + za[2*k]**2 for k in range(1, N_H + 1, 2)))


def instability(mu):
    # type of the leading multiplier
    m = mu[np.argmax(np.abs(mu))]
    if abs(m) < 1:
        return "stable"
    if abs(m.imag) > 1e-6:
        return "NS"
    return "PD" if m.real < 0 else "saddle"


def satellite_hb(s, r=R):
    kp = kappa_b/w_b
    ba, bb = 2*E_J*s*phi_a**2/w_a, 2*E_J*s*phi_b**2/w_b
    ed, bS = r*s*G0*phi_b/w_b, ba + bb
    gam = 1 + bb/(ba*(1 + kp**2))
    eeff = lambda y: np.sqrt(np.maximum(ed**2 - 4*kp**2*y**2, 0.0))
    def y_of(yb):
        y = yb + ba*ed/(2*bb)
        for _ in range(30):
            y = yb + ba*eeff(y)/(2*bb)
        return y
    def u_of(yb):
        eta = 2*bb*yb
        return (np.sqrt((1 + eta)**2 + 4*eta**2) - (1 + eta))/(2*eta)
    def x2_of(yb):
        u = u_of(yb)
        return (2*(1 + u)*yb**2 - eeff(y_of(yb))/bb)/(1 + 2*u)
    def resid(yb):
        u = u_of(yb)
        return yb*(1 + u) - 2*bS*((1 + (1 + u)**2/2)*yb**2 + (1 + u**2)*x2_of(yb))
    rv = resid(yb_grid)
    for k in np.where(rv[:-1]*rv[1:] <= 0)[0]:
        lo, hi = yb_grid[k], yb_grid[k + 1]
        for _ in range(60):
            mid = (lo + hi)/2
            if resid(lo)*resid(mid) <= 0:
                hi = mid
            else:
                lo = mid
        yb = (lo + hi)/2
        x2 = x2_of(yb)
        if x2 <= 0:
            continue
        u, y, x = u_of(yb), y_of(yb), np.sqrt(x2)
        if 2*kp*abs(y) > ed:
            continue
        th = -np.arcsin(2*kp*y/ed)/2
        a1 = -(1 + u)*yb/(2*gam)
        return dict(a0=x*np.exp(1j*th), a1=a1, am1=-ba*eeff(y)/(2*bb)*np.exp(2j*th),
                    b0=y*np.exp(2j*th), bm1=u*x*np.exp(3j*th), b2=bb/ba*a1)


def seed_z(S, N_H):
    za, zb = np.zeros(2*N_H + 1), np.zeros(2*N_H + 1)
    za[0] = 2*np.real(S["a1"])
    za[1], za[2] = np.sqrt(2)*np.imag(S["a0"]), np.sqrt(2)*np.real(S["a0"])
    za[3], za[4] = np.sqrt(2)*np.imag(S["am1"]), np.sqrt(2)*np.real(S["am1"])
    zb[0] = 2*np.real(S["b2"])
    zb[3], zb[4] = np.sqrt(2)*np.imag(S["b0"]), np.sqrt(2)*np.real(S["b0"])
    zb[5], zb[6] = np.sqrt(2)*np.imag(S["bm1"]), np.sqrt(2)*np.real(S["bm1"])
    z = np.zeros(n*(2*N_H + 1)); z[::2], z[1::2] = za, zb
    return z


def newton(hb, z, tol=1e-11, maxit=30):
    for _ in range(maxit):
        r_ = hb.residual(z)
        if np.linalg.norm(r_) < tol:
            return z
        z = z - np.linalg.solve(hb.jacobian(z), r_)
    if np.linalg.norm(hb.residual(z)) < 1e3*tol:
        return z
    raise RuntimeError("Newton did not converge")

def embed(z1, N1, N2):
    # period-1 coefficients in the period-2 basis: harmonic j of tau -> harmonic 2j of tau_d
    z2 = np.zeros(n*(2*N2 + 1)); z2[:n] = z1[:n]
    for j in range(1, min(N1, N2//2) + 1):
        z2[(4*j - 1)*n:4*j*n] = z1[(2*j - 1)*n:2*j*n]
        z2[4*j*n:(4*j + 1)*n] = z1[2*j*n:(2*j + 1)*n]
    return z2

ODD = np.zeros(n*(2*N_H2 + 1), bool)                    # coefficients of the odd harmonics (both modes)
for k in range(1, N_H2 + 1, 2):
    ODD[(2*k - 1)*n:(2*k + 1)*n] = True

PS = 100.0                                              # continuation parameter p = g2 / (100 MHz)
def extended(hb2):
    def setp(p): hb2.set_s(p*PS/g2_max)
    def F(y):
        setp(y[-1]); return hb2.residual(y[:-1])
    def dF(y, h=1e-7):
        setp(y[-1]); Jz, r_ = hb2.jacobian(y[:-1]), hb2.residual(y[:-1])
        setp(y[-1] + h)
        return np.column_stack([Jz, (hb2.residual(y[:-1]) - r_)/h])
    return F, dF

def arclength(hb2, y, t, ds0=2e-3, ds_max=3e-3, g_lo=60.0, g_hi=140.0, n_max=5000, stop_odd=True):
    # pseudo-arclength from y = (z, p) along t; stops when g2 leaves [g_lo, g_hi] or (stop_odd) the odd harmonics vanish again
    F, dF = extended(hb2)
    def corrector(y0, t, ds):
        w = y0 + ds*t
        for _ in range(20):
            G = np.append(F(w), t @ (w - y0) - ds)
            if np.linalg.norm(G) < 1e-11:
                return w
            w = w - np.linalg.solve(np.vstack([dF(w), t]), G)
        if np.linalg.norm(np.append(F(w), t @ (w - y0) - ds)) < 1e-9:
            return w
        raise RuntimeError
    ds, hist, end = ds0, [y.copy()], "n_max reached"
    for it in range(n_max):
        try:
            yn = corrector(y, t, ds)
        except (RuntimeError, np.linalg.LinAlgError):
            ds /= 2
            if ds < 1e-8:
                end = f"step too small at {y[-1]*PS:.2f} MHz"; break
            continue
        tn = np.linalg.svd(dF(yn))[2][-1]
        t = -tn if tn @ t < 0 else tn
        y = yn; hist.append(y.copy()); ds = min(1.3*ds, ds_max)
        g = y[-1]*PS
        if not g_lo < g < g_hi:
            end = f"leaves [{g_lo:g}, {g_hi:g}] MHz"; break
        if stop_odd and it > 20 and odd_content(y[:-1], hb2.N_H) < 1e-5:
            end = f"odd harmonics vanish at {g:.3f} MHz"; break
    return np.array(hist), end

def fourier_seed(y_state, s, N_H2, N=2048):
    # one period 2 T_a from a state on the section, projected on the period-2 basis (tau_d = tau/2)
    tt = np.linspace(0, 4*np.pi, N, endpoint=False)
    sol = solve_ivp(full_rhs(s), (0, 4*np.pi), y_state, t_eval=tt, method="DOP853", rtol=1e-11, atol=1e-12)
    td, k = tt/2, np.arange(1, N_H2 + 1)
    z = np.zeros(n*(2*N_H2 + 1))
    for m in range(2):
        zz = np.empty(2*N_H2 + 1)
        zz[0] = np.sqrt(2)*sol.y[m].mean()
        zz[1::2] = (2/N)*np.sin(np.outer(td, k)).T @ sol.y[m]
        zz[2::2] = (2/N)*np.cos(np.outer(td, k)).T @ sol.y[m]
        z[m::2] = zz
    return z

def state(z, N, tau, period):
    # [x_a, x_b, dx_a/dtau, dx_b/dtau] at tau (x^lambda units of the harmonic balance)
    (x_a, v_a), (x_b, v_b) = orbit(z, N, np.atleast_1d(tau/period))
    return np.array([x_a[0], x_b[0], v_a[0]/period, v_b[0]/period])

def section(z, N, period=2):
    # a on the section tau = 0 mod 2 pi (amp of bubble_crisis_lifetime.jl), and for the partner orbit (tau = pi mod 2 pi)
    taus = 2*np.pi*np.arange(period)
    amp = lambda Y: (Y[0] + 1j*Y[2])/(np.sqrt(2)*phi_a)
    return np.array([amp(state(z, N, t, period)) for t in taus]), np.array([amp(state(z, N, t + np.pi, period)) for t in taus])

def describe(H, N, period=2):
    # multipliers, type, odd content, section, <|a_R|^2> at every point of a branch H (rows (z, p))
    hb_ = HB2(N) if period == 2 else HB(N)
    tt = np.linspace(0, 2*np.pi*period, 300*period, endpoint=False)
    out = dict(g2=H[:, -1]*PS, mu=[], kind=[], odd=[], sec=[], sec_partner=[], a2=[])
    for y in H:
        hb_.set_s(y[-1]*PS/g2_max)
        mu = monodromy(hb_, y[:-1], period); mu = mu[np.argsort(-np.abs(mu))]
        s0, s1 = section(y[:-1], N, period)
        out["mu"].append(mu); out["kind"].append(instability(mu)); out["odd"].append(odd_content(y[:-1], N) if period == 2 else 0.0)
        out["sec"].append(s0); out["sec_partner"].append(s1)
        out["a2"].append(np.mean(np.abs(rotating_frame(y[:-1], N, tt, period)[0])**2))
    return {k: np.array(v) for k, v in out.items()}

def unstable_set(mu):
    # multipliers outside the unit circle: "+" real > 1, "-" real < -1, "c" complex pair (counted once)
    out = [("c" if abs(m.imag) > 1e-6 else ("+" if m.real > 0 else "-")) for m in mu if abs(m) > 1 and m.imag >= -1e-6]
    return "".join(sorted(out)) or "stable"

def transitions(D):
    # changes of the set of unstable multipliers along a branch; g2 of the crossing interpolated on the multiplier that crosses |mu| = 1
    res, sets = [], [unstable_set(m) for m in D["mu"]]
    for i in range(1, len(sets)):
        if sets[i] != sets[i - 1]:
            a, b = np.sort(np.abs(D["mu"][i - 1])), np.sort(np.abs(D["mu"][i]))
            k = np.argmax(np.abs((a > 1).astype(int) - (b > 1).astype(int))) if np.any((a > 1) != (b > 1)) else None
            g = D["g2"][i - 1] + (1 - a[k])*(D["g2"][i] - D["g2"][i - 1])/(b[k] - a[k]) if k is not None else D["g2"][i]
            res.append((g, sets[i - 1], sets[i]))
    return res
