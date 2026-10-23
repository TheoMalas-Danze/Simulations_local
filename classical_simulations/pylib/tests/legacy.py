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
