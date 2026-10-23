"""Harmonic balance with an exact one-period alternating frequency-time (AFT) scheme, for any number of modes.

Equation, in the dimensionless time tau:
    M X'' + C X' + K X + f_nl(X) = f_ext(tau)
Periodic orbit of period 2 pi * period in tau. Basis time tau_d = tau/period, so the operator uses M/period^2 and C/period.

Unknowns z = [c0, s1, c1, ..., s_NH, c_NH], each block of length n (one entry per mode, interleaved):
    X(tau_d) = c0/sqrt2 + sum_k s_k sin(k tau_d) + c_k cos(k tau_d)
So z[m::n] is mode m. One period of tau_d is sampled uniformly (endpoint=False), so Gamma^+ = (2/N) Gamma^T exactly.

Subclasses define the nonlinearity fnl(X) -> (n, N), its derivative dfnl(X) -> (N, n, n), and fill self.b_ext.
"""
import numpy as np
from scipy.optimize import root


def build_L(N_H, M, C, K):
    # linear operator: K on c0; K - k^2 M on the diagonal of harmonic k, -/+ k C between its sin and cos blocks
    n = len(M)
    L = np.zeros((n*(2*N_H + 1),)*2)
    L[:n, :n] = K
    for k in range(1, N_H + 1):
        A, Bk = K - k**2*M, k*C
        s, c = 2*k - 1, 2*k
        L[s*n:(s+1)*n, s*n:(s+1)*n] = A
        L[s*n:(s+1)*n, c*n:(c+1)*n] = -Bk
        L[c*n:(c+1)*n, s*n:(s+1)*n] = Bk
        L[c*n:(c+1)*n, c*n:(c+1)*n] = A
    return L


def build_Gamma(t, N_H, n):
    # X(t_j) = Gamma z, time-major stacking
    k = np.arange(1, N_H + 1)
    Phi = np.empty((t.size, 2*N_H + 1))
    Phi[:, 0] = 1/np.sqrt(2)
    Phi[:, 1::2] = np.sin(np.outer(t, k))
    Phi[:, 2::2] = np.cos(np.outer(t, k))
    return np.kron(Phi, np.eye(n))


class HarmonicBalance:
    def __init__(self, N_H, M, C, K, period=1, N=None):
        self.N_H, self.N, self.period, self.n = N_H, N or 64*N_H, period, len(M)
        self.M, self.K = np.asarray(M, float), np.asarray(K, float)
        self.set_damping(C)
        self.Gamma = build_Gamma(np.linspace(0, 2*np.pi, self.N, endpoint=False), N_H, self.n)
        self.Gamma_pinv = (2/self.N)*self.Gamma.T
        self.b_ext = np.zeros(self.dim)

    @property
    def dim(self):
        return self.n*(2*self.N_H + 1)

    def set_damping(self, C):
        # C in tau units (also used by the monodromy matrix); the operator L is in the basis time tau_d
        self.C = np.asarray(C, float)
        self.L = build_L(self.N_H, self.M/self.period**2, self.C/self.period, self.K)

    def fnl(self, X):
        raise NotImplementedError

    def dfnl(self, X):
        raise NotImplementedError

    def X(self, z):
        return (self.Gamma @ z).reshape(self.N, self.n).T

    def residual(self, z):
        return self.L @ z + self.Gamma_pinv @ self.fnl(self.X(z)).T.reshape(-1) - self.b_ext

    def jacobian(self, z):
        JG = np.einsum('jab,jbm->jam', self.dfnl(self.X(z)),
                       self.Gamma.reshape(self.N, self.n, -1)).reshape(self.n*self.N, -1)
        return self.L + self.Gamma_pinv @ JG

    def solve(self, z0, tol=1e-12):
        # Powell hybrid with the analytic Jacobian; raises RuntimeError if it does not converge
        sol = root(self.residual, z0, jac=self.jacobian, method="hybr", tol=tol, options={"maxfev": 100*(len(z0) + 1)})
        if not sol.success:
            raise RuntimeError(sol.message)
        return sol.x

    def solve_robust(self, z):
        # Powell sometimes stalls at tol = 1e-12 when the guess is already converged to ~1e-8
        try:
            return self.solve(z)
        except RuntimeError:
            return self.solve(z, tol=1e-9)

    def newton(self, z, tol=1e-11, maxit=30):
        # plain Newton corrector (hybr stalls when the guess is already converged, e.g. during a bisection)
        for _ in range(maxit):
            r_ = self.residual(z)
            if np.linalg.norm(r_) < tol:
                return z
            z = z - np.linalg.solve(self.jacobian(z), r_)
        if np.linalg.norm(self.residual(z)) < 1e3*tol:
            return z
        raise RuntimeError("Newton did not converge")

    def pad(self, z):
        # zero-pad (or truncate) coefficients from a smaller (larger) N_H
        z = np.asarray(z, float)[:self.dim]
        return np.pad(z, (0, self.dim - len(z)))
