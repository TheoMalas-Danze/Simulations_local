"""The autoparametric cat (ATS coupling) in the units of the Hill notebooks.

tau = w_a t, x^lambda = phi x (each mode), state Y = [x_a, x_b, dx_a/dtau, dx_b/dtau]:
    x_a'' + x_a - sqrt2 beta_a' (cos(sqrt2 (x_a + x_b)) - 1) = 0
    x_b''/4 + kappa' x_b' + x_b + sqrt2 eps_d' cos(2 tau) - sqrt2 beta_b' (cos(...) - 1) = 0
beta' = 2 E_J s phi^2/w (s = sin eps_p), eps_d' = eps_d phi_b/w_b, kappa' = kfac kappa/w_b (code convention).
Amplitudes: a = (x_a + i dx_a/dtau)/(sqrt2 phi_a), b = (x_b + i (dx_b/dtau)/2)/(sqrt2 phi_b).
"""
import numpy as np
from scipy.integrate import solve_ivp

from .params import w_a, w_b, phi_a, phi_b, E_J, kappa_b, G0
from .hill.hb import HarmonicBalance
from .hill.orbit import orbit, state, fourier_project

n = 2
M = np.diag([1.0, 0.25])
K = np.eye(2)


def betas(s):
    # (beta_a', beta_b')
    return 2*E_J*s*np.array([phi_a**2/w_a, phi_b**2/w_b])


class HB(HarmonicBalance):
    """Harmonic balance of the ATS model. period = 2 for the period-2 basis (tau_d = tau/2, drive on harmonic 4).
    kappa: code-convention kappa_b (1/ns), multiplied by kfac. r: default eps_d/g2 for set_s."""

    def __init__(self, N_H, period=1, kfac=1.0, kappa=kappa_b, r=None, N=None):
        super().__init__(N_H, M, np.diag([0.0, kfac*kappa/w_b]), K, period, N)
        self.drive_harmonic = 2*period
        self.r = r

    def set_kappa(self, kappa, kfac=1.0):
        self.set_damping(np.diag([0.0, kfac*kappa/w_b]))

    def set_drive(self, s, eps_d):
        # s = sin(eps_p), eps_d in 1/ns
        self.s, self.B = s, betas(s)
        self.b_ext = np.zeros(self.dim)
        self.b_ext[2*self.drive_harmonic*n + 1] = -np.sqrt(2)*eps_d*phi_b/w_b      # cos block of the drive harmonic, mode b

    def set_s(self, s, r=None):
        # eps_d = r g2
        r = self.r if r is None else r
        self.set_drive(s, r*s*G0)

    def set_params(self, eps_p, eps_d):
        self.set_drive(np.sin(eps_p), eps_d)

    def fnl(self, X):
        return -np.sqrt(2)*self.B[:, None]*(np.cos(np.sqrt(2)*(X[0] + X[1])) - 1)[None, :]

    def dfnl(self, X):
        return 2*np.sin(np.sqrt(2)*(X[0] + X[1]))[:, None, None]*np.outer(self.B, [1, 1])[None]


def HB2(N_H, **kw):
    # period-2 basis
    return HB(N_H, period=2, **kw)


def full_rhs(s, r, kfac=1.0, kappa=kappa_b):
    # equations of motion in tau, for solve_ivp
    beta = betas(s)
    eps_d_p, kp = r*s*G0*phi_b/w_b, kfac*kappa/w_b

    def f(t, Y):
        x_a, x_b, v_a, v_b = Y
        g = np.cos(np.sqrt(2)*(x_a + x_b)) - 1
        return [v_a, v_b, -x_a + np.sqrt(2)*beta[0]*g, 4*(-x_b - np.sqrt(2)*eps_d_p*np.cos(2*t) + np.sqrt(2)*beta[1]*g - kp*v_b)]
    return f


def amplitudes(Y):
    # (a, b) from the state [x_a, x_b, dx_a/dtau, dx_b/dtau]
    x_a, x_b, v_a, v_b = Y
    return (x_a + 1j*v_a)/(np.sqrt(2)*phi_a), (x_b + 1j*v_b/2)/(np.sqrt(2)*phi_b)


def state_of_amplitudes(a, b):
    return np.array([np.sqrt(2)*phi_a*a.real, np.sqrt(2)*phi_b*b.real, np.sqrt(2)*phi_a*a.imag, 2*np.sqrt(2)*phi_b*b.imag])


def rotating_frame(z, N_H, tau, period=1):
    # a_R = i a e^{i tau}, b_R = b e^{2 i tau} (limit_cycles.ipynb); tau in units of w_a t
    (x_a, v_a), (x_b, v_b) = orbit(z, N_H, tau/period)
    v_a, v_b = v_a/period, v_b/period                                   # d/dtau
    return 1j*(x_a + 1j*v_a)/(np.sqrt(2)*phi_a)*np.exp(1j*tau), (x_b + 1j*v_b/2)/(np.sqrt(2)*phi_b)*np.exp(2j*tau)


def mean_abs_aR(z, N_H):
    # mean |a_R| over a period-1 orbit
    t = np.linspace(0, 2*np.pi, 400, endpoint=False)
    (x_a, v_a), _ = orbit(z, N_H, t)
    return np.mean(np.abs(x_a + 1j*v_a))/(np.sqrt(2)*phi_a)


def section(z, N_H, period=2):
    # a on the section tau = 0 mod 2 pi, and for the partner orbit (a -> -a, tau = pi mod 2 pi)
    taus = 2*np.pi*np.arange(period)
    amp = lambda Y: (Y[0] + 1j*Y[2])/(np.sqrt(2)*phi_a)
    return (np.array([amp(state(z, N_H, t, period)) for t in taus]),
            np.array([amp(state(z, N_H, t + np.pi, period)) for t in taus]))


def fourier_seed(y_state, s, r, N_H, period=2, N=2048, kfac=1.0):
    # one period 2 pi * period integrated from a state on the section, projected on the basis of that period
    tt = np.linspace(0, 2*np.pi*period, N, endpoint=False)
    sol = solve_ivp(full_rhs(s, r, kfac), (0, 2*np.pi*period), y_state, t_eval=tt, method="DOP853", rtol=1e-11, atol=1e-12)
    return fourier_project(sol.y[:2], N_H)
