"""Evaluate harmonic-balance coefficients z (layout of hb.py) in time, and change of basis."""
import numpy as np


def orbit(z, N_H, t, n=2):
    # [(x, dx/dt), ...] for each mode, t in the basis time tau_d of the coefficients
    k = np.arange(1, N_H + 1)
    out = []
    for m in range(n):
        zz = z[m::n]
        s, c = zz[1::2], zz[2::2]
        x = zz[0]/np.sqrt(2) + np.sin(np.outer(t, k)) @ s + np.cos(np.outer(t, k)) @ c
        v = np.cos(np.outer(t, k)) @ (k*s) - np.sin(np.outer(t, k)) @ (k*c)
        out.append((x, v))
    return out


def state(z, N_H, tau, period=1, n=2):
    # [x_1..x_n, dx_1/dtau..dx_n/dtau] at tau (tau = period * tau_d)
    xv = orbit(z, N_H, np.atleast_1d(tau/period), n)
    return np.array([x[0] for x, _ in xv] + [v[0]/period for _, v in xv])


def strobe(z, N_H, tau=0.0, n=2):
    # period-1 orbit: state at tau
    return state(z, N_H, tau, 1, n)


def memory_phase(z, n=2):
    # phase of the first harmonic of mode 0, in units of pi
    za = z[::n]
    return np.angle(za[2] + 1j*za[1])/np.pi


def odd_content(z, N_H, n=2):
    # norm of the odd harmonics of mode 0: zero on a period-1 orbit written in the period-2 basis
    za = z[::n]
    return np.sqrt(sum(za[2*k - 1]**2 + za[2*k]**2 for k in range(1, N_H + 1, 2)))


def odd_mask(N_H, n=2):
    # coefficients of the odd harmonics (all modes)
    m = np.zeros(n*(2*N_H + 1), bool)
    for k in range(1, N_H + 1, 2):
        m[(2*k - 1)*n:(2*k + 1)*n] = True
    return m


def embed(z1, N1, N2, n=2):
    # period-1 coefficients in the period-2 basis: harmonic j of tau -> harmonic 2j of tau_d
    z2 = np.zeros(n*(2*N2 + 1))
    z2[:n] = z1[:n]
    for j in range(1, min(N1, N2//2) + 1):
        z2[(4*j - 1)*n:4*j*n] = z1[(2*j - 1)*n:2*j*n]
        z2[4*j*n:(4*j + 1)*n] = z1[2*j*n:(2*j + 1)*n]
    return z2


def fourier_project(x, N_H):
    # samples x[m, j] of each mode over one period of tau_d (uniform, endpoint=False) -> coefficients z
    n, N = x.shape
    td, k = np.linspace(0, 2*np.pi, N, endpoint=False), np.arange(1, N_H + 1)
    z = np.zeros(n*(2*N_H + 1))
    for m in range(n):
        zz = np.empty(2*N_H + 1)
        zz[0] = np.sqrt(2)*x[m].mean()
        zz[1::2] = (2/N)*np.sin(np.outer(td, k)).T @ x[m]
        zz[2::2] = (2/N)*np.cos(np.outer(td, k)).T @ x[m]
        z[m::n] = zz
    return z
