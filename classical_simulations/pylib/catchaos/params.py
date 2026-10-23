"""Standard circuit parameters (classical_simulations/CLAUDE.md) and unit helpers.

Conventions:
    g2 = E_J sin(eps_p) phi_a^2 phi_b = s G0, with s = sin(eps_p) (since 2026-10-12; g2_J1_of is the former definition)
    eps_d = r g2, with r = eps_d/g2
    kappa_b = pi/10.4 in the code (Julia) convention phi_b'' = ... - 2 kappa_b phi_b'; the GKSL kappa_b is 2 kappa_b (96 MHz)
"""
import numpy as np
from scipy.special import jv

w_a = 25.338776456203686             # rad/ns
w_b = 2*w_a
phi_a, phi_b = 0.11, 0.204
E_J = 37.12*2*np.pi*1.5
kappa_b = np.pi/10.4                 # 1/ns, code convention
G0 = E_J*phi_a**2*phi_b              # g2 = s G0


def g2_of(eps_p):
    return np.sin(eps_p)*G0


def g2_J1_of(eps_p):
    # former definition (oscillating pump), only used to continue the hard-coded seeds
    return jv(1, eps_p)*G0


def MHz(x):
    # angular rate (1/ns) -> x/2pi in MHz
    return x/2/np.pi*1e3


g2_max = MHz(G0)                     # g2/2pi at sin(eps_p) = 1


def s_of_MHz(g):
    # sin(eps_p) for g2/2pi = g MHz
    return g/g2_max
