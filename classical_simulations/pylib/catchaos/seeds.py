"""Starting points for the harmonic balance: analytic satellite, hard-coded cat and period-2 orbits, seed grids."""
import numpy as np

from .params import w_a, w_b, phi_a, phi_b, E_J, kappa_b, G0, g2_of, g2_J1_of
from .ats import n

# analytic satellite (App. "Satellites", quadratic order)
_yb_grid = np.linspace(1e-3, 3.0, 3000)


def satellite_hb(s, r, kfac=1.0):
    # complex amplitudes of the analytic satellite (first root found), or None if it does not exist
    kp = kfac*kappa_b/w_b
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

    rv = resid(_yb_grid)
    for k in np.where(rv[:-1]*rv[1:] <= 0)[0]:
        lo, hi = _yb_grid[k], _yb_grid[k + 1]
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
    return None


def seed_z(S, N_H):
    # analytic satellite amplitudes -> period-1 HB coefficients
    za, zb = np.zeros(2*N_H + 1), np.zeros(2*N_H + 1)
    za[0] = 2*np.real(S["a1"])
    za[1], za[2] = np.sqrt(2)*np.imag(S["a0"]), np.sqrt(2)*np.real(S["a0"])
    za[3], za[4] = np.sqrt(2)*np.imag(S["am1"]), np.sqrt(2)*np.real(S["am1"])
    zb[0] = 2*np.real(S["b2"])
    zb[3], zb[4] = np.sqrt(2)*np.imag(S["b0"]), np.sqrt(2)*np.real(S["b0"])
    zb[5], zb[6] = np.sqrt(2)*np.imag(S["bm1"]), np.sqrt(2)*np.real(S["bm1"])
    z = np.zeros(n*(2*N_H + 1))
    z[::2], z[1::2] = za, zb
    return z


def satellite(hb, s, r, kfac=1.0):
    # exact satellite at (s, r), converged from the analytic one
    hb.set_s(s, r)
    return hb.solve(seed_z(satellite_hb(s, r, kfac), hb.N_H))


# hard-coded orbits converged with the former J1 definition of g2 (eps_d = 10.12 g2_J1)
Z_CAT_J1 = np.array([-2.09501523e-02, -3.60273363e-02,  3.26246220e-01, -3.28386996e-17,
       -6.78344780e-03,  1.43507376e-17, -1.72514305e-04, -3.13979476e-03,
       -4.23320965e-03, -7.46552651e-02, -8.01435361e-04, -8.82665625e-03,
        5.05426053e-05,  4.29993690e-04,  7.79954440e-06,  6.01698525e-05,
        1.00817384e-04,  8.67341068e-04])   # period-1 cat at eps_p = 0.5 (Hill_method_cat_fold)

Z_P2_J1 = np.array([-8.67480681e-02, -1.49178000e-01, -1.55487349e-04, -3.92476662e-06,
        4.80123540e-02,  6.60523141e-02, -3.00220194e-02,  3.06241505e-16,
        3.31363194e-01, -1.81473366e-16, -3.88737312e-03,  1.57262610e-02,
        3.35330370e-02, -1.65080648e-01,  6.93437379e-04, -5.59462961e-02,
       -4.67601081e-03,  3.00083819e-01,  4.52017869e-05,  9.32575051e-04,
       -4.88538089e-04, -7.81645147e-03, -1.26778042e-03, -1.46772441e-02,
        4.61868963e-03,  5.06227939e-02,  2.68514538e-04,  2.61445320e-03,
       -1.01232565e-03, -9.46918403e-03, -5.89963411e-04, -5.17649744e-03,
        1.52354566e-03,  1.30588333e-02])   # period-2 orbit (period-2 basis) at eps_p = 1.2, kappa_b = pi/10.4


def _from_J1(hb, z, eps_p, r0=10.12, steps=25):
    # continue a J1 seed in eps_d from r0 g2_J1 to r0 g2 (sin definition) at fixed eps_p
    z = hb.pad(z)
    for eps_d in np.linspace(r0*g2_J1_of(eps_p), r0*g2_of(eps_p), steps):
        hb.set_params(eps_p, eps_d)
        z = hb.solve(z)
    return z


def cat_seed(hb, eps_p=0.5, r0=10.12):
    # period-1 cat at (eps_p, eps_d = r0 g2)
    return _from_J1(hb, Z_CAT_J1, eps_p, r0)


def period2_seed(hb2, eps_p=1.2, r0=10.12):
    # period-2 orbit at (eps_p, eps_d = r0 g2); hb2 in the period-2 basis
    return _from_J1(hb2, Z_P2_J1, eps_p, r0)


def grid_state(grid, key, j, i):
    # state [x_a, x_b, dx_a/dtau, dx_b/dtau] of a seed-grid JSON (satellite_seed_hill, cat_seed_hill), None outside the branch
    v = grid[key][j][i]
    return None if v is None else np.array(v)
