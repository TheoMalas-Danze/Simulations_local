"""Fast checks of the generic harmonic balance and of the ATS model against the frozen notebook code (a few seconds)."""
import numpy as np

import legacy
from catchaos.params import G0, g2_max, MHz
from catchaos.hill import (HarmonicBalance, orbit, state, embed, fourier_project, monodromy,
                           hill_exponents, liouville_residual)
from catchaos.ats import HB, HB2, full_rhs
from catchaos.seeds import satellite

rng = np.random.default_rng(0)


def random_z(N_H, n=2, scale=0.1):
    # decaying harmonics, like a real orbit
    z = scale*rng.standard_normal(n*(2*N_H + 1))
    return z*np.repeat(0.6**np.arange(2*N_H + 1), n)


def test_matches_legacy_period1():
    for kfac in (1.0, 3.0):
        new, old = HB(16, kfac=kfac), legacy.HB(16, kfac=kfac)
        new.set_s(0.6, 10.0); old.set_s(0.6, 10.0)
        z = random_z(16)
        assert np.allclose(new.L, old.L, rtol=0, atol=1e-15)
        assert np.allclose(new.b_ext, old.b_ext, rtol=0, atol=1e-15)
        assert np.allclose(new.residual(z), old.residual(z), rtol=0, atol=1e-13)
        assert np.allclose(new.jacobian(z), old.jacobian(z), rtol=0, atol=1e-13)


def test_matches_legacy_period2():
    new, old = HB2(32), legacy.HB2(32)
    new.set_s(0.75, 10.0); old.set_s(0.75, 10.0)
    z = random_z(32)
    assert np.allclose(new.L, old.L, rtol=0, atol=1e-15)
    assert np.allclose(new.residual(z), old.residual(z), rtol=0, atol=1e-13)
    assert np.allclose(new.jacobian(z), old.jacobian(z), rtol=0, atol=1e-13)


def test_L_against_time_domain():
    # L z = projection of M X'' + C X' + K X (tau derivatives, basis time tau_d = tau/period), random 3-mode system
    for period in (1, 2, 3):
        n, N_H = 3, 6
        A = rng.standard_normal((3, n, n))
        M, C, K = A[0] @ A[0].T + np.eye(n), A[1], A[2]
        hb = HarmonicBalance(N_H, M, C, K, period=period)
        z = random_z(N_H, n)
        t = np.linspace(0, 2*np.pi, hb.N, endpoint=False)
        k = np.arange(1, N_H + 1)
        X, V, Acc = [], [], []
        for m in range(n):
            zz = z[m::n]
            s, c = zz[1::2], zz[2::2]
            X.append(zz[0]/np.sqrt(2) + np.sin(np.outer(t, k)) @ s + np.cos(np.outer(t, k)) @ c)
            V.append((np.cos(np.outer(t, k)) @ (k*s) - np.sin(np.outer(t, k)) @ (k*c))/period)
            Acc.append(-(np.sin(np.outer(t, k)) @ (k**2*s) + np.cos(np.outer(t, k)) @ (k**2*c))/period**2)
        lhs = M @ np.array(Acc) + C @ np.array(V) + K @ np.array(X)
        assert np.allclose(hb.L @ z, fourier_project(lhs, N_H), rtol=0, atol=1e-12)


def test_jacobian_finite_difference():
    hb = HB2(8)
    hb.set_s(0.7, 9.0)
    z = random_z(8)
    J = hb.jacobian(z)
    h = 1e-6
    for j in range(0, len(z), 5):
        e = np.zeros(len(z)); e[j] = h
        fd = (hb.residual(z + e) - hb.residual(z - e))/(2*h)
        assert np.allclose(J[:, j], fd, rtol=0, atol=1e-8)


def test_fourier_and_embed():
    z1 = random_z(8)
    t = np.linspace(0, 2*np.pi, 256, endpoint=False)
    x = np.array([x for x, _ in orbit(z1, 8, t)])
    assert np.allclose(fourier_project(x, 8), z1, rtol=0, atol=1e-13)
    # a period-1 orbit written in the period-2 basis is the same function of tau
    z2 = embed(z1, 8, 16)
    for tau in (0.0, 0.7, 3.0, 5.5):
        assert np.allclose(state(z1, 8, tau), state(z2, 16, tau, period=2), rtol=0, atol=1e-13)


def test_satellite_period1_and_period2():
    # the 60 MHz satellite at eps_d = 10.12 g2 solves the equations of motion, also in the period-2 basis
    hb = HB(16)
    s = 60/g2_max
    z = satellite(hb, s, 10.12)
    assert np.linalg.norm(hb.residual(z)) < 1e-10
    y0 = state(z, 16, 0.0)
    y1 = legacy.solve_ivp(full_rhs(s, 10.12), (0, 2*np.pi), y0, method="DOP853", rtol=1e-11, atol=1e-13).y[:, -1]
    assert np.linalg.norm(y1 - y0)/np.linalg.norm(y0) < 1e-6
    hb2 = HB2(32)
    hb2.set_s(s, 10.12)
    assert np.linalg.norm(hb2.residual(embed(z, 16, 32))) < 1e-10


def test_monodromy_matches_legacy_and_hill():
    hb = HB(16)
    s = 60/g2_max
    z = satellite(hb, s, 10.12)
    old = legacy.HB(16); old.set_s(s, 10.12)
    mu_new, mu_old = np.sort_complex(monodromy(hb, z)), np.sort_complex(legacy.monodromy(old, z))
    assert np.allclose(mu_new, mu_old, rtol=0, atol=1e-12)
    lam = hill_exponents(hb, z)
    assert abs(liouville_residual(hb, lam)) < 1e-9
    assert np.allclose(np.sort_complex(np.exp(2*np.pi*lam)), mu_new, rtol=0, atol=1e-7)


def test_bit_identical_to_notebooks():
    # story_A_v1 caches its Julia jobs by a hash of their inputs: the library must give the notebooks' numbers to the last bit
    from catchaos.hill import odd_mask, pd_null_vector, strobe
    from catchaos.ats import fourier_seed
    from catchaos.branches import arclength_g2, describe, extended_g2
    from catchaos.seeds import satellite_hb, seed_z
    s = 103.88/g2_max
    a, b = HB(16, r=10.0), legacy.HB(16)
    a.set_s(s); b.set_s(s)
    za = a.solve(seed_z(satellite_hb(s, 10.0), 16))
    zb = b.solve(legacy.seed_z(legacy.satellite_hb(s), 16))
    assert np.array_equal(za, zb)
    assert np.array_equal(monodromy(a, za), legacy.monodromy(b, zb))
    a2, b2 = HB2(32, r=10.0), legacy.HB2(32)
    a2.set_s(s); b2.set_s(s)
    z2a, z2b = a2.newton(embed(za, 16, 32)), legacy.newton(b2, legacy.embed(zb, 16, 32))
    assert np.array_equal(z2a, z2b)
    v, _ = pd_null_vector(a2, z2a, odd_mask(32))
    y0, t0 = np.append(z2a, s*g2_max/100), np.append(v, 0.0)
    Ha, ea = arclength_g2(a2, y0, t0, 10.0, n_max=25)
    Hb, eb = legacy.arclength(b2, y0, t0, n_max=25)
    assert np.array_equal(Ha, Hb) and ea == eb
    Da, Db = describe(Ha[::8], 32, 10.0), legacy.describe(Hb[::8], 32)
    assert all(np.array_equal(Da[k], Db[k]) for k in Db)
    Fa, dFa = extended_g2(a2, 10.0)
    Fb, dFb = legacy.extended(b2)
    assert np.array_equal(dFa(Ha[-1]), dFb(Hb[-1]))
    y = strobe(za, 16)
    assert np.array_equal(fourier_seed(y, s, 10.0, 32), legacy.fourier_seed(y, s, 32))
