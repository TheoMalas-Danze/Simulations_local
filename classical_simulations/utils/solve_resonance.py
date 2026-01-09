"""$$K(t)=(\delta + \delta') a^\dagger a + (\Delta + \Delta') b^\dagger b + ...$$

with $\delta'=12g_{1,1}g_{3,1}(\frac{r^2}{\omega_d+\omega_p} + \frac{2}{\omega_d+3\omega_p} + \frac{r^2+2}{\omega_d-\omega_p})$

and $\Delta'=12g_{1,1}g_{3,1}r^2(\frac{r^2}{\omega_d+\omega_p} + \frac{2}{\omega_d+3\omega_p} + \frac{r^2+2}{\omega_d-\omega_p})$

So that we need to solve (to work on resonance)

$$\omega_a - \frac{\omega_d +\omega_p}{2}+\delta' = 0 \\
\omega_b - \omega_d + \Delta' = 0$$"""


def solve_resonance(e_p_values, w_a, w_b, varphi_a, varphi_b, E_J):
    import sympy as sp
    import numpy as np

    # ------------------------------------------------------------------
    # Symbols
    # ------------------------------------------------------------------
    w_d, w_p = sp.symbols("w_d w_p", real=True)
    e_p = sp.symbols("e_p", real=True)

    # ------------------------------------------------------------------
    # Numerical constants (keep as floats)
    # ------------------------------------------------------------------
    r = varphi_b / varphi_a

    # ------------------------------------------------------------------
    # Use SymPy Bessel function (NOT scipy.special.j1)
    # ------------------------------------------------------------------
    J1 = sp.besselj(1, e_p)

    # ------------------------------------------------------------------
    # Expressions
    # ------------------------------------------------------------------
    g_1_1_g_3_1 = (
        sp.Rational(2, 3)
        * E_J**2
        * sp.exp(-varphi_a**2 - varphi_b**2)
        * J1**2
        * varphi_a**4
    )

    delta_p = 12 * g_1_1_g_3_1 * (
        r**2 / (w_d + w_p)
        + 2 / (w_d + 3*w_p)
        + (r**2 + 2) / (w_d - w_p)
    )

    Delta_p = r**2 * delta_p
    f1 = sp.simplify(w_a - (w_d + w_p)/2 + delta_p)
    f2 = sp.simplify(w_b - w_d + Delta_p)
    
    wd_guess = float(w_b)
    wp_guess = float(2*w_a - w_b)

    w_d_val_list = []
    w_p_val_list = []

    for ep in e_p_values:
        # substitute ep, then solve numerically
        try:
            sol = sp.nsolve(
                [f1.subs(e_p, ep), f2.subs(e_p, ep)],
                [w_d, w_p],
                [wd_guess, wp_guess],
                tol=1e-14,
                maxsteps=80,
                prec=50
            )
            wd_val = float(sol[0])
            wp_val = float(sol[1])

            w_d_val_list.append(wd_val)
            w_p_val_list.append(wp_val)

            # continuation: use solution as next initial guess
            wd_guess, wp_guess = wd_val, wp_val

        except Exception as err:
            # keep going even if one point fails
            w_d_val_list.append(np.nan)
            w_p_val_list.append(np.nan)
            # (optional) reset guesses if failure happens
            wd_guess = float(w_b)
            wp_guess = float(2*w_a - w_b)
    return {
    "w_d": w_d_val_list,
    "w_p": w_p_val_list
    }