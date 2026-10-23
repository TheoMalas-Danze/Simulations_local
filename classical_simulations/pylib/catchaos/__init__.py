"""Shared Python code of classical_simulations (autoparametric cat, classical dynamics).

In a notebook:
    import sys; sys.path.insert(0, "/Users/tmalas/Desktop/Dev/Simulations_local/classical_simulations/pylib")
    from catchaos.params import *                 # w_a, phi_a, E_J, kappa_b, G0, g2_of, MHz, ...
    from catchaos.ats import HB, HB2, full_rhs, rotating_frame, amplitudes
    from catchaos.hill import monodromy, instability, branch_on_grid, ...
    from catchaos.paths import DATA, FIGS, dated, latest   # DATA/"hill_orbits", FIGS/"torus_crisis", ...
    from catchaos.jsonio import save_json, load_json
"""
