# classical_simulations — context for Claude agents

Owner: Théo Malas-Danzé (Alice & Bob). Research topic: the **autoparametric dissipative cat qubit**.
A memory mode `a` (ω_a) and a buffer mode `b` (ω_b ≈ 2ω_a) are coupled through an ATS-like Josephson
nonlinearity `cos(φ_a(a+a†) + φ_b(b+b†))`. A buffer drive at ω_d = ω_b, with a pump at ω_p = 2ω_a − ω_d,
gives the two-photon exchange g2. The buffer is lossy (κ_b) and the memory is lossless (κ_a = 0 by default).
This folder studies the **classical (mean-field) dynamics** and how they **become chaotic** as the pump
amplitude ε_p (i.e. g2) grows. The route goes through cat fixed points (limit cycles in the lab frame),
then "satellite" orbits and period doubling, then chaos.

Most notebooks are Julia 1.11. The Hill/harmonic-balance and plotting notebooks are Python.
None of the notebooks are scripts: they are run interactively, cells often out of order, and outputs are saved in place.
Several notebooks exceed 100 MB because of saved outputs, so **never `cat`/`Read` a whole `.ipynb`**.
Extract cell sources with python/jq and skip outputs.

## Standard parameters (used in almost every notebook)
```
ω_a = 25.338776456203686 (rad/ns, ≈ 2π·4.03 GHz)   ω_b = 2ω_a
φ_a = 0.11   φ_b = 0.204      (ZPF; Z = 2·(1/2)²·φ², ϕ_quantum = 1/2, ħ = e = 1)
E_J = 37.12·2π·1.5
ω_d = ω_b,  ω_p = 2ω_a − ω_d = 0   ← "no pump frequency" regime: the pump is a static flux, factor sin(ε_p)
g2  = sin(ε_p)·E_J·φ_a²·φ_b     ← since 2026-10-12, static pump (ω_p = 0), as in the article. Older notebooks/outputs use
                                 J1(ε_p)·E_J·φ_a²·φ_b ("Leon's paper"), about 2× smaller. See "g2 convention" below.
ε_d = ratio·g2,  ratio ∈ [2, 13.5] for the 12_10_2026 maps (was [5, 13.5]; typical values 8.12, 10, 10.12);  |α|² = ratio/2
κ_b = π/10.4   ← decided 2026-10-08, Julia convention φ̈_b = … − 2κ_b φ̇_b (older notebooks may still use 1/10.4)
```
Typical sweep: ε_p (0.2–1.1) × ε_d/g2. Plots are usually against g2 in MHz (`g2/2π·1e3`) and α² or ε_d/2g2.

## Julia core library (`config/`, `utils/`, `visualization/`, `simulator.jl`)
Every file uses **absolute `include` paths** (`/Users/tmalas/Desktop/Dev/Simulations_local/...`).
- Entry point for notebooks: `include(".../config/requirements.jl")`. It loads Bessels, `simulator.jl`, the plot settings and `baths.jl`.
- `simulator(t_max, system, drive, env; equation=default_equation!, solver=default_solver, kwargs...)`
  - Calls `prepare(system)`, `prepare(drive)` and `prepare(system, env)`, then `get_dt`, `get_init`, then `SimulationParams(...)`, then the solver.
- **`config/systems.jl`**: the systems.
  - `SystemATS(ω_a,ω_b,E_J,E_Ca,E_Cb,ϕ_a0,q_a0,ϕ_b0,q_b0)`: flux/charge, second-order ODE.
  - `SystemATS_ladder(ω_a,ω_b,E_J,E_Ca,E_Cb,a_0,b_0)`: complex a and b, first-order ODE. **This is the newer and preferred one.**
  - `SystemHO`.
  - `SystemMichiel(β,λ,x0,p0,ω_a)` and `SystemATS_Michiel`: the dimensionless model of a collaborator ("Michiel").
- **`config/drives.jl`**: the drives.
  - `DriveTwoModesBasics(ϵ_d,ω_d,buffer_phase,ϵ_0,ω_p,pump_phase)`. Buffer drive `ϵcos(ωt+φ)`. The pump enters as `sin(ϵ_0 cos(ω_p t+φ))`, the full coupling.
  - `DriveTwoModesJA`: the pump is the Jacobi–Anger first term, `2J1(ϵ)cos(...)`.
  - `DriveOneMode`.
- **`config/environments.jl`**: the environments.
  - `MeanField(κ_a,κ_b)`.
  - `MeanField_HO`.
  - `MeanField_Michiel(Q)`.
  - `OneChain(n, onsite, hopping, coupling)`: an explicit bath, a tridiagonal chain built from `baths.jl` spectral densities (Ohmic, exp-decay, Drude–Lorentz).
  - This file also defines `get_dt` (`1/(34√2 ω_a)` for mean field) and `get_init`.
- **`config/equations.jl`**: the EOMs, dispatched on the `SimulationParams` type.
  - `default_equation!` (full cosine).
  - `equation_1ODE!`, the first-order form for ChaosTools.
  - `equation_Taylor!` / `equation_1ODE_Taylor!`, a Taylor-truncated cosine using `utils/taylor_expansions.jl`.
  - `equation_O3!`.
  - `equation_stark!`.
  - `equation_parity_cos_sin!`, the parity-protected cat variant.
  - The Michiel equations.
  - Many notebooks instead define their own `test_equation!(du,u,p::SimulationParams{SystemATS_ladder_prepared,D,MeanField},t)` locally.
  - Known bug: `equation_1ODE_Taylor!` for `OneChain` uses an undefined `n`.
- **`config/solvers.jl`**: the solvers.
  - `default_solver` returns `(ϕ_a,ϕ_b,q_a,q_b,t)`, using VelocityVerlet.
  - For the ladder system it returns `(a,b,t)`, using Tsit5.
  - `emission(...)` returns the chain state.
- **`utils/`**: `FFT_perso(ϕ,t)`, `Shannon_entropy(F)`, `smoothstep`, `Taylor_sin/cos`.
  - `solve_resonance.py` (sympy) gives the Stark-shift-corrected (ω_d, ω_p) resonance. It is called through PyCall.
- **`visualization/`**: `plot_settings.jl` (Plots/GR defaults, `Perso_palette`, `Perso_colormap`), `plot_utils.jl` (`plot_mode`, `plot_perso`, `plot_gif`).
- Chaos diagnostics: `ChaosTools.lyapunov(CoupledODEs(...), t_max; d0=1e-7, d0_upper=1e-4, Ttr=15/κ_b)`.
  - The sign of the largest exponent gives the heatmaps.
  - Other diagnostics: Poincaré / stroboscopic sections at the memory period, FFT Shannon entropy, and IPR.
- The Python notebooks import `Plot_setting` from `/Users/tmalas/Desktop/Dev/Simulations_local/Code_python`, which is **outside this folder**.

## Python library (`pylib/catchaos/`, 2026-10-23)
Shared code for the Python notebooks, replacing the copies of the Hill code (`HB`/`HB2`, `orbit`, `monodromy`, `satellite_hb`, arclength, …) in each notebook. **Use it in new notebooks instead of copying cells.**
Not installed: `sys.path.insert(0, ".../classical_simulations/pylib")`. Notebooks run on Homebrew Python 3.11 (`/opt/homebrew/opt/python@3.11/bin/python3.11`, also used by `jupyter nbconvert`).
- `params.py`: `w_a, w_b, phi_a, phi_b, E_J, kappa_b` (code, π/10.4), `G0`, `g2_of`, `g2_J1_of`, `MHz`, `g2_max`, `s_of_MHz`.
- `hill/` (model-independent, any number of modes): `HarmonicBalance(N_H, M, C, K, period)` (exact one-period AFT; subclasses give `fnl`, `dfnl`, `b_ext`; `solve`, `solve_robust`, `newton`, `pad`); `orbit`, `state`, `strobe`, `odd_content`, `odd_mask`, `embed`, `fourier_project`; continuation `natural`, `arclength`, `continue_to_fold`, `branch_on_grid` (= `satellite_column`/`cat_column`), `pd_null_vector`; Floquet `monodromy(_matrix)`, `instability`, `unstable_set`, `transitions`, `hill_exponents`/`hill_multipliers` (genuine exponents, `select_genuine`), `liouville_residual`, `match_eigs`.
- `ats.py`: `HB(N_H, period=1, kfac, kappa, r)` with `set_s(s, r)`, `set_params(eps_p, eps_d)`, `set_kappa`; `HB2(N_H)` = `HB(N_H, period=2)`; `full_rhs(s, r, kfac)`, `amplitudes`, `state_of_amplitudes`, `rotating_frame`, `section`, `fourier_seed`, `mean_abs_aR`.
- `seeds.py`: `satellite_hb`, `seed_z`, `satellite(hb, s, r)`, `cat_seed(hb)`, `period2_seed(hb2)` (hard-coded J1 seeds continued to the sin g2).
- `branches.py`: continuation in p = g2/(100 MHz) at fixed r: `PS`, `g2_setter`, `extended_g2`, `arclength_g2`, `describe` (used by the torus_crisis period-2 notebooks).
- **Bit-identical** to the code the notebooks used to copy (`HB.set_s` keeps the notebooks' operation order; `test_bit_identical_to_notebooks`). Keep it so: `story_A_v1` caches its Julia jobs by an md5 hash of inputs computed with this code.
- Signature traps vs the old copies: `HB(N_H, period=1, kfac=…, r=…)` (2nd positional is `period`, pass `kfac=` by keyword); `full_rhs(s, r, kfac)`; `monodromy(hb, z)` takes the period from `hb`.
- **Notebooks on catchaos (refactored 2026-10-23, outputs re-checked identical):** `hill_orbits/` `Hill_method_cat_fold`, `Hill_method_satellite_fold`, `Hill_cat_seed_grid`, `Hill_satellite_seed_grid`, `floquet_rescaling_kappa_b`; `torus_crisis/` `satellite_start_bubble_edge`, `Hill_period2_branch_II`, `crisis_collision`, `story_A_v1`, `beyond_cusp_regimes`. Still with their own copies (older pinv-AFT generation): `Hill_method_corrected`, `Hill_method_corrected_cat_lobe_stability`, `Hill_method_doubling_corrected`, `floquet_zoom_period_doubling.py` (which executes cells of those notebooks).
- Git strips notebook outputs on commit (`.gitattributes`: `filter=strip-notebook-output`, nbstripout): `git show HEAD:<nb>` has no outputs; saved outputs exist only in the working copy.
- `paths.py` (`DATA` = `data/` root, use `DATA/"hill_orbits"`; `FIGS`, `ARCHIVE`, `ARTICLE_FIGS`, `dated(name)`, `latest(name, folder)`), `jsonio.py` (`save_json` with NaN → null, `load_json`, `nan2none`), `plotting.py` (`from catchaos.plotting import *` = `Plot_setting`).
- Tests: `python3.11 pylib/tests/run.py` (fast, against a frozen copy of the notebook code, `tests/legacy.py`) or `run.py all` (+ reproduces saved outputs: bubble edges at ε_d = 10 g2, branch II switching, satellite/cat seed grids, period-2 branch multipliers, Hill multipliers of `floquet_before_period_doubling`; ≈ 5 s). Run them after any change to the library.

## Folder map
Since 2026-10-23 the former `poster_chicago_2026/` (notebooks) and `analysis/output/chicago_poster_2026/` (data) are split by study.
The same study names are used in `studies/`, `data/` and `figures/`. A notebook of one study often reads the data of another (the maps read the Hill lines, the OTOC plots read the maps), always by the full path `data/<study>/<file>`.
In the notebooks, `OUT_DIR` (or `out_dir`, `OUT`, `output_dir`) is the `data/` root and every file name starts with its study, e.g. `OUT_DIR + "hill_orbits/satellite_seed_hill_22_10_2026.json"`; `FIG_DIR` is `figures/<study of the notebook>/`.
Write new outputs into the folder of the study that produces them. Superseded versions (an older date of the same file) go to `archive/data/` and `archive/figures/`.
- **`studies/`** (Apr–Oct 2026, article and Chicago 2026 poster):
  - `lyapunov_maps/`: `lyapunov_heatmap` (Julia, origin start; also the Poincaré sections → `figures/poster_chicago/poincare_sections/`), `lyapunov_heatmap_cat_start`, `lyapunov_heatmap_satellite_start`, `figure_lyapunov_heatmap` [Py], `lyapunov_convergence.jl` + `plot_lyapunov_convergence.py`.
  - `hill_orbits/` [Py]: `Hill_method_corrected` (satellite Floquet), `Hill_method_corrected_cat_lobe_stability`, `Hill_method_doubling_corrected` (period 2, Hill line, κ_b sweep), `Hill_method_cat_fold`, `Hill_method_satellite_fold` (+ beyond the cusp), `Hill_cat_seed_grid`, `Hill_satellite_seed_grid`, `floquet_rescaling_kappa_b`, `floquet_zoom_period_doubling.py`, `plot_floquet_multipliers`.
  - `limit_cycles/`: `limit_cycles`, `limit_cycles_period_doubled`, `phase_space_trajectories` (Fig. 5), `check_vital_ATS` + `plot_trajectories_animation` [Py] (poster mp4s), `meta_potential_map` (possible bug: `Z_b` computed with φ_a), `non_linearity_blow_up` (cos vs its quadratic expansion along chaotic trajectories).
  - `torus_crisis/`: `torus_breakdown`, `transient_chaos_shannon_entropy{,_not_transient}`, `beyond_cusp_regimes` + `beyond_cusp_dynamics.jl`, `satellite_start_bubble_edge` + `satellite_start_beyond_bubble.jl` + `bubble_crisis_lifetime.jl`, `Hill_period2_branch_II`, `crisis_collision{.ipynb,.jl}`, `story_A_v1` + `story_A_trajectories.jl` (figures in `figures/torus_crisis/story_A/`). The notebooks launch their `.jl` from `HERE` = this folder.
  - `otoc/`: `plot_OTOCS` [Py, J1, comparison cells], `plot_OTOCS_corrected_init_{cat,coherent}`, `classical_otoc_window.jl`, `otoc_chaotic_start_points.py`, `stochastic_twa_otoc.py`, `twa_otoc_heatmap.py`; `quantum/` = the former `Quantum_codes/` (dynamiqs OTOC/IPR/Wehrl notebooks, run elsewhere).
  - `poster/`: `cat-states-fock_population` [Py, dynamiqs].
- **`data/<study>/`**: outputs of the studies (same split). `data/variants/` holds the outputs of `analysis/chaos/{parity_protected_cat,multi-buffer_cat}` used in the article.
- **`archive/`**: `notebooks/` (the original J1 `Hill_method`, `Hill_method_doubling`), `data/` and `figures/` (superseded dated versions, e.g. the 07_04 maps, the 09/12/14_10 satellite-start maps, the 22_04 OTOC runs, the 08_10 Floquet sweeps).
- **`analysis/`**: older exploratory work (Nov 2025 – Jun 2026).
  - `chaos/`: Lyapunov heatmaps vs parameters (`Lyapunov_parameters/`: κ_b, ZPF, frequencies, sine-5 truncation) and Lyapunov over initial-state grids (`Lyapunov_phase_space/`).
    - Also Poincaré sections, FFT/entropy, and lobe/torus studies (`lobe_focus/`).
    - Variants: `chiral_cat/` (two ATS), `multi-buffer_cat/`, `parity_protected_cat/` (which has its own older `Hill_method.ipynb`), `Hamiltonian_with_stark_shift/`.
    - `period_doubling_bound_harmonic_balance.ipynb` [Py] is a 3-harmonic analytic period-doubling threshold.
    - `plot_entropy_IPR.ipynb` is a plotting hub for the quantum data.
  - `no_pump_frequency/`: the ω_p = 0 studies.
    - Main notebook: `Poincare_section.ipynb`.
    - `period_doubling*.ipynb` (κ_a sweeps, a logistic-map-like bifurcation diagram).
    - `find_period_doubling_dichotomy.ipynb`.
    - `reducing_hamiltonian.ipynb` and `minimized_Hamiltonian/`: the minimal set of monomials that keeps the chaos.
    - Stark-shift checks.
  - `checks/`: sanity checks (`SystemATS` vs `SystemATS_ladder`, `OneChain` emission).
  - `slowest_decay_rate/`: plots Floquet-Arnoldi (QuantumToolbox) bit-flip rates from `analysis/output/arnoldi_output/`.
  - `minimize_action/`: switching paths and least action.
    - `MPEP.ipynb` [Py]: Hill method plus shooting.
    - GMAM following Zakine (`from_Langevin/`).
    - Two-mode RWA Lindbladian → Hamiltonian paths (`from_Lindbladian_two_mode_RWA/`).
  - `adiabatic_elimination.ipynb` [Py], `resonance_frequencies_Leon.ipynb`.
  - `output/`: JSON/CSV data of the older work, with an `arnoldi_output/` subfolder (the `chicago_poster_2026/` subfolder moved to `data/`).
    - Many files (OTOC, Wehrl, bit-flip, T1 proxy, arnoldi) come from quantum simulations elsewhere in `Simulations_local`.
- **`Michiel_system/`**: comparison with Michiel's dimensionless one-mode or two-mode model (stability bounds, β sweeps).
- **`figures/`**: `<study>/` (working figures of each study, the former `figures/article/`), `poster_chicago/` (poster assets; `poincare_sections/`), and the older `chaos/`, `satellites/`, `switching_rate/`, `Two-photon/`. Article-ready copies stay in `article_chaos/figures_article/`.

### Conventions and traps
- Many older notebooks read or write `analysis/ouput/...`, which is a **typo**. The real folder is `analysis/output/`.
- Files are named with dates, `_DD_MM_YYYY` (e.g. `_07_04_2026`). The newest date is generally the authoritative one.
- Scratch notebooks: `test.ipynb` everywhere, `*_old`, `* copy`, `simu_Lucas/`, `noisy_dynamics/`, `config/Untitled-1.ipynb`.
- Older notebooks (`checks/check_vital_ATS`, `check_emission`, `parity_protected_cat/check_vital`, `studies/limit_cycles/check_vital_ATS`) include `simulator.jl` and the visualization files directly.
- `figures/poster_chicago/cat_*_pump_freq_*` and the OTOC data come from quantum simulations outside this folder.

## Hill method notebooks (`archive/notebooks/Hill_method*.ipynb`, `studies/hill_orbits/Hill_method*_corrected.ipynb`), in detail
Both notebooks use the same pipeline and are written in Python (numpy/scipy).

**Model.** The Hamiltonian is H = ω_a(x_a²+p_a²)/2 + ω_b(x_b²+p_b²)/2 + √2 ε_d cos(ω_d t) x_b − 2E_J sin(ε_p)[sin(√2φ_a x_a+√2φ_b x_b) − linear].

Rescaling x^λ = φ x and τ = ω_a t gives
`M Ẍ + C Ẋ + K X + f_nl(X) = f_ext`, with
`f_nl = −√2 [β_a', β_b']ᵀ (cos(√2(x_a+x_b)) − 1)`, `β' = 2E_J sin(ε_p) φ²/ω` and `ε_d' = ε_d φ_b/ω_b`.

**Ansatz.** The Fourier ansatz is `X = c0/√2 + Σ_k s_k sin(kτ) + c_k cos(kτ)`. The unknowns are `z = [c0, s1, c1, s2, c2, …]`, and each block is the pair (a, b), interleaved. So `z[::2]` is mode a and `z[1::2]` is mode b.

**Functions.** Both notebooks use the same functions, copy-pasted:
- `build_L(M,C,K,N_H,omega,nu)` is the linear harmonic-balance operator. Each harmonic k gets `K−(k ω/ν)²M` on the diagonal and `±(k ω/ν)C` off the diagonal.
- `build_b_ext(N_H, eps_d_p, nu)` puts the drive in the `c_ν` block of mode b.
- `build_Gamma(t,N_H,n,omega,nu)` is `kron(Phi, I_n)`, the time-major stacking.
  - The alternating frequency–time scheme uses `Gamma_inverse = pinv(Gamma)`.
  - It runs on `tau_j = linspace(0, 100π, 900·N_H)` (many periods, endpoint included).
- `f_nl_time`, `b_nl`, `dfnl_dX_blocks` (∂f_nl/∂X = 2 sin(y)·[[β_a,β_a],[β_b,β_b]]), `jacobian = L + Γ⁺ J Γ`.
  - `f_nl_time` and `dfnl_dX_blocks` read the **globals** `B`, `beta_a_p` and `beta_b_p`. Re-run the parameter cell, or set these in the sweep loop, before calling them.
- Solvers:
  - `powell_hybrid_solve` (scipy `root`, method `hybr`, with the analytic Jacobian) is the one in use.
  - `newton_solve` is the undamped fallback.
- `check_L_vs_Gamma` is a unit check of L against time-domain derivatives.
- `Floq_coef(z_sol, L, M, C, N_H, n)` is the Hill method for stability. It solves the quadratic eigenproblem `λ²Δ2 + λΔ1 + h_z = 0`, linearised to the generalized problem `eig(B_1, B_2)`.
  - The terms are `Δ1 = blockdiag(C) ± 2kM` (sin/cos coupling) and `Δ2 = blockdiag(M)`.
  - It keeps the `4n` eigenvalues with the smallest |Im λ| (the Floquet exponents in the first Brillouin zone).
  - It reads the **globals** `Gamma`, `Gamma_inverse` and `N`.
  - It returns `(eigvals, eigvecs)`.
- `match_eigs` does continuation sorting of eigenpairs along a sweep. It uses the Hungarian assignment (`linear_sum_assignment`) on eigenvalue distance plus eigenvector overlap.
- Floquet multipliers are `exp(λ T)` with `T = 2π`, the period of the harmonic-balance solution.
  - The plots use every other eigenvalue, `[:, 2*i]`, which assumes conjugate pairs are adjacent.

### `Hill_method.ipynb`: period-1 orbit and the period-doubling instability
- Time is τ = ω_a t and the drive sits at harmonic ν = 2 (`omega=2, nu=2`).
- Matrices: `M = diag(1, 1/4)`, `C = diag(0, κ_b'/2)`, `K = I`.
- Default parameters: ε_p = 0.58, ε_d = 8.12·g2, κ_b = 1/10.4, N_H = 4.
- The hard-coded `z_0_init` is a converged **"satellite"** orbit.
- Cell 10, `plot_phase_space`, plots the rotating-frame portraits a_R = a e^{iτ} and b_R = b e^{2iτ}. Its commented-out code dumps `satellites_{plus,buffer}_12_02_2026_Hill.json`.
- Cells 13–19 scan a grid of initial guesses (Re a, Im a) to find the distinct orbits. They classify them by the rounded first-harmonic coefficients of mode a.
- Cells 27–34 sweep ε_p from 0.4 to 0.65 (50 points), with ε_d = 10.12·g2 and N_H = 16, using continuation from the previous `z_sol`.
  - They track the Floquet exponents and multipliers. A multiplier crossing **−1** is the period-doubling bifurcation.
  - They save `archive/figures/floquet_multiplier_before_period_doubling.pdf`. The poster svg is commented out.

### `Hill_method_doubling.ipynb`: the period-doubled orbit
- Time is rescaled so the drive sits at harmonic ν = 4 (`omega=4, nu=4`). Matrices: `M = diag(1/4, 1/16)`.
- The memory oscillates at harmonic 2, so harmonics 1, 3, … of mode a carry the period doubling. Phase plots use `p_a = Σ (k/2)…`, `p_b = Σ (k/4)…`, a_R = a e^{2iτ} and b_R = b e^{4iτ}.
- Defaults: ε_p = 1.2, ε_d = 10.12·g2, κ_b = **π/10.4**, N_H = 8. The hard-coded `z_0_init` is a converged period-2 orbit.
- Cell 24 sweeps ε_p from 1.2 down to 0.58 (non-uniform grid), with N_H = 12 and continuation.
  - It tracks the Floquet multipliers, where a crossing **+1** marks the instability of the period-2 orbit.
  - It saves `figures/poster_chicago/floquet_multiplier_after_period_doubling.svg`.
- Cells 31–36 run a 2D sweep: ε_d/g2∈[5, 13.5] (40 points) × ε_p∈[0.6, 0.72] (40 points), with N_H = 15.
  - For each ratio they find the ε_p where some |multiplier| > 1.
  - The result is written to `archive/data/g2_list_07_04_2026_Hill_unstable.json`, which is the "Hill instability" line overlaid on the poster Lyapunov heatmap.

### Corrected versions (2026-10-08): `Hill_method_corrected.ipynb`, `Hill_method_doubling_corrected.ipynb`
**Prefer these over the originals.** Each one's first markdown cell lists its changes.
- The genuine 2n Floquet exponents are selected by eigenvector centring, keeping one member per family, then folded into |Im| ≤ ½. A Liouville check runs alongside.
- κ_b = π/10.4 in the Julia damping convention. Period-1: C = diag(0, κ_b′). Period-2: C = diag(0, κ_b′/2).
- The 2D sweep uses fixed storage indexing and proper continuation seeding.
- Outputs carry the suffix `_08_10_2026`. The new JSON `g2_list_08_10_2026_Hill_unstable.json` is ordered along the ascending ε_d/g2, so plot it **without** `[::-1]`.
- Results at ε_d = 10.12·g2:
  - The period-1 (satellite) orbit period-doubles at ε_p ≈ 0.627.
  - The period-2 orbit loses stability at ε_p ≈ 0.64 through a complex pair, i.e. a Neimark–Sacker bifurcation.

### g2 convention (2026-10-12)
The article notebooks now use g2 = E_J sin(ε_p) φ_a² φ_b, with ε_d = ratio·g2. At a given ratio the drive is therefore about 2× larger than with J1.
- **Re-run with this convention (outputs `_12_10_2026`):**
  - `Hill_method_corrected` → `floquet_before_period_doubling_12_10_2026.json` and `satellites_{plus,minus,buffer}_12_10_2026_Hill.json` (in `analysis/output/`).
  - `Hill_method_doubling_corrected` → `floquet_after_…`, `g2_list_12_10_2026_Hill_unstable.json`, `eps_d_ratio_list_12_10_2026_Hill_unstable.json` (the ratio grid, now 2–13.5 with 47 points) and the κ_b sweep figure.
  - `lyapunov_heatmap{,_cat_start,_satellite_start}` (grid ε_p 0.2:0.01:1.1 × ratio 2:0.1:13.5).
  - `limit_cycles` (κ_b switched to π/10.4, `angle_shift = 0`), `limit_cycles_period_doubled` (ε_p = 0.588), `transient_chaos_shannon_entropy` (ratio 5.2 = the former physical point).
  - The plotting notebooks `plot_floquet_multipliers`, `figure_lyapunov_heatmap` (writes `figures/lyapunov_maps/lyapunov_heatmap_origin_start_12_10_2026.pdf`) and `plot_OTOCS_corrected_init_{cat,coherent}`.
- **Hard-coded Hill seeds** were converged with J1. Each sweep first continues them in ε_d (`g2_J1_of` → `g2_of`).
- **Quantum OTOC/IPR data** come from cluster runs with ε_d = ratio·J1 g2. The OTOC plots only relabel them (`relabelled_mesh`: ε_d/g2 = ratio·J1/sin ≈ ratio/2, so |α|² ≈ 0.5–2.4). They should be re-run with the sin definition.
- **Still on J1, not part of the article:** `check_vital_ATS`, `meta_potential_map`, `non_linearity_blow_up`, the original `Hill_method`/`Hill_method_doubling`, `plot_OTOCS`, `analysis/…`. In `lyapunov_heatmap_satellite_start`, the check cells after the JSON export were not re-run, and their `_12_10_2026` outputs are still on J1.
- `figure_lyapunov_heatmap` also re-plots the cat-start map in the origin-map style, y-axis cropped to the data: `figures/lyapunov_maps/heatmap_lyapunov_cat_start_restyled_12_10_2026.pdf`. The origin map carries the markers a, b, c = (60, 90, 105 MHz, ε_d/2g2 = 3).
- **Lean notebooks written 2026-10-12 for the article** (sin g2, κ_b = π/10.4; no saved outputs, they were run as scripts with `julia --project=Code_Julia`):
  - `studies/limit_cycles/phase_space_trajectories.ipynb`: Fig. 5a–c, trajectories at points a, b, c (flux form + VelocityVerlet, like the maps). Saved as PNG (the vector PDFs were 6–8 MB each), as is the parity-protected panel.
  - `analysis/chaos/multi-buffer_cat/lyapunov_multi_buffer.ipynb`: replaces the Lyapunov cells of `test_multi_buffer` (which was on J1, κ_b = 1/10.4, and passed an 8-dim state to all three models). Each of the N buffers is driven at ε_d/N (ε_d = 10.12 g2_eff), so the RWA cat size ε_d/2g2 does not depend on N. With ε_d on every buffer the cat was N times larger; the threshold was then 114 / 122 MHz (bi / tri) and is now ≈ 120 MHz for both, against 75 MHz for one buffer.
  - `analysis/chaos/parity_protected_cat/phase_space_parity_protected.ipynb` (g2/2π = 100 MHz, ratio 10.12) and `Hill_method_corrected.ipynb` (satellite Floquet sweep; seed from time integration). The old parity `Hill_method` had the wrong sign on the memory force and no √2 on both forces, and computed ε_d from the previous step's J1 g2.
  - `limit_cycles` has an extra cell (after the analytic-HB cell) for Fig. 3: same figure without the green/pink analytic orbits → `classical_limit_cycles_HB_12_10_2026.pdf`.
- **`equation_parity_cos_sin!` (ladder form) fixed 2026-10-12**: `coupling_a` lacked the `/sqrt(2)` (memory force √2 too large). Checked against the flux form.
- **Plot units in `limit_cycles` and `limit_cycles_period_doubled` fixed 2026-10-12:** cell 3 plotted `a_R = √2·a·e^{iωt}·i` (built `x_lam = √2 x` then divided by √2 once), and the Hill/HB overlays used the same √2. Now `a_R = i a e^{iωt}` (divide by 2), Hill orbits divided by φ (not φ/√2), `orbit_a/orbit_b` without √2, axis limits scaled by 1/√2. Cats at |a| ≈ 2.0 (= √(ε_d/2g2) at ratio 8.12), satellites at |a| ≈ 4.2 (≈ 18 photons, not 36). Hill-notebook orbits: a = (x^λ + i p^λ)/(√2 φ).
- Article-ready copies of the new figures go to `article_chaos/figures_article/`.
- **`studies/torus_crisis/torus_breakdown.ipynb` (2026-10-19)**: answers report_v2 open question 4. It continues the satellite in g2 with steps down to 0.2 kHz (`equation_1ODE!` + Vern9, tolerance 1e-10).
  - At ratio 10.12 the sequence is: PD at 75.25 MHz → NS at 76.70 MHz (torus T²; λ1 → 0 as 1/T) → torus doubling at ≈ 77.0 MHz → abrupt jump at g_c = 77.249 MHz to a large chaotic attractor.
  - At g_c the torus is still smooth and transversally stable, and no phase locking is seen. Below g_c the chaos is a chaotic saddle (transients of 10³–10⁵ T_a). This is not a gradual torus breakdown.
  - The window is 0.55–0.7 MHz wide at ratios 6, 8, 10.12 and 13, between two grid points of the maps.
  - Outputs: `torus_breakdown_19_10_2026.json` and `figures/torus_crisis/torus_*_19_10_2026.*`.
  - It runs headless with `jupyter nbconvert --execute` (kernel julia-1.11, 4 threads, ≈ 5 min). Plot cells use `display("image/png", p)`, because IJulia otherwise stores an SVG of every point (≈ 200 MB).
- New results at ε_d = 10.12 g2: period doubling of the satellite at ε_p ≈ 0.5795 (g2/2π ≈ 75.3 MHz), Neimark–Sacker at g2/2π ≈ 76.7 MHz.
- **κ_b dependence of the threshold (2026-10-19, cells 43–52 of `Hill_method_doubling_corrected`)**: g2* grows as κ_b², not linearly. The cause is a Hamiltonian–Hopf (Krein) collision at κ_b → 0, unfolded by the loss into a Neimark–Sacker crossing at arg μ ≈ ±0.2π. The satellite existence line 2κ′_b|y| = ε′_d is not the threshold. Summary: `article_chaos/kappa_b_threshold_slope.md`.
- **`studies/hill_orbits/floquet_rescaling_kappa_b.ipynb` (2026-10-20, report_v2 point 10)**: self-contained (HB2 class, one-period AFT, κ_b in the GKSL convention, code κ = κ_b/2). It computes Λ = ln|μ_NS|/T for 10 values of κ_b/2π (48–384 MHz) and the collapse Λ/κ_b vs a(g2 − g2*(0+))/κ_b².
  - g2,0 = 76.302 MHz (lossless collision), g2*(0+) = 76.147 MHz (dissipation-induced offset), a = 0.210 ns⁻²/MHz.
  - Two-mode model Λ = −κ_b/4 + Re√D, with D computed directly from the exponents, and a check of truncation orders.
  - It runs headless with `jupyter nbconvert --execute` in ≈ 6 min. This supersedes the "σ_H ≈ κ_b/2" reading of `kappa_b_threshold_slope.md`.
- **`studies/hill_orbits/Hill_method_satellite_fold.ipynb`, section "Beyond the cusp at large g2" (2026-10-22)**: period-1 branch at ε_d = 10.12 g2 vs κ_b (multiples of κ_b⁽⁰⁾ = π/10.4 code).
  - The satellite period-doubling bubble (75.5–103.5 MHz at κ_b⁽⁰⁾) closes at κ_b ≈ 5.36 κ_b⁽⁰⁾ (g2 ≈ 99 MHz), before the cusp (9.7). Beyond the cusp: no PD, no period-2 orbit, stable up to g2_max = 137.4 MHz.
  - Formal continuation to sin ε_p > 1 (= larger E_J φ_a²φ_b): first instability is always a supercritical Neimark–Sacker of the period-1 orbit, arg μ ≈ 0.61–0.69π, at 142 MHz (κ_b⁽⁰⁾) … 168.4 MHz (11 κ_b⁽⁰⁾) … 209 MHz (20 κ_b⁽⁰⁾).
  - At 11 κ_b⁽⁰⁾, 130 MHz, other stable period-1 orbits coexist (a large pair ⟨|a_R|⟩ ≈ 5.8, a symmetric orbit around the origin); the origin and cat start end on the branch.
  - Outputs `beyond_cusp_bifurcations_22_10_2026.{json,pdf}`. The whole notebook runs headless in ≈ 3 min.
- **Satellite-start map seeded with the exact Hill satellite (2026-10-22, report_v2 "TO DO for v3")**:
  - `studies/hill_orbits/Hill_satellite_seed_grid.ipynb` [Py, ≈ 5 min headless]: for each ε_d/g2 of the map, pseudo-arclength from ε_p = 1.1 down to the satellite fold (steps past the fold are rejected, so it never continues onto the saddle). Stores the τ = 0 state [x_a, x_b, dx_a/dτ, dx_b/dτ] on the map grid (`null` below the fold), all Floquet multipliers and the fold per ratio → `satellite_seed_hill_22_10_2026.json` (`state[j_ratio][i_eps_p]`). Fold identical to `satellite_fold_line_20_10_2026.json`. Figure `figures/hill_orbits/satellite_seed_hill_grid_22_10_2026.pdf`.
  - `lyapunov_heatmap_satellite_start.ipynb` now reads it (`hill_satellite_amplitudes(j, i)`: a = (x_a + i v_a)/(√2φ_a), b = (x_b + i v_b/2)/(√2φ_b)); dashed fold line on the maps. Map cells re-run (≈ 70 min): outputs `*_satellite_start_22_10_2026.json`, `figures/lyapunov_maps/heatmap_lyapunov_satellite_start_22_10_2026.pdf`. The check cells (after the export) still use the analytic seed and were not re-run.
  - Result: below the Hill line the end state is the satellite on 99.8 % of the points (was 60 %, 40 % cat); the satellite region starts at the fold (within one grid step). The onset of chaos is unchanged (within one grid step of the old map, on the NS line; at ε_d/g2 = 2 now 91.7 MHz instead of 96.7). Above ≈ 100 MHz, 1630 points went from chaotic to regular: the start is now exactly on the restabilised period-1 satellite (> 103.7 MHz, basin |δa| ≈ 0.01–0.03), so that regular region is not robust.
  - Same for the cat (2026-10-22): `studies/hill_orbits/Hill_cat_seed_grid.ipynb` [Py, ≈ 4 min] continues the cat by arclength from ε_p = 0.2 up to its fold → `cat_seed_hill_22_10_2026.json` (`state_plus` = cat_sign 1, `state_minus` = orbit at τ = π; `null` above the fold; fold identical to `cat_fold_line_19_10_2026.json`; no fold inside the map for ε_d/g2 ≤ 6.4). `lyapunov_heatmap_cat_start.ipynb` starts on the exact cat below the fold and from a = i|α|, b = 0 above it; cat fold line on the plot; new onset-vs-fold cell. Outputs `*_cat_start_22_10_2026.json` (+ `lyapunov_values_…`, `on_cat_start_…`), `figures/lyapunov_maps/heatmap_lyapunov_cat_start_22_10_2026.pdf`; whole notebook headless ≈ 12 min.
    - Result: 0 chaotic points out of 8509 started on the cat (the old a = i|α| map had 604 there); the onset is at the first grid point above the fold for 64 of the 71 ratios with a fold in the map, 2–5 points above for 7 ratios (6.6–8.0: weakly negative λ, |λ| ≲ 0.13 κ_b, just above the fold; not classified). No chaos at all for ε_d/g2 ≤ 6.4. So the chaos onset from the cat is the cat fold; the old 6–9 % gap was the a = i|α| start.

- **`studies/torus_crisis/Hill_period2_branch_II.ipynb` (2026-10-23)** [Py, ≈ 1.5 min headless, no Lyapunov]: period-2 orbits at ε_d = 10 g2 (row of `bubble_crisis_lifetime.jl`), HB over 2T_a with N_H = 32. Branch switching from the odd null vector of the period-2 Jacobian at each period doubling, then pseudo-arclength in p = g2/100 MHz.
  - Bubble 75.340–103.883 MHz. **Branch II is subcritical**: it exists only for g2 > 103.88 (where the satellite is stable), a saddle from birth, odd ≈ 0.0075√(g2 − g_PD) (≤ 0.026), second multiplier through −1 at 114.0 MHz, no fold up to 140 MHz. Branch I is supercritical (75.34, NS at 76.80, then unstable up to 140).
  - Branch II's distance to the satellite on the section (0.001 → 0.023 over 104–120 MHz) is the thin basin of the satellite beyond the bubble (kicks of `satellite_start_bubble_edge` section 3).
  - The ω_a/2 orbit (time integration from the satellite at 103.5 MHz, then HB): saddle-node at 99.970 MHz; node stable 99.97–100.24 and 101.20–106.71, NS in between and above, PD from 111.5, folds 119.39/118.05. It reproduces every period-2 escape destination of the crisis runs (101.5–106 MHz, to 0.01; 3.0/3.9 = orbit and partner a → −a).
  - Outputs `period2_branches_eps_d10_23_10_2026.json` (multipliers, section points, HB coefficients every 10 points), `figures/torus_crisis/period2_branch_II_eps_d10_23_10_2026.pdf`. The crisis is resolved in `crisis_collision.ipynb` (next entry).
- **`studies/torus_crisis/crisis_collision.ipynb` + `crisis_collision.jl` (2026-10-23)** [Py + Julia; Julia ≈ 8.5 min on 8 threads, skipped if the output exists (`RUN_JULIA = False`); the rest ≈ 1 min]: what ends the bubble's chaotic attractor at ε_d = 10 g2.
  - **It is not a boundary crisis with a saddle.** The ω_a/2 saddle is 0.08–0.10 from the node. Above g_NS both branches of its unstable manifold go into the chaotic set, and escaping trajectories do not pass near it.
  - **It is a hole opening:** the ω_a/2 node, embedded in the attractor while Neimark–Sacker unstable, restabilises at g_NS = 101.2014 MHz. Escapes start right above it (τ = 1.1e5/κ_b at 101.22 MHz). **The g_c ≈ 101.4 of `satellite_start_bubble_edge` (section 4) is superseded**: it was an artefact of the 1490/κ_b cutoff.
  - The node's basin radius ρ50 ∝ (g2 − g_NS)^0.60 (0.007 → 0.1 over +0.02 → +1.8 MHz). The attractor measure near the node ∝ r^2.75. So γ_pred = 1.64; fits give γ = 1.60 with g* = g_NS, and g* = 101.15, γ = 1.85 with g* free. τ·μ_att(<ρ50)/T_a = 3–9 while τ varies ×530.
  - The Neimark–Sacker is supercritical: a small torus (size 0.024) at 101.18, gone at 101.15. The threshold is therefore in [101.15, 101.20] MHz.
  - Same leak in the node's first stability window (99.97–100.24 MHz: runs land on the node). The chaotic set is an attractor (on 1e5 T_a) only for g2 < 99.97 and on about 100.4–101.15 MHz.
  - Outputs `crisis_collision_{input_,,summary_}23_10_2026.json` (raw output 31 MB), `figures/torus_crisis/crisis_collision_eps_d10_23_10_2026.pdf`.
- **`studies/torus_crisis/story_A_v1.ipynb` + `story_A_trajectories.jl` (2026-10-23)** [Py + Julia]: phase-space pictures of the story of case A (κ_b⁽⁰⁾, ε_d = 10 g2), one figure per range of g2 (13 steps, 30 → 117 MHz) in `figures/torus_crisis/story_A/`.
  - Each row has four panels: HB limit cycles in the rotating frame, the Poincaré section with trajectories coloured by κ_b t and the border of the chaotic set, |a_R(2πk)| vs time with each trajectory's fate, and a basin map.
  - Basin maps: slice b(0) = 0, 32×32 grid of a_R(0) ∈ [−6.5, 6.5]², N = 3000 or 8000 T_a. Local basin radius ρ50 from random kicks.
  - Julia results are cached per job (md5 hash of the job) in `story_A_23_10_2026.json` (80 MB); only new or changed jobs run. A full run is ≈ 14 min on 8 threads; with the cache, ≈ 2 min.
  - New facts at ε_d = 10 g2:
    - End of the torus of branch I between 77.30 and 77.35 MHz (77.25 at 10.12).
    - At 76.0 MHz, 24 % of the slice is still chaotic after 224/κ_b, so the chaotic saddle already exists there.
    - The cat's basin is small on the slice (6 % at 76–77, 2 % at 85, 0 at 95 MHz). Cat ρ50 = 0.15 / 0.14 / 0.11 / 0.04 at 60 / 85 / 95 / 97.5 MHz.
    - A period-3 attractor at 99–100.2 MHz (1–2 % of the slice).
    - At 108 MHz, 96 % of the slice ends on a **large period-1 orbit winding once around the origin** (|a_R| ≈ 4.6), not computed by HB yet.
    - At 117 MHz, 16 % ends on a period-5 orbit and the rest is still chaotic after 600/κ_b.
    - Satellite at 105 MHz: ρ50 = 0.0056. Peripheric node: ρ50 = 0.037 / 0.083 / 0.087 at 100.2 / 102.4 / 105 MHz.
  - Names used: "peripheric node / saddle" = the ω_a/2 period-2 pair (Théo's naming, 23/10).
- **Classical OTOC (2026-10-21/22)**: `studies/otoc/classical_otoc_window.jl` computes C_cl = (∂x_a^rot(t)/∂x_a(0))² (tangent dynamics of `equation_1ODE!`), fitted exactly like the quantum OTOC (slope of log C vs κ_b t / 2, κ_b = 2π/10.4). Point start a = i|α| and TWA (Wigner of |iα⟩|0⟩). Output `classical_otoc_window_21_10_2026.json`; `include`-able (main only runs as a script).
  - On the quantum 3-period window, TWA reproduces the quantum OTOC rates (regular region to ~2%); the long-time slope of log C/2 is the map's λ. The 3-period window only sees the transient from |iα⟩|0⟩.
  - `otoc_chaotic_start_points.py` (Python port, scipy DOP853, same EOM and tangent dynamics; agrees with the Julia code to 1e-8 from a given state; ≈ 20 s) → `otoc_chaotic_start_points_22_10_2026.json`: states (a0, b0) on the attractor after Ttr, at the first drive period with |a|² ≤ 15 and |b|² ≤ 3 (ratio 10, ε_p 0.6–0.93), with classical references (λ over 30/κ and over the first 10 ns — above 100 MHz the chaos ends after 15–40 ns — and point and TWA log C over 24 periods). They seed the section "From the classical attractor" of `studies/otoc/quantum/OTOCS_poster_init_coherent_tr.ipynb` (whose `otoc_point` now takes `a0`, `b0` and also returns `otoc_cat`, the even cat of ±a0). Comparison cell in `plot_OTOCS.ipynb`.
  - `stochastic_twa_otoc.py` (2026-10-22, ≈ 75 s) → `otoc_stochastic_twa_22_10_2026.json`: classical counterpart of the dissipative OTOC, ⟨(∂ E_noise[x_a(t)]/∂x_a(0))²⟩ with the buffer's vacuum noise (force √κ_b ω_b dW on φ̈_b, checked to relax to ⟨φ_b²⟩ = ½), central difference h ∈ {0.05, 0.15, 0.4} on common noise paths (the pathwise tangent average does not converge), 32 Wigner × 512 noise paths, RK4 + additive noise. It reproduces the rise, peak and decay of the quantum OTOC from the attractor states (`otoc_attractor_start_na65_nb33_22_10_2026.json`); without noise there is no decay.
  - `twa_otoc_heatmap.py` (2026-10-22, ≈ 100 s) → `twa_otoc_heatmap_22_10_2026.json`: TWA OTOC rate ⟨(∂x_a^rot(t)/∂x_a(0))²⟩ (mean of squares over 16384 Wigner samples of |iα⟩|0⟩ + loss noise) on the quantum grid and 3-period window. Same sign as the quantum OTOC on 288/299 points, zero lines within one grid step. On this window, averaging the derivative over the noise before squaring (the late-time estimator of `stochastic_twa_otoc.py`) underestimates the quantum rates strongly. Article figure `figures_article/otoc_vs_twa_heatmap_22_10_2026.pdf` (cell baed0266 of `plot_OTOCS.ipynb`); section `article_chaos/twa_otoc_section.tex` + `twa_otoc_refs.bib`.
### Known issues in the ORIGINAL Hill notebooks (found 2026-10-08)
1. **The κ_b convention is inconsistent.**
   - In `Hill_method.ipynb`, the parameter cell uses `kappa_b=1/10.4` with `kappa_b_p = kappa_b/w_b*np.pi`, but the ε_p sweep (cell 27) uses `kappa_b/w_b`, which is a factor π smaller.
   - `Hill_method_doubling.ipynb` uses `kappa_b = π/10.4` with `kappa_b_p = kappa_b/w_b`.
   - So the "before" and "after" period-doubling figures were computed with damping that differs by π.
   - The Julia notebooks also mix 1/10.4 and π/10.4. The Julia EOM uses `−2κ du` while the Hill markdown uses `−κ_b p_b`, so the conventions should be reconciled before any quantitative comparison.
2. **The doubling notebook does not run top to bottom as saved.**
   - In cell 32 (the 2D sweep), `eigvals_array[i,j,:] = Floq_coef(...)` assigns a `(eigvals, eigvecs)` tuple of 8 eigenvalues into a slot of length `2n = 4`. With the current `Floq_coef`, that raises an error that is caught, so `z_guess` is never updated and the whole array stays NaN.
   - The saved outputs come from an earlier `Floq_coef` version: cell 32 ran at execution count 7, before the current definition at count 11.
   - In both notebooks, the failure fallback `np.full(4, nan)` has a different length from the 8 eigenvalues returned by `Floq_coef`.
3. **Plot labels are wrong.**
   - Doubling cell 33 plots against `eps_p_list` while the x-axis is labelled g2, and it ignores the loop variable `i`.
   - The y-label says "Real part" while the plot shows `abs`.
4. **The `Floq_coef` comment mislabels the indices.** It says "i = 2k−1 cos index, j = 2k sin index", but in the `[c0,s1,c1,…]` ordering, 2k−1 is s_k and 2k is c_k. The signs in the code are consistent with the correct labelling, so only the comment is wrong.
5. **g2 uses J1(ε_p)** (the formula for an oscillating pump, from Leon's paper), while the EOM is in the ω_p = 0 regime with a static factor sin(ε_p). Fixed in the corrected notebooks on 2026-10-12 (see "g2 convention").
6. **The alternating frequency–time step is costly.** `pinv` of a (2·900·N_H) × (2(2N_H+1)) matrix is computed over 50 periods. One period sampled uniformly with `endpoint=False` gives an exact, cheap transform (Γ⁺ is analytic).
7. **The two notebooks duplicate about 300 lines of identical functions**, plus a third copy in `analysis/chaos/parity_protected_cat/Hill_method.ipynb`. A shared `.py` module would prevent drift.
