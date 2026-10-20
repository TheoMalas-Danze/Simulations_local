# Husimi Harmonic Balance (HHB)

Periodic non-equilibrium steady states (NESS) of driven-dissipative bosonic
systems, without Fock-space truncation.

> **Scope of this file.** It describes the project *as it stands* and proposes a
> target layout with a migration path. The physics — equations, conventions,
> reference parameters, and the list of bugs not to reintroduce — lives in
> [`CLAUDE.md`](CLAUDE.md) and is not duplicated here.

---

## 1. The method

Three ingredients, applied in order:

1. **Lindblad → Husimi Q.** Map the GKSL equation to a PDE for
   `Q(α,α*,t) = ⟨α|ρ|α⟩/π` via the standard correspondence rules.
2. **Moment closure.** Assume a form for `Q` and track only its low moments,
   giving a closed set of ODEs with periodic coefficients. The closure is the
   scientifically delicate step — see §5.
3. **Harmonic balance.** Expand each moment in a truncated Fourier series and
   solve the resulting algebraic system by Newton/Powell-hybrid, reusing the
   classical structural-dynamics HB machinery of Detroux et al. (2015):
   a constant-coefficient linear part handled exactly, plus a nonlinear
   remainder evaluated pointwise in time by alternating frequency/time (AFT).

**Selling point.** Cost is independent of photon number and of the attractor's
distance from the phase-space origin: `O(N)` means plus `O(N²)` covariances,
rather than exponential in the Fock cutoff.

**Weakness.** The closure is uncontrolled exactly where the physics is most
interesting. Quantifying that is what most of §5 is about.

## 2. The two systems

| | **Kerr oscillator** | **Two-mode SNAIL cat** |
|---|---|---|
| Role | numerical testbed | the physics target |
| Modes | one | memory `a` (undamped) + lossy buffer `b` |
| Reference | `dynamiqs` (exact truncated-Fock Lindblad) | none tractable |
| Code | `Kerr_oscillator/` | `Cat/` |

The Kerr oscillator exists so that every claim about the closure can be checked
against an exact answer. The cat is the goal: `ω_b = 2ω_a = ω_d`, two-photon
dissipation stabilising a cat state in `a`.

## 3. Current layout

```
Hill_method/
├── README.md                     this file
├── CLAUDE.md                     equations, conventions, bug registry
├── .claude/settings.json         permission allowlist (shared, tracked)
├── .gitignore
│
├── hill_method.py                ORIGINAL 2nd-order "mechanical" Hill/HB solver
│                                 (∇²⊗M + ∇⊗C + I⊗K). Floquet stability via
│                                 floquet_coefficients(). Used by the notebook
│                                 below; the ancestor of everything else here.
├── Instability_threshold_kappa_b_dependence.ipynb
│                                 κ_b-dependence of the instability threshold
│
├── Cat/                          two-mode SNAIL cat qubit
│   ├── hill_method_HHB_cat.py    14 real DOF, exact to all orders in φ_a, φ_b
│   ├── husimi_hill_method_test_cat.ipynb
│   ├── husimi_hill_method_test_cat_simplified.ipynb
│   ├── Figures/                  paper figures  (NOTE: gitignored — see §6)
│   └── sol_cat_satellite_json/   saved solution branches
│
├── Kerr_oscillator/              single-mode testbed
│   ├── Kerr_3rd_order_sympy.ipynb        symbolic derivation of the 3rd-order
│   │                                     cumulant equations, with checks
│   ├── hb_moments_kerr.py                n=2  Gaussian closure       (5 DOF)
│   ├── hb_moments_kerr_3.py              n=3  cumulant closure       (9 DOF)
│   ├── hb_moments_kerr_n.py              any n, cumulant closure     ((n+1)(n+2)/2−1 DOF)
│   ├── hb_moments_kerr_max_entr_n.py     max-ent closure — NOT APPLICABLE (§5)
│   ├── sweep_kerr_hellinger_n.py         the sweep driver (CLI, all orders)
│   ├── husimi_hill_method_test_kerr.ipynb
│   ├── husimi_hill_method_test_kerr_3.ipynb   (duplicate — see §6)
│   ├── sweep_kerr_hellinger.ipynb             n=2 sweep
│   ├── sweep_kerr_hellinger_3.ipynb           n=3 sweep
│   └── results/
│       ├── data/                 *.npz benchmark records (TRACKED: regression baseline)
│       └── figures/              generated plots (gitignored)
│
└── wigner_gif/                   84 MB of animations (see §6)
```

### Module lineage

`hb_moments_kerr.py` → `_3.py` → `_n.py` is a progression, not a set of
alternatives. Each supersedes the previous; `_n.py` at `order=2` reproduces
`hb_moments_kerr.py` exactly, and at `order=3` reproduces `_3.py` exactly (both
verified). The older two are kept because the `sweep_kerr_hellinger*.ipynb`
notebooks import them directly.

From `_n.py` onward the ODEs are **re-derived symbolically at import**, never
hand-transcribed — the FPE moment recursion plus the cumulant closure, with the
triangular structure of the moment→cumulant map asserted rather than assumed.
Hand transcription is where every historical bug in this project came from.

## 4. Running things

```bash
# environment (note: this venv is currently missing sympy — see §6)
PY=~/venvs/dynamiqs-env/bin/python

# module self-tests (symbolic checks + HB vs. direct time integration)
$PY Kerr_oscillator/hb_moments_kerr_n.py
$PY Kerr_oscillator/hb_moments_kerr_3.py

# Hellinger sweep against dynamiqs, any cumulant order
cd Kerr_oscillator
$PY sweep_kerr_hellinger_n.py --order 3                 # full χ sweep
$PY sweep_kerr_hellinger_n.py --order 4 --chi-max 1.2   # order 4 stops converging past ~1.32
$PY sweep_kerr_hellinger_n.py --order 4 --chi 1.0       # single point
```

Results go to `results/data/*.npz` and `results/figures/*.png`, resolved
relative to the script, so the working directory doesn't matter. The plot
overlays every order whose `.npz` it finds.

**Derivation cost grows steeply with order**: 0.2 s (n=2), 0.8 s (n=3), 31 s
(n=4), minutes at n=6. It is cached per-process, so one sweep pays it once, but
every fresh kernel pays it again. A disk cache (`sp.srepr` + re-lambdify) is the
obvious fix if order ≥ 4 becomes routine.

## 5. Where the science stands

### Validated

HB matches direct time integration of the *same* moment ODEs to 1e-12 (χ=0) and
~1e-6–1e-7 at finite χ (the `N_H=8` Fourier truncation floor, dropping to 1e-10
at `N_H=14`). That is an internal consistency check: it validates the HB/AFT
machinery, not the closure.

### Accuracy against the exact Lindblad answer

Period-averaged Hellinger distance between the reconstructed `Q_HB` and the
`dynamiqs` steady state (Δ=1, κ=1, ε=3, ω_d=1.3, 20 Fock states):

| χ | 0.17 | 0.33 | 0.50 | 0.67 | 0.83 | 1.0 | 1.17 | 1.2 |
|---|---|---|---|---|---|---|---|---|
| order 2 (Gaussian)  | 0.0052 | 0.0073 | 0.0106 | 0.0150 | 0.0199 | 0.0253 | 0.0319 | — |
| order 3             | 0.0041 | 0.0035 | 0.0047 | 0.0075 | 0.0121 | 0.0185 | 0.0271 | — |
| order 4             | 0.0041 | **0.0030** | **0.0032** | **0.0052** | **0.0100** | 0.0197 | 0.0524 | 0.0470 |

The χ=0 value (0.0076, all orders) is the grid/tolerance floor, not a physics
result. Order 4 wins up to χ≈0.85, crosses order 3 near χ≈1.0, then degrades
sharply.

### Closure error vs. reconstruction error

These are separable, and separating them is the main diagnostic result. Feeding
the order-n Gram–Charlier reconstruction the **exact** cumulants (taken from the
`dynamiqs` density matrix) isolates the reconstruction:

| χ | n=2 | n=3 | n=4 |
|---|---|---|---|
| 0.33 | 0.0071 | 0.0036 | 0.0032 |
| 1.0  | 0.0168 | 0.0089 | 0.0053 |
| 1.2  | 0.0197 | 0.0110 | **0.0065** |

The reconstruction keeps improving monotonically with order everywhere tested —
even at χ=1.2, where the sweep reports 0.048. **So the bottleneck past χ≈1 is
the closure producing wrong cumulants, not the truncated representation of Q.**
At χ=1.0 the HB cumulants already deviate from truth by amounts comparable to
the cumulants themselves (|Δκ₃₀| = 0.055, |Δκ₄₀| = 0.063), which is why the
orders stop separating there.

### The order-4 pathology

Three distinct symptoms, all artefacts of truncating the hierarchy in the
*equations of motion*:

1. **Cumulant drift** — from χ≈1.0, as above.
2. **A spurious fold at χ ≈ 1.078.** With fine continuation (Δχ=0.05,
   substepping to 0.003) the order-4 periodic solution *ceases to exist*. The
   true Lindblad NESS is smooth and unimodal there. The sweep's Δχ≈0.167 step
   jumps the fold and Newton lands on a **disconnected spurious branch** whose
   cumulants are ~3× further from truth — that branch is the entire χ=1.1667
   spike, not a physics result. The fold is detectable without knowing the
   answer: σ_min of the HB Jacobian decays 0.58 → 0.23 → 0.114 → 0.073 as
   χ goes 0 → 0.85 → 1.0 → 1.05.
3. **Finite-time runaway.** Time-integrating the order-4 ODEs from vacuum blows
   up in under one period for χ ≳ 0.83; order 3 stays periodic throughout.
   Started *on* the HB orbit, order 4 stays periodic for 200+ periods, so the
   orbit is stable but its basin has collapsed — the classic quartic
   moment-closure instability.

**Practical consequence:** never initialise the order-4 solve from vacuum;
always continue from a converged neighbour.

### Closure attempts (the record)

| Approach | Positive? | Enough DOF? | Verdict |
|---|---|---|---|
| Cumulant, order 3 | tails can go negative (mass ~1e-4) | yes | **dynamically robust across all χ tested** |
| Cumulant, order 4 | same | yes | best reconstruction, pathological dynamics |
| Maximum entropy | yes, by construction | — | **impossible here** |
| Gaussian scale mixture (NVMM) | yes, by construction | **no** | **too rigid** |

- **Max-ent** (`hb_moments_kerr_max_entr_n.py`): a max-ent density `exp(-P)`,
  `deg P = n`, needs its top-degree form positive, so its tails are *lighter*
  than Gaussian and it is necessarily platykurtic. The Kerr Q is **leptokurtic
  at every χ>0** (max excess kurtosis +0.007 / +0.027 / +0.145 at χ = 1/6, 1/3,
  1). The moments lie outside the set where max-ent has a solution at all.
  It fails *silently* — the inner Newton "converges" to 1e-14 while the
  quadrature integrates a divergent integral; the tell is 51% grid dependence.
  Order 6 does not rescue it. Module retained, with a header, as a record.
- **Gaussian scale mixture** with drift (`ζ = β(s−1) + √s·L·u`, `s~Gamma`):
  positive, unimodal, leptokurtic, analytic moments — but only 6 non-mean
  parameters against 12 moments through degree 4, and its skewness is confined
  to a 2-parameter pattern inside a 4-dimensional space. It captures 25% of the
  skewness and 19% of the kurtosis, ends up only 1.3–2.4× better than a plain
  Gaussian (worse than order-3 Gram–Charlier), and — decisively — cannot even
  hold the covariance it is supposed to track. Not implemented beyond the
  feasibility test.

### Chosen next direction: polynomial transport map

Every attempt so far trades positivity against flexibility. A transport map
breaks the trade: `ζ = T(w)` with `w ~ N(0,I₂)` and `T` polynomial. The density
is a pushforward, hence **positive by construction** and unimodal whenever `T`
is a diffeomorphism (checkable from its Jacobian); moments are exact Gaussian
moments of a polynomial, hence **analytic, no quadrature, no realizability
cliff**; and a cubic `T` carries ~18 coefficients against the 12 moments needed
— surplus flexibility rather than a deficit, for the first time.

Cheaper fallback, if that proves too large a build: keep the order-4 cumulant
closure (its reconstruction is the best measured, 0.0065 at χ=1.2) and stop the
iterate leaving the realizable set — Hankel-matrix positivity as a constraint,
plus pseudo-arclength continuation and σ_min monitoring to traverse the fold
instead of jumping it.

## 6. Known issues

| # | Issue | Fix |
|---|---|---|
| 1 | `husimi_hill_method_test_kerr_3.ipynb` is **byte-identical** to `husimi_hill_method_test_kerr.ipynb` (same md5). It never received its third-order content. | delete it, or populate it |
| 2 | Solver code is **duplicated**: cell 0 of `husimi_hill_method_test_kerr.ipynb` *is* `hb_moments_kerr.py`. They can drift apart silently. | notebook should import the module |
| 3 | `wigner_gif/` is **84 MB of GIFs tracked in git**, in a repo whose source is ~1 MB. | `git rm --cached`, move to external storage or Git LFS |
| 4 | Three notebooks hard-code `sys.path.insert(0, "/Users/tmalas/...")` — breaks for anyone else. | editable install (§7), or a relative bootstrap |
| 5 | `Cat/Figures/` is silently **gitignored** by the monorepo's `**/Figures/` rule, so paper figures are not version-controlled. | rename the directory, or add a negation rule |
| 6 | The `dynamiqs-env` venv **lacks sympy**, so every solver module fails to import in the project's own kernel. | `~/venvs/dynamiqs-env/bin/pip install sympy` |
| 7 | `hill_method.py` has known internal discrepancies (eigenvalue count, reversed comments, `tau_j` spanning 50 periods, parameters differing from the reference set). | catalogued in `CLAUDE.md` §"Known discrepancies" |
| 8 | A tracked `__pycache__/hill_method.cpython-311.pyc` predates the new `.gitignore`. | `git rm --cached -r __pycache__` |

## 7. Proposed restructuring

The current layout is *workable* — each project directory is self-contained, so
notebooks can `import hb_moments_kerr` with no path setup. That property is
worth keeping until it's deliberately traded away. Three tiers, in increasing
order of disruption:

### Tier 1 — done in this pass, no breakage

- `.claude/settings.json`, `.gitignore`, this file.
- Generated artefacts consolidated into `Kerr_oscillator/results/{data,figures}`,
  with all three writers (`sweep_kerr_hellinger_n.py` and the two sweep
  notebooks) updated to resolve paths relative to the script and verified to
  run from any working directory.
- The `.npz` records in `results/data/` are the **regression baseline** and are
  tracked on purpose; `results/figures/` is ignored.

### Tier 2 — cheap, low risk, not yet done

- Fix issues 1, 3, 6, 8 above (all one-liners; 1 and 3 delete things, so they
  need a decision).
- Make `husimi_hill_method_test_kerr.ipynb` import `hb_moments_kerr` instead of
  inlining it (issue 2).

### Tier 3 — the package layout

Convert to an installable package. This is the "nice repo" end state, and the
only tier that breaks existing imports:

```
Hill_method/
├── pyproject.toml            → pip install -e .
├── src/hhb/
│   ├── hill_method.py        (shared Floquet/HB machinery)
│   ├── cat/hhb_cat.py
│   └── kerr/{moments_n2,moments_n3,moments_n,moments_maxent_n}.py
├── notebooks/{kerr,cat}/
├── scripts/sweep_kerr_hellinger_n.py
├── results/{data,figures}/
└── media/                    (the GIFs, if they stay in-tree at all)
```

Migration, in order:
1. Add `pyproject.toml`; `pip install -e .` into `dynamiqs-env` (this also
   fixes issue 6 by declaring `sympy`, `scipy`, `numpy`, `dynamiqs`).
2. `git mv` modules into `src/hhb/`, keeping **module names unchanged** so only
   the import prefix changes.
3. Rewrite notebook imports to `from hhb.kerr.moments_n import ...` and delete
   the three hard-coded `sys.path` inserts (issue 4).
4. Re-run every module self-test and the χ=0.33 sweep point; they are fast and
   catch path errors immediately.

**Worth noting:** this directory sits inside the `Simulations_local` monorepo
(git root is four levels up, alongside `Cheb_Ar/`, `FIF/`, `MPEP/`), and
`Plot_setting.py` — imported by two notebooks — lives in the parent
`Code_python/`. Extracting `Hill_method/` into a standalone repository is
therefore a larger change than Tier 3 and would need `Plot_setting` vendored or
packaged first. Tier 3 does not require it, and is worth doing first either way.
