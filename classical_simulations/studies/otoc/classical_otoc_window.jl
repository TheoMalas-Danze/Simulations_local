# Classical counterpart of the OTOC of Quantum_codes/OTOCS_poster_init_coherent.ipynb, on the same grid and time window.
#
# Quantum: C(t) = -<[x_a(t), p_a(0)]^2>, x_a(t) in the rotating frame of the memory, from |i alpha>|0>, fitted as
# log C against kappa_b t on t = j tau/8 (tau = 2pi/w_b), over 3 drive periods (kappa_b T = 0.225); rate = slope/2.
# Classical: the Poisson bracket {x_a(t), p_a(0)} = d x_a^rot(t) / d x_a(0), so C_cl(t) = (d x_a^rot(t)/d x_a(0))^2,
# from the tangent dynamics of equation_1ODE! (the EOM of the Lyapunov maps). Two initial conditions:
#   point: a(0) = i|alpha|, b(0) = 0, as in lyapunov_heatmap_cat_start.ipynb;
#   TWA:   C_cl averaged over the Wigner function of |i alpha>|0> (vacuum noise, no noise from the loss).
# Rates are fitted exactly as the quantum ones, over 3, 30 and 300 drive periods (point) and 3, 30 (TWA).
#
# State of equation_1ODE!: u = [ϕ_a, ϕ_b, ϕ̇_a, ϕ̇_b] = √2 [Re a, Re b, ω_a Im a, ω_b Im b] (get_init with
# ϕ_0 = sqrt(2Z) Re a, q_0 = sqrt(2/Z) Im a), so x_a^rot(t) = √2 Re(a e^{iω_a t}) = ϕ_a cos(ω_a t) - ϕ̇_a sin(ω_a t)/ω_a.
#
# Run: julia -t 10 --project=/Users/tmalas/Desktop/Dev/Simulations_local/Code_Julia classical_otoc_window.jl

using ChaosTools, LinearAlgebra, JSON, Random, Statistics
include("/Users/tmalas/Desktop/Dev/Simulations_local/classical_simulations/simulator.jl")

const out_dir = "/Users/tmalas/Desktop/Dev/Simulations_local/classical_simulations/data/"

const ω_a = 25.338776456203686
const ω_b = 2ω_a
const ϕ_quantum = 1/2
const φ_a, φ_b = 0.11, 0.204
const Z_a, Z_b = 2ϕ_quantum^2*φ_a^2, 2ϕ_quantum^2*φ_b^2
const E_Ca, E_Cb = Z_a*ω_a/2, Z_b*ω_b/2
const E_J = 37.12*2π*1.5
const κ_code = π/10.4        # Julia convention (-2κ ϕ̇_b), i.e. κ_b/2
const κ_b = 2κ_code          # GKSL rate, the one the OTOC times are normalised by
const τ = 2π/ω_b
const n_sub = 8
const windows_point = (3, 30, 300)   # drive periods
const windows_twa = (3, 30)
const n_twa = 64

g2_of(ϵ_p) = sin(ϵ_p)*E_J*φ_a^2*φ_b

function make_params(ϵ_p, ratio)
    drive = DriveTwoModesBasics(ratio*g2_of(ϵ_p), ω_b, 0.0, ϵ_p, 0.0, 0.0)   # ω_p = 2ω_a - ω_d = 0
    system = prepare(SystemATS(ω_a, ω_b, E_J, E_Ca, E_Cb, 0.0, 0.0, 0.0, 0.0))
    SimulationParams(system, prepare(drive), MeanField(0.0, κ_code))
end

state_of(a, b) = [√2*real(a), √2*real(b), √2*ω_a*imag(a), √2*ω_b*imag(b)]

"""log C_cl(t_j), t_j = j τ/n_sub, j = 0..n_periods n_sub, from (a0, b0)."""
function log_otoc(p, a0, b0, n_periods)
    ds = CoupledODEs(equation_1ODE!, state_of(a0, b0), p; diffeq=(alg=Vern9(), abstol=1e-11, reltol=1e-11))
    tds = TangentDynamicalSystem(ds; k=1, Q0=reshape([1.0, 0.0, 0.0, 0.0], 4, 1))   # δ along x_a(0) = ϕ_a(0)
    ts = (0:n_periods*n_sub) .* (τ/n_sub)
    logC = similar(ts, Float64)
    log_scale = 0.0
    for (j, t) in enumerate(ts)
        j > 1 && step!(tds, t - current_time(tds), true)
        Q = current_deviations(tds)
        δx = Q[1, 1]*cos(ω_a*t) - Q[3, 1]*sin(ω_a*t)/ω_a
        logC[j] = 2*(log(abs(δx)) + log_scale)
        nQ = norm(Q)
        if nQ > 1e50   # keep the linear deviation finite over 300 periods
            set_deviations!(tds, Q ./ nQ)
            log_scale += log(nQ)
        end
    end
    return collect(ts), logC
end

"""Rate as in fit_log_otoc of the quantum notebook: slope of log C against κ_b t, halved; and R²."""
function fit_rate(ts, logC)
    x = ts .* κ_b
    X = hcat(x, ones(length(x)))
    m, c = X \ logC
    res = logC .- (m .* x .+ c)
    return m/2, 1 - sum(abs2, res)/sum(abs2, logC .- mean(logC))
end

window(ts, logC, n_periods) = (ts[1:n_periods*n_sub+1], logC[1:n_periods*n_sub+1])

function run_point(ϵ_p, ratio; seed)
    p = make_params(ϵ_p, ratio)
    α = sqrt(ratio/2)
    ts, logC = log_otoc(p, 1im*α, 0.0im, maximum(windows_point))
    point = [fit_rate(window(ts, logC, w)...) for w in windows_point]

    # TWA: vacuum Wigner noise on both modes, Re and Im of δa with variance 1/4
    rng = Xoshiro(seed)
    n_long = maximum(windows_twa)
    C_sum = zeros(n_long*n_sub + 1)
    for _ in 1:n_twa
        a0 = 1im*α + (randn(rng) + 1im*randn(rng))/2
        b0 = (randn(rng) + 1im*randn(rng))/2
        ts_s, logC_s = log_otoc(p, a0, b0, n_long)
        C_sum .+= exp.(logC_s)
    end
    ts_t = ts[1:n_long*n_sub+1]
    logC_twa = log.(C_sum ./ n_twa)
    twa = [fit_rate(window(ts_t, logC_twa, w)...) for w in windows_twa]
    return point, twa, logC[1:windows_point[1]*n_sub+1], logC_twa[1:windows_twa[1]*n_sub+1]
end

function main(; quick=false)
    q = JSON.parsefile(out_dir*"otoc/lyap_best_init_coherent_15_10_2026.json")
    ϵ_p_list = round.(Float64.(q["epsilon_0"]), digits=2)
    ratio_list = collect(2.0:1.0:14.0)
    if quick   # one regular and one chaotic point, against the quantum rates
        for (ϵ_p, ratio) in ((0.2, 9.0), (0.6, 9.0), (0.93, 9.0))
            point, twa, _, _ = run_point(ϵ_p, ratio; seed=1)
            println("ϵ_p = $ϵ_p, ratio = $ratio, g2/2π = $(round(g2_of(ϵ_p)/2π*1e3, digits=1)) MHz: ",
                    "point ", [round(r[1], digits=3) for r in point], "  TWA ", [round(r[1], digits=3) for r in twa])
        end
        return
    end
    n_e, n_r = length(ϵ_p_list), length(ratio_list)
    rate_point = zeros(length(windows_point), n_e, n_r); R2_point = similar(rate_point)
    rate_twa = zeros(length(windows_twa), n_e, n_r); R2_twa = similar(rate_twa)
    logC_point_3 = Array{Vector{Float64}}(undef, n_e, n_r); logC_twa_3 = similar(logC_point_3)
    idx = vec(collect(Iterators.product(1:n_e, 1:n_r)))
    done = Threads.Atomic{Int}(0)
    t0 = time()
    Threads.@threads :dynamic for (i, j) in idx
        point, twa, lp, lt = run_point(ϵ_p_list[i], ratio_list[j]; seed=1000i + j)
        for (w, r) in enumerate(point); rate_point[w, i, j], R2_point[w, i, j] = r; end
        for (w, r) in enumerate(twa); rate_twa[w, i, j], R2_twa[w, i, j] = r; end
        logC_point_3[i, j], logC_twa_3[i, j] = lp, lt
        n = Threads.atomic_add!(done, 1) + 1
        n % 20 == 0 && println("$n/$(length(idx)) points, $(round(time() - t0, digits=1)) s")
    end
    # JSON layout as the quantum files: [i_ϵ_p][j_ratio]
    rows(A) = [[A[i, j] for j in 1:n_r] for i in 1:n_e]
    out = Dict(
        "description" => "Classical OTOC C_cl(t) = (d x_a^rot(t)/d x_a(0))^2 from tangent dynamics of equation_1ODE! " *
            "(classical_otoc_window.jl). rate = slope(log C vs kappa_b t)/2 with kappa_b = 2pi/10.4 (GKSL), as the quantum OTOC. " *
            "point: a(0) = i alpha, b(0) = 0; twa: mean of C_cl over the Wigner function of |i alpha>|0> ($n_twa samples). " *
            "Window = number of drive periods tau = 2pi/w_b, n_sub = $n_sub points per period. Arrays are [i_epsilon_0][j_ratio].",
        "epsilon_0" => ϵ_p_list, "ratio" => ratio_list, "kappa_b" => κ_b, "n_sub" => n_sub, "n_twa" => n_twa,
        "windows_point" => collect(windows_point), "windows_twa" => collect(windows_twa),
        "rate_point" => [rows(rate_point[w, :, :]) for w in eachindex(windows_point)],
        "R_sq_point" => [rows(R2_point[w, :, :]) for w in eachindex(windows_point)],
        "rate_twa" => [rows(rate_twa[w, :, :]) for w in eachindex(windows_twa)],
        "R_sq_twa" => [rows(R2_twa[w, :, :]) for w in eachindex(windows_twa)],
        "logC_point_3" => rows(logC_point_3), "logC_twa_3" => rows(logC_twa_3),
    )
    path = out_dir*"otoc/classical_otoc_window_21_10_2026.json"
    open(io -> JSON.print(io, out), path, "w")
    println("wrote $path in $(round(time() - t0, digits=1)) s")
end

if abspath(PROGRAM_FILE) == @__FILE__   # not when included
    main(quick = "--quick" in ARGS)
end
