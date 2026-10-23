# Bifurcation diagrams of the attractor followed in g2, for several kappa_b (22/10/2026), and the largest Lyapunov exponent
# along three of them only (the slow part). Called by beyond_cusp_regimes.ipynb, which writes the seeds
# (beyond_cusp_seeds_22_10_2026.json) and plots the output (beyond_cusp_dynamics_22_10_2026.json).
#
# Model: the same equations as the Hill notebooks (full_rhs), rescaled flux form with tau = w_a t:
#   x_a'' = -x_a + sqrt2 beta_a' (cos(sqrt2 (x_a + x_b)) - 1)
#   x_b'' = 4 (-x_b - sqrt2 eps_d' cos 2tau + sqrt2 beta_b' (cos(...) - 1) - kappa' x_b')
# with s = sin(eps_p), beta' = 2 E_J s phi^2/w, eps_d' = r s G0 phi_b/w_b, kappa' = kfac kappa_b0/w_b (code convention, kappa_b0 = pi/10.4).
# s > 1 is allowed (formal continuation above g2_max = G0). The memory period T_a is 2pi in tau; the drive period is pi.
#
# Sections: the state is recorded at every drive period (tau = n pi). The two partners +-a of a period-1 orbit are the same orbit
# shifted by one drive period, so they give the same set of points on this section (on the section at T_a they do not).
# Periods are detected on the samples at multiples of T_a (every other sample; the last sample is at a multiple of T_a).
#
# Run: julia -t 8 --project=/Users/tmalas/Desktop/Dev/Simulations_local/Code_Julia beyond_cusp_dynamics.jl
using DifferentialEquations, ChaosTools, JSON, Statistics, LinearAlgebra, Base.Threads

const OUT_DIR = "/Users/tmalas/Desktop/Dev/Simulations_local/classical_simulations/data/"
const w_a = 25.338776456203686
const w_b = 2w_a
const phi_a = 0.11
const phi_b = 0.204
const E_J = 37.12*2π*1.5
const κ0 = π/10.4
const G0 = E_J*phi_a^2*phi_b
const r0 = 10.12
const TOL = 1e-10
MHz_to_s(g) = g*2π*1e-3/G0

function rhs!(du, u, p, t)
    x_a, x_b, v_a, v_b = u
    β_a, β_b, ed, kp = p
    g = cos(sqrt(2)*(x_a + x_b)) - 1
    du[1] = v_a
    du[2] = v_b
    du[3] = -x_a + sqrt(2)*β_a*g
    du[4] = 4*(-x_b - sqrt(2)*ed*cos(2t) + sqrt(2)*β_b*g - kp*v_b)
    nothing
end
pars(g, kfac) = (s = MHz_to_s(g); [2E_J*s*phi_a^2/w_a, 2E_J*s*phi_b^2/w_b, r0*s*G0*phi_b/w_b, kfac*κ0/w_b])

# states at tau = n pi (drive periods), from 2 n_tr to 2 (n_tr + n_rec): n_tr and n_rec are in memory periods T_a
function strobe(u0, g, kfac; n_tr, n_rec)
    prob = ODEProblem(rhs!, copy(u0), (0.0, 2π*(n_tr + n_rec)), pars(g, kfac))
    solve(prob, Vern9(); abstol = TOL, reltol = TOL, saveat = π .* (2n_tr:2(n_tr + n_rec)), maxiters = 10^9).u
end
amp_a(u) = (u[1] + 1im*u[3])/(sqrt(2)*phi_a)
# largest Lyapunov exponent in units of the code kappa_b (kfac kappa_b0); u0 is already on the attractor (at a multiple of T_a)
λ1(u0, g, kfac; N) = lyapunov(CoupledODEs(rhs!, copy(u0), pars(g, kfac); diffeq = (alg = Vern9(), abstol = TOL, reltol = TOL)),
                              2π*N; Δt = 2π, d0 = 1e-8, d0_upper = 1e-5, d0_lower = 1e-11)*w_a/(kfac*κ0)
# smallest p with A[n + p] = A[n] on the record (relative tolerance), 0 if none up to pmax
function period_of(A; tol = 1e-6, pmax = 64)
    s = mean(abs.(A))
    for p in 1:pmax
        maximum(abs.(A[1+p:end] .- A[1:end-p])) < tol*s && return p
    end
    return 0
end

# slow ramp from g_from to g_to: sub-steps of dg (MHz), n_sub memory periods each. A direct jump of 0.2 MHz can throw the
# trajectory out of a narrow basin (the satellite at kappa_b0 near g2_max), which a slow increase of g2 would not do.
function ramp(u, g_from, g_to, kfac; dg = 0.02, n_sub = 50)
    m = max(1, round(Int, abs(g_to - g_from)/dg))
    for j in 1:m-1
        u = strobe(u, g_from + j*(g_to - g_from)/m, kfac; n_tr = n_sub, n_rec = 0)[end]
    end
    return u
end

# parameter continuation: each g2 starts from the end state of the previous one, reached by a slow ramp.
# n_keep: number of samples (drive periods) kept for the diagram; N_λ > 0: largest Lyapunov exponent over N_λ T_a
function sweep(u0, gs, kfac; n_tr = 2000, n_rec = 400, n_keep = 0, n_first = 4000, N_λ = 0)
    u = copy(u0)
    re = Vector{Vector{Float64}}(); im_ = Vector{Vector{Float64}}()
    per = Int[]; amean = Float64[]; amin = Float64[]; amax = Float64[]; lam = Float64[]
    for (i, g) in enumerate(gs)
        i > 1 && (u = ramp(u, gs[i-1], g, kfac))
        U = strobe(u, g, kfac; n_tr = (i == 1 ? n_first : n_tr), n_rec = n_rec)
        u = U[end]
        A = amp_a.(U)
        push!(per, period_of(A[1:2:end]))                 # samples at multiples of T_a
        push!(amean, mean(abs.(A))); push!(amin, minimum(abs.(A))); push!(amax, maximum(abs.(A)))
        N_λ > 0 && push!(lam, λ1(u, g, kfac; N = N_λ))
        if n_keep > 0
            push!(re, round.(real.(A[end-n_keep+1:end]), digits = 4)); push!(im_, round.(imag.(A[end-n_keep+1:end]), digits = 4))
        end
    end
    out = Dict("g2_MHz" => gs, "period_Ta" => per, "abs_a_mean" => amean, "abs_a_min" => amin, "abs_a_max" => amax,
               "re_a" => re, "im_a" => im_)
    N_λ > 0 && (out["lambda1_over_kappa"] = lam)
    return out
end

seeds = JSON.parsefile(OUT_DIR*"torus_crisis/beyond_cusp_seeds_22_10_2026.json")
g2_max = seeds["g2_max_MHz"]
g_down = seeds["g_down_MHz"]                            # start of the down sweeps (just below g2_max), where the satellite seeds are
u_cat = [0.0, 0.0, sqrt(2)*phi_a*sqrt(r0/2), 0.0]       # a(0) = i|alpha|, b(0) = 0: ends on a cat at 20 MHz (checked in the output)
y_sat(k) = Float64.(seeds["satellite_at_g2max"][string(k)])
seed(name) = Float64.(seeds[name])

# tasks: (group, kfac, name, seed, g2 grid, options). Diagrams: 160 samples kept = 80 T_a.
tasks = Any[]
for k in seeds["kappa_diagrams"]
    push!(tasks, ("diagram", k, "up from the cat", u_cat, collect(20.0:0.2:180.0), (n_keep = 160, N_λ = (k == 11.0 ? 2000 : 0))))
    push!(tasks, ("diagram", k, "down from the satellite", y_sat(k), collect(g_down:-0.2:20.0), (n_keep = 160,)))
end
push!(tasks, ("diagram", 1.0, "up from the satellite", seed("satellite_45MHz_kappa1"), collect(45.0:0.2:180.0), (n_keep = 160,)))
push!(tasks, ("diagram", 8.0, "up from the satellite", seed("satellite_103MHz_kappa8"), collect(103.0:0.2:180.0), (n_keep = 160, N_λ = 2000)))
# regime A: the window of torus_breakdown.ipynb (satellite up, 74.9-79 MHz in steps of 20 kHz, 3000 T_a per step, section over 600 T_a)
push!(tasks, ("window", 1.0, "up from the satellite", seed("satellite_74.9MHz_kappa1"), collect(74.9:0.02:79.0),
              (n_tr = 3000, n_rec = 600, n_keep = 1201, N_λ = 2000)))
for k in seeds["kappa_map"]
    push!(tasks, ("map", k, "up from the cat", u_cat, collect(20.0:0.5:180.0), (n_tr = 1000, n_rec = 200)))
    push!(tasks, ("map", k, "down from the satellite", y_sat(k), collect(g_down:-0.5:20.0), (n_tr = 1000, n_rec = 200)))
end

println("threads: ", nthreads(), ", tasks: ", length(tasks)); flush(stdout)
results = Vector{Any}(undef, length(tasks))
t0 = time()
@threads :dynamic for i in eachindex(tasks)
    group, k, name, u0, gs, opt = tasks[i]
    results[i] = sweep(u0, gs, k; opt...)
    group != "map" && (println("  done: ", group, " kappa x ", k, ", ", name, " (", round(time() - t0, digits = 0), " s)"); flush(stdout))
end
println("all done in ", round(time() - t0, digits = 0), " s")

out = Dict("description" => "Attractor followed in g2 at eps_d = 10.12 g2 (slow ramp in 20 kHz sub-steps of 50 T_a, then 2000 T_a (diagrams), 3000 T_a (window) " *
                            "or 1000 T_a (maps) at each recorded g2): a(n pi) at every drive period (re_a, im_a: last samples, the last one at a multiple of T_a), " *
                            "period on the section at T_a (0: none up to 64, i.e. torus or chaos), largest Lyapunov exponent in units of the code kappa_b " *
                            "(kfac pi/10.4) on three sweeps. From beyond_cusp_dynamics.jl, called by beyond_cusp_regimes.ipynb",
           "g2_max_MHz" => g2_max, "kappa_b0_code" => κ0, "T_a_ns" => 2π/w_a,
           [grp => [Dict("kappa_over_kappa0" => t[2], "sweep" => t[3], "data" => results[i]) for (i, t) in enumerate(tasks) if t[1] == g]
            for (grp, g) in (("diagrams", "diagram"), ("window", "window"), ("maps", "map"))]...)
open(io -> JSON.print(io, out), OUT_DIR*"torus_crisis/beyond_cusp_dynamics_22_10_2026.json", "w")
