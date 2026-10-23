# Lifetime of the chaotic transient just below the end of the torus (g_c = 77.249 MHz at eps_d = 10.12 g2, torus_breakdown.ipynb), 23/10/2026.
# Called by torus_transient_lifetime.ipynb, which plots the output (torus_transient_lifetime_23_10_2026.json).
#
# Model: the same equations as beyond_cusp_dynamics.jl / crisis_collision.jl (rescaled flux form, tau = w_a t, s = sin(eps_p)),
# kappa_b = pi/10.4 (code convention). Section tau = 0 mod 2 pi, state y = [x_a, x_b, dx_a/dtau, dx_b/dtau] (x^lambda units),
# distances are Euclidean in y. Every attractor is represented by its states at every drive period (tau = n pi), i.e. its section
# points and those of its partner (shift by T_a/2).
#
# Tasks, for each g2 of the grid:
#   reference: the torus (period-2 orbit below 76.70 MHz), followed in g2 from the HB satellite at 74.9 MHz; the cat, from a = +-i|alpha|, b = 0;
#   lifetime:  n_traj runs from points of the chaotic attractor at 77.3 MHz, up to N_max T_a. "Landed on the torus" = within 0.01 of it for
#              M consecutive T_a (landing time = first of them); "landed on the cat" = within 1e-3 of it. Histograms of the distances to the
#              torus and to the cat over the chaotic part of each run (from iterate 50 to 200 iterates before landing);
#   kicks:     local basins of the torus and of the cat: kicks of size eps in random directions from points of the attractor; "back" = within
#              0.005 (torus) or 1e-3 (cat) for 300 T_a, without leaving a 0.3-neighbourhood.
# Run: julia -t 8 --project=/Users/tmalas/Desktop/Dev/Simulations_local/Code_Julia torus_transient_lifetime.jl
using DifferentialEquations, JSON, LinearAlgebra, Random, Statistics, Base.Threads

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
pars(g) = (s = g*2π*1e-3/G0; [2E_J*s*phi_a^2/w_a, 2E_J*s*phi_b^2/w_b, r0*s*G0*phi_b/w_b, κ0/w_b])
integrator(u, g) = init(ODEProblem(rhs!, copy(u), (0.0, 1e12), pars(g)), Vern9(); abstol = TOL, reltol = TOL, save_everystep = false, maxiters = 10^12)
next!(it, dt = 2π) = (step!(it, dt, true); it.u)                 # one memory period T_a (or one drive period with dt = pi)
relax!(it, n) = (for _ in 1:n; next!(it); end; copy(it.u))
drive_samples(u, g, n) = (it = integrator(u, g); [copy(next!(it, Float64(π))) for _ in 1:n])
mind(u, S) = minimum(norm(u .- s) for s in S)
const EDGES = 10 .^ range(-4, 1, length = 51)                    # log bins of the distance
binof(d) = clamp(searchsortedlast(EDGES, d), 1, length(EDGES) - 1)

const G = [76.6, 76.8, 77.0, 77.1, 77.15, 77.2, 77.22, 77.23, 77.24, 77.244, 77.246, 77.248]
const N_TRAJ, N_MAX, M_LAND = 64, 300_000, 1000
const EPS = 10 .^ range(-3.5, -0.5, length = 13)
const N_KICK_PTS, N_KICK_DIR, N_KICK_MAX = 16, 4, 20_000

# ---------- references ----------
TOR = Dict{Float64, Vector{Vector{Float64}}}()
let g_prev = 74.9, u = [0.5780652298528897, 0.49470486744052217, -0.1252445559440952, -0.5917230870531783]   # HB satellite at 74.9 MHz (beyond_cusp_seeds_22_10_2026.json)
    for g in G
        for gg in range(g_prev, g, length = max(2, round(Int, (g - g_prev)/0.02) + 1))[2:end]
            u = relax!(integrator(u, gg), 3000)
        end
        u = relax!(integrator(u, g), 10000)
        TOR[g] = drive_samples(u, g, 4000)                        # 2000 T_a, orbit and partner
        g_prev = g
    end
end
CAT = Dict(g => vcat([drive_samples(relax!(integrator([0.0, 0.0, sg*sqrt(2)*phi_a*sqrt(r0/2), 0.0], g), 3000), g, 2) for sg in (1, -1)]...) for g in G)
it0 = integrator(zeros(4), 77.3); relax!(it0, 3000)
CH = [relax!(it0, 97) for _ in 1:N_TRAJ]                            # chaotic attractor at 77.3 MHz (origin start)
println("references done"); flush(stdout)

# ---------- lifetimes ----------
function run_lifetime(g, u0)
    it = integrator(u0, g); S = TOR[g]; C = CAT[g]
    dT = Float64[]; dC = Float64[]; inside = 0; fate = "none"; t = N_MAX
    for n in 1:N_MAX
        y = next!(it)
        d = mind(y, S); c = mind(y, C)
        push!(dT, d); push!(dC, c)
        if c < 1e-3
            fate, t = "cat", n; break
        end
        inside = d < 0.01 ? inside + 1 : 0
        if inside >= M_LAND
            fate, t = "torus", n - M_LAND + 1; break
        end
    end
    stop = fate == "none" ? length(dT) : t - 200
    hT = zeros(Int, length(EDGES) - 1); hC = zeros(Int, length(EDGES) - 1)
    for k in 50:stop
        hT[binof(dT[k])] += 1; hC[binof(dC[k])] += 1
    end
    return Dict("fate" => fate, "t_Ta" => t, "hist_torus" => hT, "hist_cat" => hC,
                "dmin_torus" => (stop >= 50 ? minimum(dT[50:stop]) : NaN), "dmin_cat" => (stop >= 50 ? minimum(dC[50:stop]) : NaN))
end

# ---------- kicks ----------
function run_kick(g, which, k, ε, seed)
    S = which == "torus" ? TOR[g] : CAT[g]; tol = which == "torus" ? 0.005 : 1e-3
    y0 = S[which == "torus" ? 1 + (k - 1)*(length(S) ÷ N_KICK_PTS) : 1 + (k - 1) % 2]
    it = integrator(y0 .+ ε .* normalize(randn(MersenneTwister(seed), 4)), g); inside = 0
    for n in 1:N_KICK_MAX
        y = next!(it, 2π); d = mind(y, S)
        d > 0.3 && return false
        inside = d < tol ? inside + 1 : 0
        inside >= 300 && return true
    end
    return false
end

jobs = Any[]
for g in G, j in 1:N_TRAJ
    push!(jobs, ("lifetime", g, j))
end
for g in G, which in ("torus", "cat"), ε in EPS, k in 1:N_KICK_PTS, d in 1:N_KICK_DIR
    push!(jobs, ("kick", g, (which, k, ε, 1000k + d)))
end
println(length(jobs), " jobs on ", nthreads(), " threads"); flush(stdout)
res = Vector{Any}(undef, length(jobs)); t0 = time()
@threads :dynamic for j in reverse(eachindex(jobs))                # longest runs (closest to g_c) first
    kind, g, a = jobs[j]
    res[j] = kind == "lifetime" ? merge(run_lifetime(g, CH[a]), Dict("task" => kind, "g2_MHz" => g, "start" => a)) :
             Dict("task" => kind, "g2_MHz" => g, "which" => a[1], "point" => a[2], "eps" => a[3], "back" => run_kick(g, a...))
end
println("done in ", round(time() - t0, digits = 0), " s"); flush(stdout)

out = Dict("description" => "Lifetime of the chaotic transient below the end of the torus at eps_d = 10.12 g2 (kappa_b = pi/10.4 code), from the chaotic attractor " *
                            "at 77.3 MHz; landing on the torus or the cat; distance histograms over the chaotic part of each run; local basins (kicks). " *
                            "From torus_transient_lifetime.jl, called by torus_transient_lifetime.ipynb",
           "ratio" => r0, "g2_MHz" => G, "N_max_Ta" => N_MAX, "M_land_Ta" => M_LAND, "edges" => collect(EDGES), "eps" => collect(EPS),
           "T_a_over_kappa_inv" => 2π*κ0/w_a,
           "torus" => Dict(string(g) => [round.(y, digits = 6) for y in TOR[g][1:4:end]] for g in G),
           "cat" => Dict(string(g) => CAT[g] for g in G), "runs" => res)
open(io -> JSON.print(io, out), OUT_DIR*"torus_crisis/torus_transient_lifetime_23_10_2026.json", "w")
