# Lifetime of the chaos beyond the crisis of the bubble's chaotic attractor (22/10/2026), at fixed eps_d/g2.
# Called by satellite_start_bubble_edge.ipynb (section 4), which writes the input (bubble_crisis_input_22_10_2026.json).
# Same model as the Hill notebooks (rescaled flux form, tau = w_a t, s = sin(eps_p)), kappa_b = pi/10.4 (code convention).
#
# Two families of initial states, run at each g2 of the input grid:
#   "attractor": points of the bubble's chaotic attractor, sampled at g0 (inside the bubble) every 50 T_a;
#   "seed":      the analytic satellite of App. "Satellites" at that g2 (the initial state of the 12/10 map), plus random kicks of 1e-3.
# Escape: the trajectory has settled on a periodic orbit (period p <= 8 on the section at T_a, 100 iterates within 1e-6 relative);
# the escape time is the first iterate within 1e-3 of that orbit. Censored at N_max memory periods.
# Run: julia -t 8 --project=/Users/tmalas/Desktop/Dev/Simulations_local/Code_Julia bubble_crisis_lifetime.jl
using DifferentialEquations, ChaosTools, JSON, Statistics, LinearAlgebra, Random, Base.Threads

const OUT_DIR = "/Users/tmalas/Desktop/Dev/Simulations_local/classical_simulations/analysis/output/chicago_poster_2026/"
const w_a = 25.338776456203686
const w_b = 2w_a
const phi_a = 0.11
const phi_b = 0.204
const E_J = 37.12*2π*1.5
const κ0 = π/10.4
const G0 = E_J*phi_a^2*phi_b
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
pars(g, r) = (s = g*2π*1e-3/G0; [2E_J*s*phi_a^2/w_a, 2E_J*s*phi_b^2/w_b, r*s*G0*phi_b/w_b, κ0/w_b])
strobes(u, p, N) = solve(ODEProblem(rhs!, copy(u), (0.0, 2π*N), p), Vern9(); abstol = TOL, reltol = TOL, saveat = 2π .* (0:N), maxiters = 10^9).u
amp(u) = (u[1] + 1im*u[3])/(sqrt(2)*phi_a)

# period (<= 8) of the last 100 iterates, 0 if not settled
function settled_period(A)
    s = mean(abs.(A))
    for p in 1:8
        maximum(abs.(A[end-99+p:end] .- A[end-99:end-p])) < 1e-6*s && return p
    end
    return 0
end

# integrates in chunks of 500 T_a until settled or N_max; returns escape time (T_a, or nothing if censored),
# period and <|a|> of the final orbit, and the chaotic part of the section (iterates 50 .. escape, at most n_keep)
function escape(u0, p; N_max, chunk = 500, n_keep = 400)
    U = [copy(u0)]
    A = ComplexF64[amp(u0)]
    while length(A) - 1 < N_max
        V = strobes(U[end], p, chunk)
        append!(U, V[2:end]); append!(A, amp.(V[2:end]))
        length(A) > 200 || continue
        per = settled_period(A)
        if per > 0
            cyc = A[end-per+1:end]
            k = findfirst(a -> minimum(abs.(cyc .- a)) < 1e-3, A)
            ch = A[min(51, k):k]
            return (k - 1, per, mean(abs.(cyc)), ch[1:max(1, length(ch) ÷ n_keep):end])
        end
        length(U) > 2 && (U = U[end:end])          # keep memory small
    end
    ch = A[51:end]
    return (nothing, 0, NaN, ch[1:max(1, length(ch) ÷ n_keep):end])
end

inp = JSON.parsefile(OUT_DIR*"bubble_crisis_input_22_10_2026.json")
r = inp["ratio"]; g0 = inp["g0_MHz"]; gs = Float64.(inp["g2_MHz"]); N_max = inp["N_max_Ta"]; n_traj = inp["n_traj"]
seeds = [Float64.(y) for y in inp["seed_analytic"]]

# chaotic attractor at g0: from the analytic seed at g0, 2000 T_a of transient, then n_traj points every 50 T_a
p0 = pars(g0, r)
u = strobes(Float64.(inp["seed_g0"]), p0, 2000)[end]
λ0 = lyapunov(CoupledODEs(rhs!, copy(u), p0; diffeq = (alg = Vern9(), abstol = TOL, reltol = TOL)), 2π*2000; Δt = 2π, d0 = 1e-8, d0_upper = 1e-5)*w_a/κ0
V = strobes(u, p0, 50*n_traj)
att = [V[1 + 50*k] for k in 0:n_traj-1]
println("attractor at g0 = ", g0, " MHz: lambda_1/kappa_b = ", round(λ0, digits = 3)); flush(stdout)

tasks = [(ig, fam, k) for ig in eachindex(gs) for fam in ("attractor", "seed") for k in 1:n_traj]
res = Vector{Any}(undef, length(tasks))
rng_seeds = rand(MersenneTwister(1), UInt32, length(tasks))
t0 = time()
@threads :dynamic for n in eachindex(tasks)
    ig, fam, k = tasks[n]
    p = pars(gs[ig], r)
    u0 = fam == "attractor" ? att[k] : seeds[ig] .+ 1e-3 .* randn(MersenneTwister(rng_seeds[n]), 4)
    t, per, a_end, ch = escape(u0, p; N_max = N_max)
    res[n] = Dict("g2_MHz" => gs[ig], "family" => fam, "t_esc_Ta" => t, "period" => per, "abs_a_end" => a_end,
                  "chaos_re" => round.(real.(ch), digits = 4), "chaos_im" => round.(imag.(ch), digits = 4))
end
println("done in ", round(time() - t0, digits = 0), " s")
open(io -> JSON.print(io, Dict("ratio" => r, "g0_MHz" => g0, "lambda_g0_over_kappa" => λ0, "N_max_Ta" => N_max,
                               "T_a_over_kappa_inv" => 2π/(w_a/κ0), "runs" => res)),
     OUT_DIR*"bubble_crisis_lifetime_22_10_2026.json", "w")
