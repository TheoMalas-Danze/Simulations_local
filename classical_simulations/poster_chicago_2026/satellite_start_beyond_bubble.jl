# Satellite-start map above the period-doubling bubble (22/10/2026): why the 12_10 map is chaotic there and the 22_10 map is not.
# Called by satellite_start_bubble_edge.ipynb, which writes the points (satellite_start_beyond_bubble_points_22_10_2026.json).
# Same model as the Hill notebooks (rescaled flux form, tau = w_a t, s = sin(eps_p)), kappa_b = pi/10.4 (code convention).
# For each point: lambda_1 from the analytic seed with Ttr = 15/kappa_b (protocol of the 12_10 map) and 300/kappa_b (22_10 map),
# lambda_1 from the exact HB satellite with Ttr = 15/kappa_b, the fate of the analytic-seed trajectory over 2000/kappa_b
# (landing time on the satellite pair, lambda_1 over the last 100/kappa_b), and whether kicks of size eps from the exact satellite,
# along the direction of the analytic seed, come back to it within 300/kappa_b.
# Run: julia -t 8 --project=/Users/tmalas/Desktop/Dev/Simulations_local/Code_Julia satellite_start_beyond_bubble.jl
using DifferentialEquations, ChaosTools, JSON, Statistics, LinearAlgebra, Base.Threads

const OUT_DIR = "/Users/tmalas/Desktop/Dev/Simulations_local/classical_simulations/analysis/output/chicago_poster_2026/"
const w_a = 25.338776456203686
const w_b = 2w_a
const phi_a = 0.11
const phi_b = 0.204
const E_J = 37.12*2π*1.5
const κ0 = π/10.4
const G0 = E_J*phi_a^2*phi_b
const TOL = 1e-10
const KT = w_a/κ0                       # 1/kappa_b in units of tau
const EPS_KICK = [0.003, 0.01, 0.03, 0.1]

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
pars(s, r) = [2E_J*s*phi_a^2/w_a, 2E_J*s*phi_b^2/w_b, r*s*G0*phi_b/w_b, κ0/w_b]
# largest Lyapunov exponent in units of kappa_b; T and Ttr in units of 1/kappa_b
lyap(u, p; T, Ttr) = lyapunov(CoupledODEs(rhs!, copy(u), p; diffeq = (alg = Vern9(), abstol = TOL, reltol = TOL)),
                              T*KT; Δt = 2π, Ttr = Ttr*KT, d0 = 1e-7, d0_upper = 1e-4)*w_a/κ0
# states at tau = 2 pi n, n = 0..N
strobes(u, p, N) = solve(ODEProblem(rhs!, copy(u), (0.0, 2π*N), p), Vern9(); abstol = TOL, reltol = TOL, saveat = 2π .* (0:N), maxiters = 10^9).u

pts = JSON.parsefile(OUT_DIR*"satellite_start_beyond_bubble_points_22_10_2026.json")
res = Vector{Any}(undef, length(pts))
t0 = time()
@threads :dynamic for n in eachindex(pts)
    P = pts[n]
    p = pars(P["s"], P["ratio"])
    y_an, y_ex = Float64.(P["y_analytic"]), Float64.(P["y_exact"])
    y_pm = solve(ODEProblem(rhs!, y_ex, (0.0, π), p), Vern9(); abstol = TOL, reltol = TOL).u[end]   # partner: one drive period later
    dist(u) = min(norm(u - y_ex), norm(u - y_pm))
    λ15, λ300, λ15_ex = lyap(y_an, p; T = 30, Ttr = 15), lyap(y_an, p; T = 30, Ttr = 300), lyap(y_ex, p; T = 30, Ttr = 15)
    U = strobes(y_an, p, round(Int, 2000*KT/2π))
    k = findfirst(u -> dist(u) < 1e-3, U)
    λ_end = lyap(U[end], p; T = 100, Ttr = 0)
    dir = (y_an - y_ex)/norm(y_an - y_ex)
    back = [dist(strobes(y_ex + e*dir, p, round(Int, 300*KT/2π))[end]) < 1e-3 for e in EPS_KICK]
    res[n] = Dict("j" => P["j"], "i" => P["i"], "ratio" => P["ratio"], "g2_MHz" => P["g2_MHz"], "dist_seed" => norm(y_an - y_ex),
                  "lam15" => λ15, "lam300" => λ300, "lam15_exact" => λ15_ex,
                  "t_land_over_kappa" => isnothing(k) ? nothing : (k - 1)*2π/KT, "lam_end" => λ_end, "kick_returns" => back)
end
println("done in ", round(time() - t0, digits = 0), " s")
open(io -> JSON.print(io, Dict("eps_kick" => EPS_KICK, "points" => res)),
     OUT_DIR*"satellite_start_beyond_bubble_22_10_2026.json", "w")
