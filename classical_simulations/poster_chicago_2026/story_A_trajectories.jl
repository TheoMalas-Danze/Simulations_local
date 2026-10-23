# Time integrations for story_A_v1.ipynb (23/10/2026), from a list of jobs written by the notebook.
# Same model as crisis_collision.jl (rescaled flux form, tau = w_a t, s = sin(eps_p)), kappa_b = pi/10.4 (code convention).
# States: [x_a, x_b, dx_a/dtau, dx_b/dtau] at tau = 0 mod 2 pi (x^lambda units). Three kinds of jobs:
#   "traj":  one trajectory, N_Ta memory periods, optional "pre" = [g2_pre, N_pre] (run first at g2_pre, then continue from there),
#            optional "dense" = [[start_Ta, length_Ta], ...] (40 samples per T_a). Section states: all iterates if N_Ta <= keep,
#            otherwise every m-th iterate (m odd) plus the last n_tail in full, with their index.
#   "basin": one row of a grid of initial states; for each, the last n_last section states after N_Ta periods (classified by the notebook).
#   "kick":  local basin of a stable orbit (section states "ref"): random kicks of size eps in n_dir directions; back = within 1e-5 of
#            a reference state without leaving a 0.3-neighbourhood, within N_max periods.
# Run: julia -t 8 --project=/Users/tmalas/Desktop/Dev/Simulations_local/Code_Julia story_A_trajectories.jl input.json output.json
using DifferentialEquations, JSON, LinearAlgebra, Random, Base.Threads

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
integrator(u, p) = init(ODEProblem(rhs!, copy(u), (0.0, 1e12), p), Vern9(); abstol = TOL, reltol = TOL, save_everystep = false, maxiters = 10^12)
dense(u, p, n) = solve(ODEProblem(rhs!, copy(u), (0.0, 2π*n), p), Vern9(); abstol = TOL, reltol = TOL, saveat = range(0, 2π*n, length = 40n + 1)).u
mind(u, S) = minimum(norm(u .- s) for s in S)

function run_traj(J; keep = 20000, n_tail = 2000)
    r = Float64(J["ratio"]); u = Float64.(J["u0"])
    if haskey(J, "pre")
        it = integrator(u, pars(Float64(J["pre"][1]), r))
        for _ in 1:Int(J["pre"][2]); step!(it, 2π, true); end
        u = copy(it.u)
    end
    p = pars(Float64(J["g2_MHz"]), r); N = Int(J["N_Ta"])
    windows = haskey(J, "dense") ? [(Int(w[1]), Int(w[2])) for w in J["dense"]] : Tuple{Int, Int}[]
    m = N <= keep ? 1 : cld(N - n_tail, keep - n_tail)
    m > 1 && iseven(m) && (m += 1)                 # odd stride: no aliasing with period-2 orbits
    idx = Int[0]; sec = [round.(u, digits = 6)]; dw = Dict{String, Any}()
    for (s, l) in windows
        s == 0 && (dw["0"] = [round.(x, digits = 5) for x in dense(u, p, l)])
    end
    it = integrator(u, p)
    for n in 1:N
        step!(it, 2π, true)
        if n % m == 0 || n > N - n_tail
            push!(idx, n); push!(sec, round.(copy(it.u), digits = 6))
        end
        for (s, l) in windows
            s == n && (dw[string(n)] = [round.(x, digits = 5) for x in dense(copy(it.u), p, l)])
        end
    end
    return Dict{String, Any}("index" => idx, "section" => sec, "dense" => dw)
end

function run_basin(J; n_last = 64)
    r = Float64(J["ratio"]); p = pars(Float64(J["g2_MHz"]), r); N = Int(J["N_Ta"])
    out = Any[]
    for u0 in J["u0s"]
        it = integrator(Float64.(u0), p); last = Vector{Vector{Float64}}()
        for n in 1:N
            step!(it, 2π, true)
            n > N - n_last && push!(last, round.(copy(it.u), digits = 6))
        end
        push!(out, last)
    end
    return Dict{String, Any}("row" => J["row"], "last" => out)
end

function run_kick(J)
    r = Float64(J["ratio"]); p = pars(Float64(J["g2_MHz"]), r); S = [Float64.(s) for s in J["ref"]]
    frac = Float64[]
    for (ie, ε) in enumerate(J["eps"])
        back = 0
        for k in 1:Int(J["n_dir"])
            u0 = S[1] .+ ε .* normalize(randn(MersenneTwister(1000ie + k), 4))
            it = integrator(u0, p)
            for _ in 1:Int(J["N_max_Ta"])
                step!(it, 2π, true); d = mind(it.u, S)
                d < 1e-5 && (back += 1; break)
                d > 0.3 && break
            end
        end
        push!(frac, back/Int(J["n_dir"]))
    end
    return Dict{String, Any}("eps" => J["eps"], "frac_back" => frac)
end

jobs = JSON.parsefile(ARGS[1])["jobs"]
println(length(jobs), " jobs on ", nthreads(), " threads"); flush(stdout)
res = Vector{Any}(undef, length(jobs)); t0 = time()
@threads :dynamic for j in eachindex(jobs)
    J = jobs[j]
    R = J["kind"] == "traj" ? run_traj(J) : J["kind"] == "basin" ? run_basin(J) : run_kick(J)
    res[j] = merge(R, Dict{String, Any}("id" => J["id"], "kind" => J["kind"], "g2_MHz" => J["g2_MHz"], "hash" => get(J, "hash", "")))
end
println("done in ", round(time() - t0, digits = 0), " s")
open(io -> JSON.print(io, Dict("T_a_over_kappa_inv" => 2π/(w_a/κ0), "runs" => res)), ARGS[2], "w")
