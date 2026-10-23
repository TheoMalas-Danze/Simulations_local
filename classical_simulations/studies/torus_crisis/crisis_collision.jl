# Boundary crisis of the bubble's chaotic attractor at eps_d = 10 g2 (23/10/2026): what does the attractor collide with?
# Called by crisis_collision.ipynb, which writes the input (section states, multipliers and leading eigenvectors of the candidate
# orbits from harmonic balance: omega_a/2 node and saddle, branch I, satellite) on a grid of g2.
# Same model as bubble_crisis_lifetime.jl (rescaled flux form, tau = w_a t, s = sin(eps_p)), kappa_b = pi/10.4 (code convention).
# Section: tau = 0 mod 2 pi, state [x_a, x_b, dx_a/dtau, dx_b/dtau] (x^lambda units). "Landed" = within 1e-3 of the node (orbit or partner).
#
# Tasks (all on the section):
#   "attractor": long runs from the bubble's attractor (taken at g0 = 100 MHz) for g2 <= g_max_att: histogram of the distance
#                to each candidate orbit, minimum distance, landing time if any, subsample of the points;
#   "escape":    runs above the crisis until they land on the node; the last n_pre iterates before landing are kept;
#   "manifold":  both branches of the unstable manifold of the omega_a/2 saddle (saddle state +- delta v_u): landing time or not;
#   "ns":        near the node around its Neimark-Sacker point: distance to the node vs time from a small kick (below g_NS),
#                and return or not (without leaving a 0.3 neighbourhood) after kicks of size eps (above g_NS).
# Run: julia -t 8 --project=/Users/tmalas/Desktop/Dev/Simulations_local/Code_Julia crisis_collision.jl [input.json output.json]
using DifferentialEquations, JSON, LinearAlgebra, Random, Statistics, Base.Threads

const OUT_DIR = "/Users/tmalas/Desktop/Dev/Simulations_local/classical_simulations/data/"
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
function next!(it)                     # one memory period T_a
    step!(it, 2π, true)
    return it.u
end
mind(u, S) = minimum(norm(u .- s) for s in S)

inpath = length(ARGS) >= 1 ? ARGS[1] : OUT_DIR*"torus_crisis/crisis_collision_input_23_10_2026.json"
outpath = length(ARGS) >= 2 ? ARGS[2] : OUT_DIR*"torus_crisis/crisis_collision_23_10_2026.json"
inp = JSON.parsefile(inpath)
r = inp["R"]; G = Float64.(inp["g2_MHz"])
NAMES = ["node", "saddle", "branch_I", "satellite"]
orb(k, ig) = inp["orbits"][k][ig]
states(k, ig) = [Float64.(s) for s in vcat(orb(k, ig)["states"], orb(k, ig)["partner"])]
node(ig) = states("node", ig)
gidx(g) = findfirst(x -> abs(x - g) < 1e-9, G)
EDGES = 10 .^ range(-4, 1, length = 51)                    # log bins of the distance
function binof(d)
    d <= EDGES[1] && return 1
    d >= EDGES[end] && return length(EDGES) - 1
    return searchsortedlast(EDGES, d)
end

# bubble's attractor at g0 = 100 MHz (same start as bubble_crisis_lifetime.jl)
crin = JSON.parsefile(OUT_DIR*"torus_crisis/bubble_crisis_input_22_10_2026.json")
it0 = integrator(Float64.(crin["seed_g0"]), pars(100.0, r))
for _ in 1:2000; next!(it0); end
ATT = Vector{Vector{Float64}}()
for k in 1:64
    for _ in 1:50; next!(it0); end
    push!(ATT, copy(it0.u))
end
println("attractor sample at 100 MHz: ", length(ATT), " points"); flush(stdout)

# ---------- attractor: distances to the candidates ----------
function run_attractor(ig, u0; N, n_keep = 4000)
    p = pars(G[ig], r); it = integrator(u0, p)
    S = Dict(k => states(k, ig) for k in NAMES)
    hist = Dict(k => zeros(Int, length(EDGES) - 1) for k in NAMES)
    dmin = Dict(k => Inf for k in NAMES); kmin = Dict(k => 0 for k in NAMES)
    keep = Vector{Vector{Float64}}(); land = nothing
    for _ in 1:200; next!(it); end                     # short transient at the new g2
    for n in 1:N
        u = next!(it)
        for k in NAMES
            d = mind(u, S[k]); hist[k][binof(d)] += 1
            d < dmin[k] && (dmin[k] = d; kmin[k] = n)
        end
        n % max(1, N ÷ n_keep) == 0 && push!(keep, copy(u))
        if mind(u, S["node"]) < 1e-3
            land = n; break
        end
    end
    return Dict("g2_MHz" => G[ig], "landed_Ta" => land, "hist" => hist, "dmin" => dmin, "n_at_dmin" => kmin,
                "points" => [round.(x, digits = 5) for x in keep])
end

# ---------- escape routes ----------
function run_escape(ig, u0; N_max, n_pre = 600)
    p = pars(G[ig], r); it = integrator(u0, p); S = node(ig)
    buf = Vector{Vector{Float64}}()
    for n in 1:N_max
        u = next!(it)
        push!(buf, copy(u)); length(buf) > n_pre + 5 && popfirst!(buf)
        if mind(u, S) < 1e-3
            return Dict("g2_MHz" => G[ig], "landed_Ta" => n, "pre" => buf)
        end
    end
    return Dict("g2_MHz" => G[ig], "landed_Ta" => nothing, "pre" => buf)
end

# ---------- unstable manifold of the omega_a/2 saddle ----------
function run_manifold(ig, sgn, δ; N_max, n_keep = 2000)
    o = orb("saddle", ig); y = Float64.(o["states"][1]); v = Float64.(o["v_lead"]); v ./= norm(v)
    p = pars(G[ig], r); it = integrator(y .+ sgn*δ .* v, p); S = node(ig)
    keep = Vector{Vector{Float64}}()
    for n in 1:N_max
        u = next!(it)
        if mind(u, S) < 1e-3
            return Dict("g2_MHz" => G[ig], "sign" => sgn, "delta" => δ, "landed_Ta" => n, "points" => keep)
        end
        n > 500 && length(keep) < n_keep && n % 10 == 0 && push!(keep, round.(copy(u), digits = 5))
    end
    return Dict("g2_MHz" => G[ig], "sign" => sgn, "delta" => δ, "landed_Ta" => nothing, "points" => keep)
end

# ---------- Neimark-Sacker of the node ----------
function run_ns_below(ig, seed; N)
    S = states("node", ig)[1:2]; u0 = S[1] .+ 1e-5 .* normalize(randn(MersenneTwister(seed), 4))
    it = integrator(u0, pars(G[ig], r)); d = Float64[]
    for n in 1:N
        u = next!(it)
        n % 10 == 0 && push!(d, mind(u, S))
        mind(u, S) > 1.0 && break
    end
    return Dict("g2_MHz" => G[ig], "seed" => seed, "dist_every_10Ta" => d)
end
function run_ns_kick(ig, ε, seed; N)
    S = states("node", ig)[1:2]; u0 = S[1] .+ ε .* normalize(randn(MersenneTwister(seed), 4))
    it = integrator(u0, pars(G[ig], r)); dmax = 0.0
    for n in 1:N
        u = next!(it); d = mind(u, states("node", ig))
        dmax = max(dmax, d)
        d < 1e-6 && return Dict("g2_MHz" => G[ig], "eps" => ε, "seed" => seed, "returned_Ta" => n, "max_excursion" => dmax)
        d > 0.3 && break                              # left the neighbourhood of the node
    end
    return Dict("g2_MHz" => G[ig], "eps" => ε, "seed" => seed, "returned_Ta" => nothing, "max_excursion" => dmax)
end

T = inp["tasks"]
jobs = Any[]
for g in T["attractor"]["g2_MHz"], k in 1:T["attractor"]["n_traj"]
    push!(jobs, ("attractor", gidx(g), k))
end
for g in T["escape"]["g2_MHz"], k in 1:T["escape"]["n_traj"]
    push!(jobs, ("escape", gidx(g), k))
end
for g in T["manifold"]["g2_MHz"], sgn in (-1, 1), δ in T["manifold"]["delta"]
    push!(jobs, ("manifold", gidx(g), (sgn, δ)))
end
for g in T["ns"]["g2_below"], k in 1:T["ns"]["n_below"]
    push!(jobs, ("ns_below", gidx(g), k))
end
for g in T["ns"]["g2_above"], ε in T["ns"]["eps"], k in 1:T["ns"]["n_kick"]
    push!(jobs, ("ns_kick", gidx(g), (ε, k)))
end
println(length(jobs), " jobs on ", nthreads(), " threads"); flush(stdout)
res = Vector{Any}(undef, length(jobs)); t0 = time()
@threads :dynamic for j in eachindex(jobs)
    kind, ig, a = jobs[j]
    att = ATT[mod1(isa(a, Int) ? a : 1, length(ATT))]
    res[j] = kind == "attractor" ? run_attractor(ig, att; N = T["attractor"]["N_Ta"]) :
             kind == "escape"    ? run_escape(ig, ATT[mod1(a + 17, length(ATT))]; N_max = T["escape"]["N_max_Ta"]) :
             kind == "manifold"  ? run_manifold(ig, a[1], a[2]; N_max = T["manifold"]["N_max_Ta"]) :
             kind == "ns_below"  ? run_ns_below(ig, a; N = T["ns"]["N_below_Ta"]) :
                                   run_ns_kick(ig, a[1], a[2]; N = T["ns"]["N_kick_Ta"])
    res[j] = merge(Dict{String, Any}(res[j]), Dict{String, Any}("task" => kind))
end
println("done in ", round(time() - t0, digits = 0), " s"); flush(stdout)
open(io -> JSON.print(io, Dict("R" => r, "edges" => collect(EDGES), "T_a_over_kappa_inv" => 2π/(w_a/κ0), "runs" => res)), outpath, "w")
