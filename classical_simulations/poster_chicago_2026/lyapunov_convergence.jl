# Run: julia --project=Code_Julia -t 10 lyapunov_convergence.jl base,T2,T4,T8,Ttr4,d0,tol   (from Simulations_local)
# Times in the code convention (κ_b = π/10.4, −2κ_b φ̇_b): T = 30/κ_b here is 60/κ_b in the article.
# Convergence of the vacuum-start Lyapunov map (lyapunov_heatmap.ipynb settings) in T, d0 and solver tolerance.
using ChaosTools, JSON, Base.Threads
include("/Users/tmalas/Desktop/Dev/Simulations_local/classical_simulations/config/baths.jl")
include("/Users/tmalas/Desktop/Dev/Simulations_local/classical_simulations/simulator.jl")

ϕ_quantum=1/2
ω_a = 25.338776456203686; ω_b = 2ω_a
φ_a = 0.11; Z_a = 2ϕ_quantum^2*φ_a^2; E_Ca = Z_a*ω_a/2
φ_b = 0.204; Z_b = 2ϕ_quantum^2*φ_b^2; E_Cb = Z_b*ω_b/2
E_J = 37.12*2π*1.5
ω_d = ω_b; ω_p = 2ω_a-ω_d
κ_b = π/10.4; κ_a = 0.0   # code convention (−2κ_b φ̇_b)

system = SystemATS(ω_a, ω_b, E_J, E_Ca, E_Cb, 0.0, 0.0, 0.0, 0.0)
environment = MeanField(κ_a, κ_b)
sp = prepare(system); ep = prepare(system, environment)
dt = get_dt(system, environment)
u_dot_0, u_0 = get_init(sp, ep); u0 = vcat(u_0, u_dot_0)

function lyap_point(ϵ_p, ratio; T=30/κ_b, Ttr=15/κ_b, d0=1e-7, tol=1e-6)
    g2 = sin(ϵ_p)*E_J*φ_a^2*φ_b
    drive = DriveTwoModesBasics(ratio*g2, ω_d, 0.0, ϵ_p, ω_p, 0.0)
    p = SimulationParams(sp, prepare(drive), ep)
    ds = CoupledODEs(equation_1ODE!, copy(u0), p; diffeq=(alg=Tsit5(), abstol=tol, reltol=tol))
    lyapunov(ds, T; Δt=dt, d0=d0, d0_upper=d0*1e3, d0_lower=d0*1e-3, Ttr=Ttr)
end

mode = ARGS[1]
if mode == "time"
    t = @elapsed lyap_point(0.8, 8.0); println("first ", t)
    t = @elapsed λ = lyap_point(0.8, 8.0); println("second ", t, " λ=", λ)
    t = @elapsed λ = lyap_point(0.8, 8.0; tol=1e-10); println("tol1e-10 ", t, " λ=", λ)
    println("threads ", nthreads())
else
    ϵ_p_list = collect(0.2:0.01:1.1)
    ratios = collect(2:0.1:13.5)
    variants = Dict(
        "base"   => (;),
        "T2"     => (T=60/κ_b,),
        "T4"     => (T=120/κ_b,),
        "Ttr4"   => (Ttr=60/κ_b,),
        "d0"     => (d0=1e-5,),
        "tol"    => (tol=1e-10,),
        "T8"     => (T=240/κ_b,),
    )
    vnames = mode == "all" ? collect(keys(variants)) : split(mode, ",")
    for name in vnames
        kw = variants[name]
        jobs = [(i,j) for i in eachindex(ϵ_p_list) for j in eachindex(ratios)]
        out = fill(NaN, length(ϵ_p_list), length(ratios))
        t = @elapsed @threads for k in eachindex(jobs)
            i, j = jobs[k]
            out[i,j] = lyap_point(ϵ_p_list[i], ratios[j]; kw...)
        end
        println(name, " done in ", round(t, digits=1), " s"); flush(stdout)
        open("/Users/tmalas/Desktop/Dev/Simulations_local/classical_simulations/analysis/output/chicago_poster_2026/lyapunov_convergence_$(name)_13_10_2026.json", "w") do io
            JSON.print(io, Dict("eps_p"=>ϵ_p_list, "ratio"=>ratios, "lambda"=>[out[:,j] for j in eachindex(ratios)]))
        end
    end
end
