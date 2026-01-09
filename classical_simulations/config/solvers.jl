using DifferentialEquations, LinearAlgebra
include("/Users/tmalas/Desktop/Dev/Simulations_local/classical_simulations/config/equations.jl")

function default_solver(
    u_0_tot::Tuple{Vector{Float64}, Vector{Float64}},
    tspan::Tuple{Float64, Float64},
    p::Union{SimulationParams{SystemATS_prepared,D,MeanField},SimulationParams_Taylor{SystemATS_prepared,D,MeanField}},
    dt::Float64;
    equation::Function,
    t_save::Union{StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64},Nothing}=nothing
) where D<:AbstractDrive

    u_dot_0, u_0 = u_0_tot

    prob = SecondOrderODEProblem(equation,u_dot_0, u_0, tspan, p)
    t_save = isnothing(t_save) ? (tspan[1]:dt:tspan[2]) : t_save
    sol = solve(prob, VelocityVerlet(),dt=dt,
        saveat = t_save
        )

    ϕ_dot = [getindex.(sol.u,i) for i in 1:2]
    ϕ = [getindex.(sol.u,i+2) for i in 1:2]

    ϕ_a = ϕ[1] * sqrt(p.system_prepared.Z_a)
    q_a = ϕ_dot[1]/p.system_prepared.system.ω_a / sqrt(p.system_prepared.Z_a)

    ϕ_b = ϕ[2] * sqrt(p.system_prepared.Z_b)
    q_b = ϕ_dot[2]/p.system_prepared.system.ω_b / sqrt(p.system_prepared.Z_b)

    t_list = sol.t
    return ϕ_a,ϕ_b,q_a,q_b,t_list
end

function default_solver(
    u_0_tot::Tuple{Vector{Float64}, Vector{Float64}},
    tspan::Tuple{Float64, Float64},
    p::SimulationParams{SystemATS_prepared,D,OneChain_prepared},
    dt::Float64;
    equation::Function,
    t_min_save::Union{Float64,Nothing}=nothing
) where D<:AbstractDrive

    u_dot_0, u_0 = u_0_tot

    prob = SecondOrderODEProblem(equation,u_dot_0, u_0, tspan, p)
    save_range = isnothing(t_min_save) ? (dt:dt:tspan[2]) : (t_min_save:dt:tspan[2])
    save_idxs=[1,2,p.environment_prepared.n+3,p.environment_prepared.n+4] # We only save the system DoFs (ϕ_a, ϕ_b) and not the chain DoFs (Φ)
    sol = solve(prob, VelocityVerlet(),dt=dt,
        saveat=save_range,                      # how often to save (or choose a coarser step)
        save_idxs=save_idxs,            # only store system DoFs
        save_everystep=false            # do not store intermediate solver steps
        )

    ϕ_dot = [getindex.(sol.u,i) for i in 1:2]
    ϕ = [getindex.(sol.u,i+2) for i in 1:2]

    ϕ_a = ϕ[1] * sqrt(p.system_prepared.Z_a)
    q_a = ϕ_dot[1]/p.system_prepared.system.ω_a / sqrt(p.system_prepared.Z_a)

    ϕ_b = ϕ[2] * sqrt(p.system_prepared.Z_b)
    q_b = ϕ_dot[2]/p.system_prepared.system.ω_b / sqrt(p.system_prepared.Z_b)

    t_list = sol.t
    return ϕ_a,ϕ_b,q_a,q_b,t_list
end


function default_solver(
    u_0_tot::Tuple{Vector{Float64}, Vector{Float64}},
    tspan::Tuple{Float64, Float64},
    p::SimulationParams{SystemHO_prepared,D,MeanField_HO},
    dt::Float64;
    equation::Function,
    t_save::Union{StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64},Nothing}=nothing
) where D<:AbstractDrive

    u_dot_0, u_0 = u_0_tot

    prob = SecondOrderODEProblem(equation,u_dot_0, u_0, tspan, p)
    t_save = isnothing(t_save) ? (tspan[1]:dt:tspan[2]) : t_save
    sol = solve(prob, VelocityVerlet(),dt=dt,
        saveat = t_save
        )

    ϕ_dot = getindex.(sol.u,1) 
    ϕ = getindex.(sol.u,2)

    ϕ_a = ϕ * sqrt(p.system_prepared.Z_a)
    q_a = ϕ_dot/p.system_prepared.system.ω_a / sqrt(p.system_prepared.Z_a)

    t_list = sol.t
    return ϕ_a,q_a,t_list
end


function default_solver(
    u_0::Vector{ComplexF64},
    tspan::Tuple{Float64, Float64},
    p::Union{SimulationParams{SystemATS_ladder_prepared,D,MeanField},SimulationParams_Taylor{SystemATS_ladder_prepared,D,MeanField}},
    dt::Float64;
    equation::Function,
    t_save::Union{StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64},Nothing}=nothing
) where D<:AbstractDrive

    prob = ODEProblem(equation, u_0, tspan, p)
    t_save = isnothing(t_save) ? (tspan[1]:dt:tspan[2]) : t_save
    sol = solve(prob, Tsit5(),dt=dt,
       saveat = t_save
        )
    #sol = solve(prob, Tsit5(); reltol=1e-8, abstol=1e-10,
    #    saveat = t_save
    #    )

    a = getindex.(sol.u,1)
    b = getindex.(sol.u,2)

    t_list = sol.t
    return a, b, t_list
end



function emission(
    u_0_tot::Tuple{Vector{Float64}, Vector{Float64}},
    tspan::Tuple{Float64, Float64},
    p::SimulationParams{S,D,OneChain_prepared},
    dt::Float64;
    equation::Function
) where {S<:AbstractSystem, D<:AbstractDrive}

    u_dot_0, u_0 = u_0_tot

    t_max = tspan[2]
    n = p.environment_prepared.n
    prob = SecondOrderODEProblem(equation,u_dot_0, u_0, tspan, p)
    save_idxs = vcat(3:n+2,n+5:2*n+4) # We save the chain DoFs (Φ)

    sol = solve(prob, VelocityVerlet(),dt=dt,
    saveat = t_max,                      # how often to save (or choose a coarser step)
    save_idxs = save_idxs,            # only store system DoFs
    save_everystep = false            # do not store intermediate solver steps
    )

    ϕ_dot = [getindex.(sol.u,i)[end] for i in 1:n]
    ϕ = [getindex.(sol.u,i+n)[end] for i in 1:n]
    return ϕ,ϕ_dot,sol.t,p.environment_prepared.Mq_chain, p.environment_prepared.Mϕ_chain
end

function default_solver(
    u_0_tot::Tuple{Vector{Float64}, Vector{Float64}},
    tspan::Tuple{Float64, Float64},
    p::SimulationParams{SystemMichiel_prepared,D,MeanField_Michiel},
    dt::Float64;
    equation::Function,
    t_save::Union{StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64},Nothing}=nothing
) where D<:AbstractDrive

    u_dot_0, u_0 = u_0_tot

    prob = SecondOrderODEProblem(equation, u_dot_0, u_0, tspan, p)
    t_save = isnothing(t_save) ? (tspan[1]:dt:tspan[2]) : t_save
    sol = solve(prob, VelocityVerlet(),dt=dt,
        saveat = t_save
        )

    p = getindex.(sol.u,1) 
    x = getindex.(sol.u,2)

    t_list = sol.t
    return x,p,t_list
end

function default_solver(
    u_0_tot::Tuple{Vector{Float64}, Vector{Float64}},
    tspan::Tuple{Float64, Float64},
    p::SimulationParams{SystemATS_Michiel_prepared,D,MeanField},
    dt::Float64;
    equation::Function,
    t_save::Union{StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64},Nothing}=nothing
) where D<:AbstractDrive

    u_dot_0, u_0 = u_0_tot

    prob = SecondOrderODEProblem(equation, u_dot_0, u_0, tspan, p)
    t_save = isnothing(t_save) ? (tspan[1]:dt:tspan[2]) : t_save
    sol = solve(prob, VelocityVerlet(),dt=dt,
        saveat = t_save
        )
    
    p = [getindex.(sol.u,i) for i in 1:2]
    x = [getindex.(sol.u,i+2) for i in 1:2]

    x_a = x[1]
    p_a = p[1] /(1/2)

    x_b = x[2] 
    p_b = p[2]

    t_list = sol.t
    return x_a, x_b, p_a, p_b, t_list
end