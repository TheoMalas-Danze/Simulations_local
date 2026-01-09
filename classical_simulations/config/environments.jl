include("/Users/tmalas/Desktop/Dev/Simulations_local/classical_simulations/config/systems.jl")
using LinearAlgebra, StaticArrays

abstract type AbstractEnv end

abstract type AbstractMeanField <: AbstractEnv end

struct MeanField <: AbstractMeanField
    κ_a::Float64
    κ_b::Float64
end

struct MeanField_HO <: AbstractMeanField
    κ_a::Float64
end

struct MeanField_Michiel <: AbstractMeanField
    Q::Float64
end

struct OneChain <: AbstractEnv
    n::Int64
    onsite_energies::Vector{Float64}
    hopping_energies::Vector{Float64}
    system_chain_coupling::Float64
end

struct OneChain_prepared <: AbstractEnv
    n::Int64
    system_chain_coupling::Float64
    Mϕ_chain::SymTridiagonal{Float64, Vector{Float64}}   # matrix for the chain Hamiltonian for the charge variable
    Mq_chain::SymTridiagonal{Float64, Vector{Float64}}  # matrix for the chain Hamiltonian for the flux variable
    Mq_full::SymTridiagonal{Float64, Vector{Float64}}
    temp_buf::Vector{Float64}
end

### prepare ####

function prepare(system::AbstractSystem, environment::AbstractEnv,)
    throw(MethodError(prepare, (system, environment,)))
end

function prepare(system::AbstractSystem, environment::E) where E<:AbstractMeanField
    return environment
end

function prepare(system::Union{SystemATS, SystemATS_ladder}, environment::OneChain)
    println("Preparing OneChain environment…")
    onsite_energies, hopping_energies = environment.onsite_energies, environment.hopping_energies
    n = environment.n
    temp_buf = zeros(n)

    gq=vcat(- hopping_energies)
    gϕ=vcat(hopping_energies)

    Mϕ_chain=SymTridiagonal(onsite_energies, gϕ)

    Mq_full=SymTridiagonal(vcat([system.ω_a,system.ω_b], onsite_energies), vcat([0,0],gq))
    Mq_chain=SymTridiagonal(onsite_energies, gϕ)

    environment_prepared = OneChain_prepared(n, environment.system_chain_coupling, Mϕ_chain, Mq_chain, Mq_full, temp_buf)
    return environment_prepared
end
### get_dt ####

function get_dt(system::AbstractSystem, environment::AbstractEnv)
    throw(MethodError(get_dt, (system,environment,)))
end

function get_dt(system::Union{SystemATS, SystemATS_ladder, SystemHO, SystemMichiel},  environment::E) where E<:AbstractMeanField
    dt = 1 / (34*sqrt(2) * system.ω_a) # Time step for the simulation
    return dt
end

function get_dt(system::Union{SystemATS, SystemATS_ladder}, environment::OneChain)
    dt = 1 / (3.4*sqrt(2) * maximum(vcat(system.ω_a,system.ω_b,environment.onsite_energies))) # Time step for the simulation
    return dt
end

function get_dt(system::Union{SystemATS_Michiel},  environment::E) where E<:AbstractMeanField
    dt = 2pi/(50*(1/2)) # Time step for the simulation
    return dt
end

### get_init ###

function get_init(system::AbstractSystem, environment::AbstractEnv)
    throw(MethodError(get_init, (system,environment,)))
end

function get_init(system_prepared::SystemATS_prepared, environment::MeanField)
    ϕ_a_0, q_a_0 = system_prepared.system.ϕ_a_0, system_prepared.system.q_a_0
    ϕ_b_0, q_b_0 = system_prepared.system.ϕ_b_0, system_prepared.system.q_b_0
    ω_a, ω_b = system_prepared.system.ω_a, system_prepared.system.ω_b
    Z_a, Z_b = system_prepared.Z_a, system_prepared.Z_b
    
    ϕ_0 = vcat(ϕ_a_0/sqrt(Z_a),ϕ_b_0/sqrt(Z_b))
    ϕ_dot_0 = vcat(q_a_0 * sqrt(Z_a)*ω_a,q_b_0 * sqrt(Z_b)*ω_b)
    return ϕ_dot_0, ϕ_0
end

function get_init(system_prepared::SystemATS_ladder_prepared, environment::MeanField)
    a_0 = system_prepared.system.a_0
    b_0 = system_prepared.system.b_0
    
    u_0 = vcat(a_0, b_0)
    return u_0
end

function get_init(system_prepared::SystemATS_prepared, environment_prepared::OneChain_prepared)
    ϕ_a_0, q_a_0 = system_prepared.system.ϕ_a_0, system_prepared.system.q_a_0
    ϕ_b_0, q_b_0 = system_prepared.system.ϕ_b_0, system_prepared.system.q_b_0
    Z_a, Z_b = system_prepared.Z_a, system_prepared.Z_b

    n, Mq_full = environment_prepared.n, environment_prepared.Mq_full
    
    ϕ_0 = vcat(ϕ_a_0/sqrt(Z_a),ϕ_b_0/sqrt(Z_b), fill(0.0, n))
    q_0 = vcat(q_a_0 * sqrt(Z_a),q_b_0 * sqrt(Z_b), fill(0.0, n))
    ϕ_dot_0 = Mq_full*q_0

    return ϕ_dot_0, ϕ_0
end

function get_init(system_prepared::SystemATS_ladder_prepared, environment_prepared::OneChain_prepared)
    a_0 = system_prepared.system.a_0
    b_0 = system_prepared.system.b_0

    n = environment_prepared.n
    
    u_0 = vcat(a_0, b_0, fill(0.0, n))

    return u_0
end

function get_init(system_prepared::SystemHO_prepared, environment::MeanField_HO)
    ϕ_a_0, q_a_0 = system_prepared.system.ϕ_a_0, system_prepared.system.q_a_0
    ω_a = system_prepared.system.ω_a
    Z_a = system_prepared.Z_a
    
    ϕ_0 = vcat(ϕ_a_0/sqrt(Z_a))
    ϕ_dot_0 = vcat(q_a_0 * sqrt(Z_a)*ω_a)
    return ϕ_dot_0, ϕ_0
end

function get_init(system_prepared::SystemMichiel_prepared, environment::MeanField_Michiel)
    x_0, p_0 = system_prepared.system.x_0, system_prepared.system.p_0
    return ([p_0], [x_0])
end

function get_init(system_prepared::SystemATS_Michiel_prepared, environment::MeanField)
    x_0, p_0 = system_prepared.system.x_0, system_prepared.system.p_0
    return (p_0, x_0)
end