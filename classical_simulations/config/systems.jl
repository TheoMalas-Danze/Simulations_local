abstract type AbstractSystem end

struct SystemATS <: AbstractSystem
    ω_a::Float64
    ω_b::Float64
    E_J::Float64
    E_Ca::Float64
    E_Cb::Float64
    ϕ_a_0::Float64
    q_a_0::Float64
    ϕ_b_0::Float64
    q_b_0::Float64
end

struct SystemATS_prepared <: AbstractSystem
    system::SystemATS
    Z_a::Float64
    Z_b::Float64
    ϕ_quantum::Float64
end

struct SystemATS_ladder <: AbstractSystem
    ω_a::Float64
    ω_b::Float64
    E_J::Float64
    E_Ca::Float64
    E_Cb::Float64
    a_0::ComplexF64
    b_0::ComplexF64
end

struct SystemATS_ladder_prepared <: AbstractSystem
    system::SystemATS_ladder
    Z_a::Float64
    Z_b::Float64
    ϕ_quantum::Float64
end

struct SystemHO <: AbstractSystem
    ω_a::Float64
    E_Ca::Float64
    ϕ_a_0::Float64
    q_a_0::Float64
end

struct SystemHO_prepared <: AbstractSystem
    system::SystemHO
    Z_a::Float64
end


struct SystemMichiel <: AbstractSystem
    β::Float64
    λ::Float64
    x_0::Float64
    p_0::Float64
    ω_a::Int8
end

struct SystemMichiel_prepared <: AbstractSystem
    system::SystemMichiel
end

struct SystemATS_Michiel <: AbstractSystem
    β_a::Float64
    β_b::Float64
    λ_a::Float64
    λ_b::Float64
    x_0::Vector{Float64}
    p_0::Vector{Float64}
end

struct SystemATS_Michiel_prepared <: AbstractSystem
    system::SystemATS_Michiel
end

###########################################################

function prepare(system::AbstractSystem)
    throw(MethodError(prepare, (system,)))
end

function prepare(system::Union{SystemATS, SystemATS_ladder})
    ω_a, ω_b = system.ω_a, system.ω_b
    E_Ca, E_Cb = system.E_Ca, system.E_Cb
    C_a_inv=2*E_Ca # hbar/e^2 GHz
    C_b_inv=2*E_Cb # hbar/e^2 GHz
    Z_a=C_a_inv/ω_a # Impedance of the resonator a
    Z_b=C_b_inv/ω_b # Impedance of the resonator b

    ϕ_quantum = 1/2 # ħ/e 

        if system isa SystemATS
        return SystemATS_prepared(system, Z_a, Z_b, ϕ_quantum)
    elseif system isa SystemATS_ladder
        return SystemATS_ladder_prepared(system, Z_a, Z_b, ϕ_quantum)
    else
        error("Unsupported system type: $(typeof(system))")
    end
end

function prepare(system::SystemHO)
    ω_a = system.ω_a
    E_Ca = system.E_Ca
    C_a_inv=2*E_Ca # hbar/e^2 GHz
    Z_a=C_a_inv/ω_a # Impedance of the resonator a

    system_prepared = SystemHO_prepared(system, Z_a)
    return system_prepared
end

function prepare(system::SystemMichiel)
    return SystemMichiel_prepared(system)
end

function prepare(system::SystemATS_Michiel)
    return SystemATS_Michiel_prepared(system)
end