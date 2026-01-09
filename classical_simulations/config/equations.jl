include("/Users/tmalas/Desktop/Dev/Simulations_local/classical_simulations/config/systems.jl")
include("/Users/tmalas/Desktop/Dev/Simulations_local/classical_simulations/config/drives.jl")
include("/Users/tmalas/Desktop/Dev/Simulations_local/classical_simulations/config/environments.jl")
include("/Users/tmalas/Desktop/Dev/Simulations_local/classical_simulations/utils/taylor_expansions.jl")


abstract type AbstractSimulationParams end

struct SimulationParams{S<:AbstractSystem, D<:AbstractDrive, E<:AbstractEnv} <: AbstractSimulationParams
    system_prepared::S
    drive_prepared::D
    environment_prepared::E
end

struct SimulationParams_Taylor{S<:AbstractSystem, D<:AbstractDrive, E<:AbstractEnv} <: AbstractSimulationParams
    Taylor_order::Int
    system_prepared::S
    drive_prepared::D
    environment_prepared::E
end

#ATS + Mean_Field
function default_equation!(ddu, du, u, p::SimulationParams{SystemATS_prepared, D, MeanField}, t) where D<:DriveTwoModes_prepared
    ω_a, ω_b, E_J, Z_a, Z_b, ϕ_quantum = p.system_prepared.system.ω_a, p.system_prepared.system.ω_b, p.system_prepared.system.E_J, p.system_prepared.Z_a, p.system_prepared.Z_b, p.system_prepared.ϕ_quantum
    ϵ_d, ω_d, buffer_phase, ϵ_0, ω_p, pump_phase =  p.drive_prepared.drive.ϵ_d, p.drive_prepared.drive.ω_d, p.drive_prepared.drive.buffer_phase, p.drive_prepared.drive.ϵ_0, p.drive_prepared.drive.ω_p, p.drive_prepared.drive.pump_phase
    ϕ_a = u[1]
    ϕ_b = u[2]
    coupling = (cos(ϕ_a * sqrt(Z_a)/ϕ_quantum + ϕ_b * sqrt(Z_b)/ϕ_quantum) -1) / ϕ_quantum
    sin_pump = 2*E_J*p.drive_prepared.drive_pump(ϵ_0, ω_p, t, pump_phase)
    ddu[1] = -ω_a*(ω_a*ϕ_a - sqrt(Z_a) * sin_pump*coupling) - 2 * p.environment_prepared.κ_a * du[1]
    ddu[2] = -ω_b*(ω_b*ϕ_b + p.drive_prepared.drive_buffer(ϵ_d, ω_d, t, buffer_phase)*sqrt(2) - sqrt(Z_b) * sin_pump*coupling) - 2 * p.environment_prepared.κ_b * du[2]
    return(nothing)
end

#ATS + 1_Chain
function default_equation!(ddu, du, u, p::SimulationParams{SystemATS_prepared, D, OneChain_prepared}, t) where D<:DriveTwoModes_prepared
    ω_a, ω_b, E_J, Z_a, Z_b, ϕ_quantum = p.system_prepared.system.ω_a, p.system_prepared.system.ω_b, p.system_prepared.system.E_J, p.system_prepared.Z_a, p.system_prepared.Z_b, p.system_prepared.ϕ_quantum
    ϵ_d, ω_d, buffer_phase, ϵ_0, ω_p, pump_phase =  p.drive_prepared.drive.ϵ_d, p.drive_prepared.drive.ω_d, p.drive_prepared.drive.buffer_phase, p.drive_prepared.drive.ϵ_0, p.drive_prepared.drive.ω_p, p.drive_prepared.drive.pump_phase

    ϕ_a = u[1]
    ϕ_b = u[2]
    Φ=u[3:end]

    coupling = (cos(ϕ_a * sqrt(Z_a)/ϕ_quantum + ϕ_b * sqrt(Z_b)/ϕ_quantum) - 1) / ϕ_quantum
    sin_pump = 2*E_J*p.drive_prepared.drive_pump(ϵ_0, ω_p, t, pump_phase)

    ddu[1] = -ω_a*(ω_a*ϕ_a - sqrt(Z_a) * sin_pump*coupling)
    ddu[2] = -ω_b*(ω_b*ϕ_b + p.drive_prepared.drive_buffer(ϵ_d, ω_d, t, buffer_phase)*sqrt(2) - sqrt(Z_b) * sin_pump*coupling) + ω_b * p.environment_prepared.system_chain_coupling * sqrt(Z_b) * Φ[1]
    mul!(p.environment_prepared.temp_buf, p.environment_prepared.Mϕ_chain, Φ)
    @. p.environment_prepared.temp_buf = -p.environment_prepared.temp_buf
    p.environment_prepared.temp_buf[1] += p.environment_prepared.system_chain_coupling * sqrt(Z_b) * ϕ_b
    mul!(view(ddu, 3:length(ddu)), p.environment_prepared.Mq_chain, p.environment_prepared.temp_buf)
    return nothing
end

#ATS + Harmonic oscillator
function default_equation!(ddu, du, u, p::SimulationParams{SystemHO_prepared, D, MeanField_HO}, t) where D<:DriveTwoModes_prepared
    ddu[1] = -p.system_prepared.system.ω_a^2*sin(u[1]) - 2 * p.environment_prepared.κ_a * du[1]
    return(nothing)
end

#ATS + Mean_Field
function equation_1ODE!(du,u,p::SimulationParams{SystemATS_prepared, D, MeanField}, t) where D<:DriveTwoModes_prepared
    ω_a, ω_b, E_J, Z_a, Z_b, ϕ_quantum = p.system_prepared.system.ω_a, p.system_prepared.system.ω_b, p.system_prepared.system.E_J, p.system_prepared.Z_a, p.system_prepared.Z_b, p.system_prepared.ϕ_quantum
    ϵ_d, ω_d, buffer_phase, ϵ_0, ω_p, pump_phase =  p.drive_prepared.drive.ϵ_d, p.drive_prepared.drive.ω_d, p.drive_prepared.drive.buffer_phase, p.drive_prepared.drive.ϵ_0, p.drive_prepared.drive.ω_p, p.drive_prepared.drive.pump_phase

    ϕ_a = u[1]
    ϕ_b = u[2]
    ϕ_p_a = u[3]
    ϕ_p_b = u[4]

    coupling = (cos(ϕ_a * sqrt(Z_a)/ϕ_quantum + ϕ_b * sqrt(Z_b)/ϕ_quantum) - 1) / ϕ_quantum
    sin_pump = 2*E_J*p.drive_prepared.drive_pump(ϵ_0, ω_p, t, pump_phase)
        
    du[1] = ϕ_p_a
    du[2] = ϕ_p_b
    du[3] = -ω_a*(ω_a*ϕ_a - sqrt(Z_a) * sin_pump*coupling) - 2 * p.environment_prepared.κ_a * du[1]
    du[4] = -ω_b*(ω_b*ϕ_b + p.drive_prepared.drive_buffer(ϵ_d, ω_d, t, buffer_phase)*sqrt(2) - sqrt(Z_b) * sin_pump*coupling) - 2 * p.environment_prepared.κ_b * du[2]
    return nothing
end

#ATS + Mean_Field + ladder_operator
function default_equation!(du,u,p::SimulationParams{SystemATS_ladder_prepared, D, MeanField}, t) where D<:DriveTwoModes_prepared
    ω_a, ω_b, E_J, Z_a, Z_b, ϕ_quantum = p.system_prepared.system.ω_a, p.system_prepared.system.ω_b, p.system_prepared.system.E_J, p.system_prepared.Z_a, p.system_prepared.Z_b, p.system_prepared.ϕ_quantum
    ϵ_d, ω_d, buffer_phase, ϵ_0, ω_p, pump_phase =  p.drive_prepared.drive.ϵ_d, p.drive_prepared.drive.ω_d, p.drive_prepared.drive.buffer_phase, p.drive_prepared.drive.ϵ_0, p.drive_prepared.drive.ω_p, p.drive_prepared.drive.pump_phase

    a = u[1]
    b = u[2]

    coupling = (cos((a+conj(a))/sqrt(2) * sqrt(Z_a)/ϕ_quantum + (b+conj(b))/sqrt(2) * sqrt(Z_b)/ϕ_quantum) - 1) / ϕ_quantum / sqrt(2)
    sin_pump = 2*E_J*p.drive_prepared.drive_pump(ϵ_0, ω_p, t, pump_phase)
        
    du[1] = -1im*(ω_a*a - sqrt(Z_a)*sin_pump*coupling) - p.environment_prepared.κ_a * a
    du[2] = -1im*(ω_b*b + p.drive_prepared.drive_buffer(ϵ_d, ω_d, t, buffer_phase) - sqrt(Z_b)*sin_pump*coupling) - p.environment_prepared.κ_b * b
    return nothing
end


#ATS + Mean_Field
function equation_1ODE_Taylor!(du,u,p::SimulationParams_Taylor{SystemATS_prepared, D, MeanField}, t) where D<:DriveTwoModes_prepared
    ω_a, ω_b, E_J, Z_a, Z_b, ϕ_quantum = p.system_prepared.system.ω_a, p.system_prepared.system.ω_b, p.system_prepared.system.E_J, p.system_prepared.Z_a, p.system_prepared.Z_b, p.system_prepared.ϕ_quantum
    ϵ_d, ω_d, buffer_phase, ϵ_0, ω_p, pump_phase =  p.drive_prepared.drive.ϵ_d, p.drive_prepared.drive.ω_d, p.drive_prepared.drive.buffer_phase, p.drive_prepared.drive.ϵ_0, p.drive_prepared.drive.ω_p, p.drive_prepared.drive.pump_phase

    ϕ_a = u[1]
    ϕ_b = u[2]
    ϕ_p_a = u[3]
    ϕ_p_b = u[4]

    coupling = (Taylor_cos(ϕ_a * sqrt(Z_a)/ϕ_quantum + ϕ_b * sqrt(Z_b)/ϕ_quantum, p.Taylor_order) - 1)/ϕ_quantum
    sin_pump = 2*E_J*sin(p.drive_prepared.drive_pump(ϵ_0, ω_p, t, pump_phase))
        
    du[1] = ϕ_p_a
    du[2] = ϕ_p_b
    du[3] = -ω_a*(ω_a*ϕ_a - sqrt(Z_a) * sin_pump*coupling) - 2 * p.environment_prepared.κ_a * du[1]
    du[4] = -ω_b*(ω_b*ϕ_b + p.drive_prepared.drive_buffer(ϵ_d, ω_d, t, buffer_phase)*sqrt(2) - sqrt(Z_b) * sin_pump*coupling) - 2 * p.environment_prepared.κ_b * du[2]
    return nothing
end

#ATS + 1_chain
function equation_1ODE!(du,u,p::SimulationParams{SystemATS_prepared, D, OneChain_prepared}, t) where D<:DriveTwoModes_prepared
    ω_a, ω_b, E_J, Z_a, Z_b, ϕ_quantum = p.system_prepared.system.ω_a, p.system_prepared.system.ω_b, p.system_prepared.system.E_J, p.system_prepared.Z_a, p.system_prepared.Z_b, p.system_prepared.ϕ_quantum
    ϵ_d, ω_d, buffer_phase, ϵ_0, ω_p, pump_phase =  p.drive_prepared.drive.ϵ_d, p.drive_prepared.drive.ω_d, p.drive_prepared.drive.buffer_phase, p.drive_prepared.drive.ϵ_0, p.drive_prepared.drive.ω_p, p.drive_prepared.drive.pump_phase
    n=p.environment_prepared.n
    ϕ_a = u[1]
    ϕ_b = u[2]
    Φ=u[3:n+2]
    ϕ_d_a = u[3+n]
    ϕ_d_b = u[4+n]
    Φ_d=u[5+n:end]
    
    du[1] = ϕ_d_a
    du[2] = ϕ_d_b
    du[3:n+2] = Φ_d
    
    coupling = (cos(ϕ_a * sqrt(Z_a)/ϕ_quantum + ϕ_b * sqrt(Z_b)/ϕ_quantum) - 1) / ϕ_quantum
    sin_pump = 2*E_J*sin(p.drive_prepared.drive_pump(ϵ_0, ω_p, t, pump_phase))

    du[3+n] = -ω_a*(ω_a*ϕ_a - sqrt(Z_a) * sin_pump*coupling)
    du[4+n] = -ω_b*(ω_b*ϕ_b + p.drive_prepared.drive_buffer(ϵ_d, ω_d, t, buffer_phase)*sqrt(2) - sqrt(Z_b) * sin_pump*coupling) + ω_b * p.environment_prepared.system_chain_coupling * sqrt(Z_b) * Φ[1]
    mul!(p.environment_prepared.temp_buf, p.environment_prepared.Mϕ_chain, Φ)
    @. p.environment_prepared.temp_buf = -p.environment_prepared.temp_buf
    p.environment_prepared.temp_buf[1] += p.environment_prepared.system_chain_coupling * sqrt(Z_b) * ϕ_b
    mul!(view(du, 5+n:length(du)), p.environment_prepared.Mq_chain, p.environment_prepared.temp_buf)
    return nothing
end

#ATS + 1_chain
function equation_1ODE_Taylor!(du,u,p::SimulationParams_Taylor{SystemATS_prepared, D, OneChain_prepared}, t) where D<:DriveTwoModes_prepared
    ω_a, ω_b, E_J, Z_a, Z_b, ϕ_quantum = p.system_prepared.system.ω_a, p.system_prepared.system.ω_b, p.system_prepared.system.E_J, p.system_prepared.Z_a, p.system_prepared.Z_b, p.system_prepared.ϕ_quantum
    ϵ_d, ω_d, buffer_phase, ϵ_0, ω_p, pump_phase =  p.drive_prepared.drive.ϵ_d, p.drive_prepared.drive.ω_d, p.drive_prepared.drive.buffer_phase, p.drive_prepared.drive.ϵ_0, p.drive_prepared.drive.ω_p, p.drive_prepared.drive.pump_phase

    ϕ_a = u[1]
    ϕ_b = u[2]
    Φ=u[3:n+2]
    ϕ_d_a = u[3+n]
    ϕ_d_b = u[4+n]
    Φ_d=u[5+n:end]

    du[1] = ϕ_d_a
    du[2] = ϕ_d_b
    du[3:n+2] = Φ_d

    coupling = (Taylor_cos(ϕ_a * sqrt(Z_a)/ϕ_quantum + ϕ_b * sqrt(Z_b)/ϕ_quantum, p.Taylor_order) - 1)/ϕ_quantum
    sin_pump = 2*E_J*sin(p.drive_prepared.drive_pump(ϵ_0, ω_p, t, pump_phase))
        

    du[3] = -ω_a*(ω_a*ϕ_a - sqrt(Z_a) * sin_pump*coupling)
    du[4] = -ω_b*(ω_b*ϕ_b + p.drive_prepared.drive_buffer(ϵ_d, ω_d, t, buffer_phase)*sqrt(2) - sqrt(Z_b) * sin_pump*coupling)  + ω_b * p.environment_prepared.system_chain_coupling * sqrt(Z_b) * Φ[1]
    mul!(p.environment_prepared.temp_buf, p.environment_prepared.Mϕ_chain, Φ)
    @. p.environment_prepared.temp_buf = -p.environment_prepared.temp_buf
    p.environment_prepared.temp_buf[1] += p.environment_prepared.system_chain_coupling * sqrt(Z_b) * ϕ_b
    mul!(view(du, 5+n:length(du)), p.environment_prepared.Mq_chain, p.environment_prepared.temp_buf)
    return nothing
end


#ATS + Mean_Field
function equation_O3!(ddu, du, u, p::SimulationParams{SystemATS_prepared, D, MeanField}, t) where D<:DriveTwoModes_prepared
    ω_a, ω_b, E_J, Z_a, Z_b, ϕ_quantum = p.system_prepared.system.ω_a, p.system_prepared.system.ω_b, p.system_prepared.system.E_J, p.system_prepared.Z_a, p.system_prepared.Z_b, p.system_prepared.ϕ_quantum
    ϵ_d, ω_d, buffer_phase, ϵ_0, ω_p, pump_phase =  p.drive_prepared.drive.ϵ_d, p.drive_prepared.drive.ω_d, p.drive_prepared.drive.buffer_phase, p.drive_prepared.drive.ϵ_0, p.drive_prepared.drive.ω_p, p.drive_prepared.drive.pump_phase
    ϕ_a = u[1]
    ϕ_b = u[2]
    coupling = - (ϕ_a * sqrt(Z_a)/ϕ_quantum + ϕ_b * sqrt(Z_b)/ϕ_quantum)^2 / ϕ_quantum / 2
    sin_pump = 2*E_J*sin(p.drive_prepared.drive_pump(ϵ_0, ω_p, t, pump_phase))
    ddu[1] = -ω_a*(ω_a*ϕ_a - sqrt(Z_a) * sin_pump*coupling) - 2 * p.environment_prepared.κ_a * du[1]
    ddu[2] = -ω_b*(ω_b*ϕ_b + p.drive_prepared.drive_buffer(ϵ_d, ω_d, t, buffer_phase)*sqrt(2) - sqrt(Z_b) * sin_pump*coupling) - 2 * p.environment_prepared.κ_b * du[2]
    return(nothing)
end

#ATS + Mean_Field
function equation_Taylor!(ddu, du, u, p::SimulationParams_Taylor{SystemATS_prepared, D, MeanField}, t) where D<:DriveTwoModes_prepared
    ω_a, ω_b, E_J, Z_a, Z_b, ϕ_quantum = p.system_prepared.system.ω_a, p.system_prepared.system.ω_b, p.system_prepared.system.E_J, p.system_prepared.Z_a, p.system_prepared.Z_b, p.system_prepared.ϕ_quantum
    ϵ_d, ω_d, buffer_phase, ϵ_0, ω_p, pump_phase =  p.drive_prepared.drive.ϵ_d, p.drive_prepared.drive.ω_d, p.drive_prepared.drive.buffer_phase, p.drive_prepared.drive.ϵ_0, p.drive_prepared.drive.ω_p, p.drive_prepared.drive.pump_phase
    ϕ_a = u[1]
    ϕ_b = u[2]
    coupling = (Taylor_cos(ϕ_a * sqrt(Z_a)/ϕ_quantum + ϕ_b * sqrt(Z_b)/ϕ_quantum, p.Taylor_order) - 1)/ϕ_quantum
    sin_pump = 2*E_J*sin(p.drive_prepared.drive_pump(ϵ_0, ω_p, t, pump_phase))
    ddu[1] = -ω_a*(ω_a*ϕ_a - sqrt(Z_a) * sin_pump*coupling) - 2 * p.environment_prepared.κ_a * du[1]
    ddu[2] = -ω_b*(ω_b*ϕ_b + p.drive_prepared.drive_buffer(ϵ_d, ω_d, t, buffer_phase)*sqrt(2) - sqrt(Z_b) * sin_pump*coupling) - 2 * p.environment_prepared.κ_b * du[2]
    return(nothing)
end

#ATS + Mean_Field + ladder_operator
function default_equation!(du,u,p::SimulationParams_Taylor{SystemATS_ladder_prepared, D, MeanField}, t) where D<:DriveTwoModes_prepared
    ω_a, ω_b, E_J, Z_a, Z_b, ϕ_quantum = p.system_prepared.system.ω_a, p.system_prepared.system.ω_b, p.system_prepared.system.E_J, p.system_prepared.Z_a, p.system_prepared.Z_b, p.system_prepared.ϕ_quantum
    ϵ_d, ω_d, buffer_phase, ϵ_0, ω_p, pump_phase =  p.drive_prepared.drive.ϵ_d, p.drive_prepared.drive.ω_d, p.drive_prepared.drive.buffer_phase, p.drive_prepared.drive.ϵ_0, p.drive_prepared.drive.ω_p, p.drive_prepared.drive.pump_phase

    a = u[1]
    b = u[2]

    coupling = (Taylor_cos((a+conj(a))/sqrt(2) * sqrt(Z_a)/ϕ_quantum + (b+conj(b))/sqrt(2) * sqrt(Z_b)/ϕ_quantum, p.Taylor_order) - 1) / ϕ_quantum / sqrt(2)
    sin_pump = 2*E_J*p.drive_prepared.drive_pump(ϵ_0, ω_p, t, pump_phase)
        
    du[1] = -1im*(ω_a*a - sqrt(Z_a)*sin_pump*coupling) - p.environment_prepared.κ_a * a
    du[2] = -1im*(ω_b*b + p.drive_prepared.drive_buffer(ϵ_d, ω_d, t, buffer_phase) - sqrt(Z_b)*sin_pump*coupling) - p.environment_prepared.κ_b * b
    return nothing
end

#Test Lucas
function equation_test_Lucas!(ddu, du, u, p::SimulationParams{SystemATS_prepared, D, OneChain_prepared}, t) where D<:DriveTwoModes_prepared
    ω_a, ω_b, E_J, Z_a, Z_b, ϕ_quantum = p.system_prepared.system.ω_a, p.system_prepared.system.ω_b, p.system_prepared.system.E_J, p.system_prepared.Z_a, p.system_prepared.Z_b, p.system_prepared.ϕ_quantum
    ϵ_d, ω_d, buffer_phase, ϵ_0, ω_p, pump_phase =  p.drive_prepared.drive.ϵ_d, p.drive_prepared.drive.ω_d, p.drive_prepared.drive.buffer_phase, p.drive_prepared.drive.ϵ_0, p.drive_prepared.drive.ω_p, p.drive_prepared.drive.pump_phase

    ϕ_a = u[1]
    ϕ_b = u[2]
    Φ=u[3:end]

    coupling = (cos(ϕ_a * sqrt(Z_a)/ϕ_quantum + ϕ_b * sqrt(Z_b)/ϕ_quantum) - 1) / ϕ_quantum
    sin_pump = 2*E_J/2

    ddu[1] = -ω_a*(ω_a*ϕ_a - sqrt(Z_a) * sin_pump*coupling)
    ddu[2] = -ω_b*(ω_b*ϕ_b + p.drive_prepared.drive_buffer(ϵ_d, ω_d, t, buffer_phase)*sqrt(2) + p.drive_prepared.drive_buffer(ϵ_0, ω_p, t, pump_phase)*sqrt(2) - sqrt(Z_b) * sin_pump*coupling) + ω_b * p.environment_prepared.system_chain_coupling * sqrt(Z_b) * Φ[1]
    mul!(p.environment_prepared.temp_buf, p.environment_prepared.Mϕ_chain, Φ)
    @. p.environment_prepared.temp_buf = -p.environment_prepared.temp_buf
    p.environment_prepared.temp_buf[1] += p.environment_prepared.system_chain_coupling * sqrt(Z_b) * ϕ_b
    mul!(view(ddu, 3:length(ddu)), p.environment_prepared.Mq_chain, p.environment_prepared.temp_buf)
    return(nothing)
end

#Michiel's system
function default_equation!(ddu, du, u, p::SimulationParams{SystemMichiel_prepared, D, MeanField_Michiel}, t) where D<:DriveOneMode_prepared
    β, λ = p.system_prepared.system.β, p.system_prepared.system.λ
    ξ_d, ν_d, phase =  p.drive_prepared.drive.ϵ, p.drive_prepared.drive.ω, p.drive_prepared.drive.phase
    x = u[1]
    ddu[1] = - x - β/(sqrt(2)*λ)*sin(sqrt(2)*λ*x + p.drive_prepared.drive_func(ξ_d, ν_d, t, phase)+pi/2)  -  du[1] / p.environment_prepared.Q
    return(nothing)
end

#Michiel's system
function equation_1ODE!(du,u,params::SimulationParams{SystemMichiel_prepared, D, MeanField_Michiel}, t) where D<:DriveOneMode_prepared
    β, λ = params.system_prepared.system.β, params.system_prepared.system.λ
    ξ_d, ν_d, phase =  params.drive_prepared.drive.ϵ, params.drive_prepared.drive.ω, params.drive_prepared.drive.phase
    p = u[1]
    x = u[2]

    du[1] = - x - β/(sqrt(2)*λ)*sin(sqrt(2)*λ*x + params.drive_prepared.drive_func(ξ_d, ν_d, t, phase))  -  p / params.environment_prepared.Q
    du[2] = p
    return nothing
end

#ATS + Mean_Field + ladder_operator
function equation_stark!(du,u,p::SimulationParams{SystemATS_ladder_prepared, D, MeanField}, t) where D<:DriveTwoModes_prepared
    ω_a, ω_b, E_J, Z_a, Z_b, ϕ_quantum = p.system_prepared.system.ω_a, p.system_prepared.system.ω_b, p.system_prepared.system.E_J, p.system_prepared.Z_a, p.system_prepared.Z_b, p.system_prepared.ϕ_quantum
    ϵ_d, ω_d, buffer_phase, ϵ_0, ω_p, pump_phase =  p.drive_prepared.drive.ϵ_d, p.drive_prepared.drive.ω_d, p.drive_prepared.drive.buffer_phase, p.drive_prepared.drive.ϵ_0, p.drive_prepared.drive.ω_p, p.drive_prepared.drive.pump_phase

    a = u[1]
    b = u[2]

    sin_pump =  - 2*E_J*sin(ϵ_0)
    coupling = (cos((a+conj(a))/sqrt(2) * sqrt(Z_a)/ϕ_quantum + (b+conj(b))/sqrt(2) * sqrt(Z_b)/ϕ_quantum)) / ϕ_quantum / sqrt(2)
        
    du[1] = -1im*(ω_a*a + sqrt(Z_a)*sin_pump*coupling) - p.environment_prepared.κ_a * a
    du[2] = -1im*(ω_b*b + p.drive_prepared.drive_buffer(ϵ_d, ω_d, t, buffer_phase) + sqrt(Z_b)*sin_pump*coupling) - p.environment_prepared.κ_b * b
    return nothing
end

#Michiel's writing of the ATS
function default_equation!(ddu, du, u, p::SimulationParams{SystemATS_Michiel_prepared, D, MeanField}, t) where D<:DriveOneMode_prepared
    β_a, β_b, λ_a, λ_b = p.system_prepared.system.β_a, p.system_prepared.system.β_b, p.system_prepared.system.λ_a, p.system_prepared.system.λ_b
    ϵ_d, ν_d, phase =  p.drive_prepared.drive.ϵ, p.drive_prepared.drive.ω, p.drive_prepared.drive.phase

    x_a = u[1]
    x_b = u[2]
    
    coupling = cos(sqrt(2)*λ_a*x_a + sqrt(2)*λ_b*x_b) - 1

    ddu[1] = - (x_a - β_a/(sqrt(2)*λ_a)*coupling)/4  - 2 * p.environment_prepared.κ_a * du[1]
    ddu[2] = - (x_b + p.drive_prepared.drive_func(ϵ_d, ν_d, t, phase) - β_b/(sqrt(2)*λ_b)*coupling)  -  2 * p.environment_prepared.κ_b * du[2]
    return(nothing)
end



function equation_1ODE!(du, u, p::SimulationParams{SystemATS_Michiel_prepared, D, MeanField}, t) where D<:DriveOneMode_prepared
    β_a, β_b, λ_a, λ_b = p.system_prepared.system.β_a, p.system_prepared.system.β_b, p.system_prepared.system.λ_a, p.system_prepared.system.λ_b
    ϵ_d, ν_d, phase =  p.drive_prepared.drive.ϵ, p.drive_prepared.drive.ω, p.drive_prepared.drive.phase

    x_a = u[1]
    x_b = u[2]
    p_a = u[3]
    p_b = u[4]
    
    coupling = cos(sqrt(2)*λ_a*x_a + sqrt(2)*λ_b*x_b) - 1

    du[1] = p_a / 2
    du[2] = p_b
    du[3] = (- (x_a - β_a/(sqrt(2)*λ_a)*coupling)  - 2 * p.environment_prepared.κ_a * p_a) / 2
    du[4] = - (x_b + p.drive_prepared.drive_func(ϵ_d, ν_d, t, phase) - β_b/(sqrt(2)*λ_b)*coupling)  -  2 * p.environment_prepared.κ_b * p_b
    return(nothing)
end