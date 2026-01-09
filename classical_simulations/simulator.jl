using DifferentialEquations
include("/Users/tmalas/Desktop/Dev/Simulations_local/classical_simulations/config/systems.jl")
include("/Users/tmalas/Desktop/Dev/Simulations_local/classical_simulations/config/drives.jl")
include("/Users/tmalas/Desktop/Dev/Simulations_local/classical_simulations/config/environments.jl")
include("/Users/tmalas/Desktop/Dev/Simulations_local/classical_simulations/config/equations.jl")
include("/Users/tmalas/Desktop/Dev/Simulations_local/classical_simulations/config/solvers.jl")

"""
INPUT
t_max : maximum time for the simulation
config_name : name of the configuration
config_params : parameters for the configuration
equation_name : name of the differential equation
solver_name : name for the solving function
"""

function simulator(t_max::Float64, system::AbstractSystem, drive::AbstractDrive, environment::AbstractEnv; t_min=0.0, equation=default_equation!, solver=default_solver, kwargs...)
    
    tspan = (t_min, t_max) # Time span for the simulation

    #preparation
    system_prepared = prepare(system)
    drive_prepared = prepare(drive)
    environment_prepared = prepare(system,environment)

    dt = get_dt(system, environment)
    u_0_tot = get_init(system_prepared, environment_prepared)

    p = SimulationParams(system_prepared, drive_prepared, environment_prepared)
    
    output = solver(u_0_tot, tspan, p, dt; equation=equation, kwargs...)

    return output
end