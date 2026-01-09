using LinearAlgebra, Plots

# Function to create the matrices of the system of coupled harmonic oscillators
function nearest_neighbor_coupling(ω, C; gq=nothing, gϕ=nothing)
    """ 
    #INPUT
    ω : vector of onsite energies
    C : vector of capacitances
    gq : vector of hopping energies for momentum coupling
    gϕ : vector of hopping energies for position coupling
    

    #OUTPUT
    Mq : mass matrix
    Mϕ : stiffness matrix
    """
    if gq === nothing
        Mq = Diagonal(1.0 ./ C)
        if gϕ === nothing
            d = C .* ω.^2
            Mϕ = Diagonal(d)
        else
            d = C .* ω.^2
            Mϕ = SymTridiagonal(d, gϕ)
        end
    else
        Mq = SymTridiagonal(1.0 ./ C,gq)
        if gϕ === nothing
            d = C .* ω.^2
            Mϕ = Diagonal(d)
        else
            d = C .* ω.^2
            Mϕ = SymTridiagonal(d, gϕ)
        end
    end    
    return Mq, Mϕ
end

# Function to solve the normal modes of a system of coupled oscillators, looking at the normal modes, in an analytical way
function solve_normal_modes(Mq, Mϕ, ϕ_0, q_0, tspan, dt; energy=false)
    """ 
    #INPUT
    Mq : mass matrix
    Mϕ : stiffness matrixs
    ϕ_0 : initial value for the position of system oscillator
    q_0 : initial value for the momentum of the system oscillator
    tspan = (tmin, tmax) : time span
    dt : time step for the solver, default is 1/(10*maximum(ω)) to verify Shannon's theorem
    energy : boolean, if true compute the energy of the system

    #OUTPUT
    tlist : time list
    ϕ_t : time evolution of the position of system oscillator
    q_t : time evolution of the momentum of system oscillator
    E : time evolution of the energy of the system if energy=true
    """
    npoints = Int(round((tspan[2]-tspan[1])/dt))  # number of time points

    inv_Mq = inv(Mq)  # inverse of mass matrix

    # A = M1 * M2, to be diagonalized
    A = Mq * Mϕ
    vals, vecs = eigen(A)

    # Frequencies and mode vectors
    ωk² = real(vals)

    if count( x -> (x < 0), ωk²) > 0
        display(print(vecs*ϕ_0))
        display(scatter(ωk², label="Eigenvalues"))
        error("Negative eigenvalues found in the system.")
    end

    ωk = sqrt.(ωk²)
    Φk = vecs  # columns = normal modes
    Φk_inv=inv(Φk)
    # Project initial conditions into normal mode basis
    a0 = Φk_inv * ϕ_0
    adot0 = Φk_inv * Mq * q_0

    # Time evolution in normal coordinates
    tlist = range(tspan[1], tspan[2], length=npoints)
    a_t = [a0 .* cos.(ωk .* t) .+ (adot0 ./ ωk) .* sin.(ωk .* t) for t in tlist]
    a_t_dot= [a0 .* (-ωk) .* sin.(ωk .* t) .+ (adot0) .* cos.(ωk .* t) for t in tlist]

    # Reconstruct real-space fields
    ϕ_t = [Φk * a for a in a_t]
    q_t = [inv_Mq * Φk * a for a in a_t_dot]  # q = \dot{ϕ}/Mq


    if energy
        # Compute energy
        E = 0.5 * [q_t[t]' * Mq * q_t[t] + ϕ_t[t]' * (Mϕ) * ϕ_t[t] for t in 1:npoints]
        return tlist, ϕ_t, q_t,E
    end

    return tlist, ϕ_t, q_t
end

# funcion to solve numerically with step by step integration

function solve_numeric(Mq,Mϕ,ϕ_0, q_0, tspan, dt ; energy=false, method=VelocityVerlet())
    """
    #INPUT
    Mq : mass matrix
    Mϕ : stiffness matrix
    ϕ_0 : initial value for the position of system oscillator
    q_0 : initial value for the momentum of the system oscillator
    tspan = (tmin, tmax) : time span
    method : method to use for the integration, default is VelocityVerlet
    dt : time step for the solver, default is 1/(10*maximum(ω)) to verify Shannon's theorem
    energy : boolean, if true compute the energy of the system

    #OUTPUT
    ϕ : time evolution of the position of system oscillator
    q : time evolution of the momentum of system oscillator
    sol.t : time list
    E : time evolution of the energy of the system if energy=true
    """
    # Definition of the second order ODE
    function coupled_pendulum_2ODE(ddu,du,u,p,t)
        ddu[1:end]=-A*u[1:end]
    end

    # Initial conditions
    ϕ_dot_0 = Mq*q_0


    A=Mq*Mϕ
    prob = SecondOrderODEProblem(coupled_pendulum_2ODE, ϕ_dot_0, ϕ_0, tspan, A)
    sol = solve(prob, method,dt=dt)
    ϕ_dot = [getindex.(sol.u,i) for i in 1:n+1]
    ϕ = [getindex.(sol.u,i+n+1) for i in 1:n+1]

    inv_Mq = inv(Mq)
    q = [inv_Mq*getindex.(ϕ_dot,t) for t in 1:length(sol.t)]

    if energy
        # Compute energy
        E = 0.5 * [q[t]' * Mq * q[t] + getindex.(ϕ,t)' * Mϕ * getindex.(ϕ,t) for t in 1:length(sol.t)]
        return(ϕ,q,sol.t,E)
    end
    return(ϕ,q,sol.t)
end

########################################################
# Example usage
########################################################
# n = 10
# ω = LinRange(10, 10, n+1)
# g = vcat([0.2], fill(0.2, n))  # uniform coupling
# C = fill(0.01, n+1)

# ϕ_init = 1.0
# q_init = 1.0

# ϕ_0 = vcat(SVector(ϕ_init),SVector{n}(zeros(n)))
# q_0 = vcat(SVector(q_init),SVector{n}(zeros(n)))

# tspan = (0.0, 4π / minimum(ω))
# dt=1/(10*maximum(ω))


# Mq,Mϕ=nearest_neighbor_coupling(ω,C,gϕ=g)


# #First we solve the system of coupled oscillators in an analytical way
# t, ϕ_t, q_t,E = solve_normal_modes(Mq,Mϕ, ϕ_0, q_0, tspan, dt, energy=true)

# # Now we solve the system of coupled oscillators in a numerical way
# ϕ_0 = vcat([ϕ_init],fill(0.0,n))
# q_0 = vcat([q_init],fill(0.0,n))

# ϕ,q,t_prime,E_prime=solve_numeric(Mq,Mϕ,ϕ_0, q_0, tspan, 1/(10*maximum(ω)), energy=true)

# plot(t, [ϕ_t[i][1] for i in 1:length(t)], label="ϕ₀(t) symplectic", xlabel="t")
# plot!(t_prime, [ϕ[1][i] for i in 1:length(t_prime)], label="ϕ₀(t) numerical solver")
;