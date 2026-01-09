using LinearAlgebra, MPSDynamics, Statistics, StaticArrays, DifferentialEquations
include("/Users/tmalas/Desktop/Dev/Simulations_local/Code_Julia/Julia_file/Plot_setting.jl")
include("/Users/tmalas/Desktop/Dev/Simulations_local/Code_Julia/Julia_file/Utils.jl")

## Bath definition ##
"""
INPUT :
bath_params 
    ωc_dwn : lower cut-off frequency of the spectral density J(ω) in GHz
    ωc_up : upper cut-off frequency of the spectral density J(ω) in GHz

OUTPUT :
AB : Array of the segments of the frequency ranges where the spectral density J(ω) is defined.
        Each segment is an array with two elements [ω_min ω_max] in GHz.
Jω_fct : spectral density J(ω)
"""
function Ohmic_bath(;bath_params=(ωc_dwn=0.0*2*pi, ωc_up=10*2*pi), κ=1/20, C_inv=0.147*2*pi, plot_bool= false)


    ωc_dwn, ωc_up = bath_params.ωc_dwn, bath_params.ωc_up

    ### Caracterization of the impedance #####
    α = 2*κ/C_inv # no unit

    # We design the impedance to be a smooth pass-band filter
    ω_max = ωc_up-.8*2*pi # Lower cut-off of the spectral density J(ω), GHz
    ω_min = ωc_dwn+.8*2*pi # Cut-off of the spectral density J(ω), GHz


    # AB segments correspond to the different frequency ranges where the spectral density is defined.
    # It corresponds to i=1 ; i=2 and i=3 of the Jω_fct(ω,i) function.

    AB = [[-Inf ωc_dwn];[ωc_dwn ω_min];[ω_min ω_max];[ω_max ωc_up];[ωc_up Inf]]
    #AB=[[-Inf ω_min];[ω_min ω_max];[ω_max Inf]]


    function Jω_fct(ω,i)
        if i==1 # First segment of AB (here [-Inf ωc_dwn]), zero for negative frequencies when β == Inf
            y = 0
        elseif i==2 # Second segment of AB (here [ωc_dwn ω_min])
            y = @. smoothstep((ω - ωc_dwn) / (ω_min - ωc_dwn))
        elseif i==3 # Third segment of AB (here [ω_min ω_max])
            y = 1
        elseif i==4 # Fourth segment of AB (here [ω_max ωc_up])``
            y = @. (1 - smoothstep((ω - ω_max) / (ωc_up - ω_max)))
        elseif i==5 # Fifth segment of AB (here [ωc_up Inf])
            y = 0
        end
        return y*α.*ω
    end
    if plot_bool
        ω = range(ωc_dwn, stop=ωc_up, length=1000)
        J_values = zeros(length(ω))
        for i in 1:length(ω)
            if ω[i] < ω_min
                J_values[i] = Jω_fct(ω[i], 2)
            elseif ω[i] < ω_max
                J_values[i] = Jω_fct(ω[i], 3)
            else
                J_values[i]=Jω_fct(ω[i], 4)
            end
        end
        return plot(ω/2pi, J_values, label="J(ω)", xlabel="ω/2π (GHz)", ylabel="J(ω)")
    end
    return AB, Jω_fct
end

"""
INPUT :
bath_params 
    ω_c : characteristic frequency of the exponential decay in GHz
    ω_cut_off : cut-off frequency of the spectral density J(ω) in GHz

OUTPUT :
AB : Array of the segments of the frequency ranges where the spectral density J(ω) is defined.
        Each segment is an array with two elements [ω_min ω_max] in GHz.
Jω_fct : spectral density J(ω)
"""
function Exp_decay(;bath_params=(ω_0=50*2pi, ω_cut_off=90), κ=1/20, C_inv=0.147*2*pi, plot_bool= false)


    ω_0, ω_cut_off = bath_params.ω_0, bath_params.ω_cut_off
    ω_up_zero = ω_cut_off + 0.5*2*pi # Upper cut-off of the spectral density J(ω), GHz
    
    ### Caracterization of the impedance #####
    α = 2*κ/C_inv # no unit

    AB=[[-Inf 0];[0 ω_cut_off];[ω_cut_off ω_up_zero];[ω_up_zero Inf]]

    function J_exp_decay(ω,i)
        if i==1 # First segment of AB (here [-Inf 0]), zero for negative frequencies when β == Inf
            y = 0
        elseif i==2 # Second segment of AB (here [0 ω_cut_off])
            y = @.α*ω*exp(-ω/ω_0)
        elseif i==3 # Third segment of AB (here [ω_cut_off ω_up_zero])
            y = @. α*ω*exp(-ω_cut_off/ω_0)*(1 - smoothstep((ω - ω_cut_off) / (ω_up_zero - ω_cut_off)))
        elseif i==4 # Fourth segment of AB (here [ω_up_zero Inf])
            y=0
        end
        return y
    end
    if plot_bool
        ω = range(0, stop=ω_up_zero, length=1000)
        J_values = zeros(length(ω))
        for i in 1:length(ω)
            if ω[i] < ω_cut_off
                J_values[i] = J_exp_decay(ω[i], 2)
            elseif ω[i] < ω_up_zero
                J_values[i] = J_exp_decay(ω[i], 3)
            end
        end
        return plot(ω/2pi, J_values, label="J(ω)", xlabel="ω/2π (GHz)", ylabel="J(ω)")
    end
    return AB, J_exp_decay
end

"""
INPUT :
bath_params
    Ω : characteristic frequency of the Drude-Lorentz function in GHz
    ω_cut_off : cut-off frequency of the spectral density J(ω) in GHz

OUTPUT :
onsite_energies : vector of on-site energies ϵ_i of the chain
hopping_energies : vector of hopping energies t_i of the chain
system_chain_coupling : coupling between the system and the chain
"""
function Drude_lorentz(;bath_params=(Ω=50*2pi, ω_cut_off=100), κ=1/20, C_inv=0.147*2*pi, plot_bool= false)


    Ω, ω_cut_off = bath_params.Ω, bath_params.ω_cut_off

    AB_drude= [[-Inf 0];[0 ω_cut_off];[ω_cut_off Inf]]
    function J_drude(ω,i)
        if i==1 # First segment of AB (here [-Inf 0]), zero for negative frequencies when β == Inf
            y = 0
        elseif i==2 # Second segment of AB (here [0 ω_cut_off])
            y = @. 2*κ/(C_inv)*ω*Ω^2/((ω^2+Ω^2))
        elseif i==3 # Third segment of AB (here [ω_cut_off Inf])
            y = 0
        end
        return y
    end
    if plot_bool
        ω = range(0, stop=ω_cut_off, length=1000)
        J_values = [J_drude(ω[i], 2) for i in 1:length(ω)]
        plot(ω, J_values, label="J(ω)", xlabel="ω (GHz)", ylabel="J(ω)")
    end
    return AB_drude, J_drude
end

## Find the optimal chain length ##

function endsiteocc_d(t, cparams)
    es=cparams[1]
    ts=cparams[2]
    N=length(es)
    hmat = SymTridiagonal(es, ts)
    F = eigen(hmat)
    U = F.vectors
    S = F.values
    real.(transpose(U[:,1].*exp.(im*t.*S))*U[:,N]*transpose(U[:,N])*(U[:,1].*exp.(-im*t.*S)))
end

function findchainlength_dich(T, cparams::Vector; eps=1e-4, verbose=false)
    Nmax = length(cparams[1])

    # Check if the full chain already fails the precision test
    occ_full = endsiteocc_d(T, [cparams[1][1:Nmax], cparams[2][1:Nmax-1]])
    occ_one_less = endsiteocc_d(T, [cparams[1][1:Nmax-1], cparams[2][1:Nmax-2]])
    if abs(occ_full - occ_one_less) > eps
        error("Suitable truncation not found, try larger starting chain length")
    end

    low = 1
    high = Nmax - 1
    best = Nmax - 1  # initialize to full chain minus 1

    while low <= high
        mid = (low + high) ÷ 2
        occ_mid = endsiteocc_d(T, [cparams[1][1:mid], cparams[2][1:mid-1]])
        occ_next = endsiteocc_d(T, [cparams[1][1:mid+1], cparams[2][1:mid]])

        verbose && println("Testing ntrunc = $mid")

        if abs(occ_mid - occ_next) > eps
            low = mid + 1  # too short: try longer
        else
            best = mid     # found acceptable length
            high = mid - 1 # try even shorter
        end
    end

    return best
end;

## Comparison between simulations ##
"""
INPUT :
ϕ : vector of the field ϕ
q : vector of the field q
L : inductance
C_inv : inverse capacitance in GHz

OUTPUT :
Energy of the linear system
"""
function Energy_linear(ϕ, q, (L, C_inv))

    return q.^2 /2 *C_inv + ϕ.^2 / (2*L)
end

"""
INPUT :
ϕ : vector of the field ϕ
q : vector of the field q
E_J : Josephson energy in GHz
C_inv : inverse capacitance in GHz

OUTPUT :
Energy of the non-linear system
"""
function Energy_non_linear(ϕ, q, (E_J, C_inv))

    ϕ_quantum = 1/2 # ħ/e 
    return q.^2 /2 *C_inv - E_J*cos.(ϕ/ϕ_quantum)
end

function comparison_simu(simu_list,E, E_params)
    ϕ_list = []
    q_list = []
    t_list = []
    E_list = []
    spectrum_list = []

    for simu in simu_list
        ϕ, q, t = simu
        push!(ϕ_list, ϕ)
        push!(q_list, q)
        push!(t_list, t)
        push!(E_list, E(ϕ, q, E_params))
        spectrum = FFT_perso(ϕ, t)
        push!(spectrum_list, spectrum)
    end

    return(E_list, spectrum_list)
end


############################################################
## Solvers #################################################
############################################################

## Linear oscillator solver ##
"""
INPUT :
n : number of oscillators in the chain
bath : function to define the spectral density J(ω) of the bath
bath_params : parameters of the bath function
ω_sys : system frequency in GHz (default: 5 * 2π)
κ : decay rate of the bath in GHz (default: 1/20)
C_inv : inverse of the capacitance in GHz (default: 0.147 * 2π)
β : inverse temperature (default: ∞ for zero temperature T=0)
energy : boolean to indicate if the energy should be calculated (default: false)
all_modes : boolean to indicate if all modes should be returned (default: false)
emission : boolean to indicate if the emission spectrum should be calculated (default: false)

OUTPUT :
ϕ_sys : array of the system's ϕ_i values at each time step
q_sys : array of the system's q_i values at each time step
sol.t : time vector for the simulation
E : array of the energy values at each time step if energy=true
"""
function One_d_linear(n, bath, bath_params, t_max; ω_sys=5*2pi, κ=1/20, C_inv=0.147*2*pi, β=∞, energy=false, all_modes=false, emission=false)


    AB, J = bath(bath_params=bath_params, κ=κ, C_inv=C_inv)
    onsite_energies,hopping_energies,system_chain_coupling = chaincoeffs_finiteT(n, β, false; J=J, mc=size(AB)[1], mp=0, AB=AB, iq=1, idelta=2, procedure=:Lanczos, Mmax=5000, save=false)  # chain parameters, i.e. on-site energies ϵ_i, hopping energies t_i, and system-chain coupling c_0

    ### Mapping ###
    Z_sys = C_inv / ω_sys # Characteristic impedance in GHz
    ω = vcat(ω_sys,onsite_energies) # GHz
    g_q=vcat(0, -hopping_energies) 
    g_ϕ=vcat(-system_chain_coupling*sqrt(Z_sys), hopping_energies) 

    ϕ_init = 1.0
    q_init = 1.0

    ϕ_0 = vcat(SVector(ϕ_init/sqrt(Z_sys)),SVector{n}(zeros(n)))
    q_0 = vcat(SVector(q_init*sqrt(Z_sys)),SVector{n}(zeros(n)))

    Mq=SymTridiagonal(ω, g_q)
    inv_Mq= inv(Mq)
    Mϕ=SymTridiagonal(ω, g_ϕ)

    A=Mq * Mϕ
    vals, vecs = eigen(A)

    ωk² = real(vals)
    ωk = sqrt.(ωk²)
    P = real(vecs)  # columns = normal modes

    P_inv=inv(P)

    # Project initial conditions into normal mode basis
    ϕ_0_normal_modes = (P_inv * ϕ_0)
    ϕ_dot_0_normal_modes = (P_inv * Mq * q_0)

    tspan = (0.0, t_max)
    dt=1/(20*maximum(ωk))
    npoints = Int(round((tspan[2]-tspan[1])/dt))  # number of time points


    # Time evolution in normal coordinates
    tlist = range(tspan[1], tspan[2], length=npoints)
    ϕ_t_normal_modes = [ϕ_0_normal_modes .* cos.(ωk .* t) .+ (ϕ_dot_0_normal_modes ./ ωk) .* sin.(ωk .* t) for t in tlist]
    ϕ_t_dot_normal_modes= [ϕ_0_normal_modes .* (-ωk) .* sin.(ωk .* t) .+ (ϕ_dot_0_normal_modes) .* cos.(ωk .* t) for t in tlist]
    if all_modes
        q_t_normal_modes = [P_inv * inv_Mq * P * a for a in ϕ_t_dot_normal_modes]
        return ϕ_t_normal_modes,  q_t_normal_modes, tlist
    end

    # Reconstruct real-space fields
    ϕ_t = [P * a for a in ϕ_t_normal_modes]
    if emission
        Mq_chain = SymTridiagonal(onsite_energies, -hopping_energies)
        Mϕ_chain = SymTridiagonal(onsite_energies, hopping_energies)
        ϕ_dot_t = [P * a for a in ϕ_t_dot_normal_modes]
        return ϕ_t, ϕ_dot_t, tlist, Mq_chain,Mϕ_chain
    end
    q_t = [inv_Mq * P * a for a in ϕ_t_dot_normal_modes]  # q = \dot{ϕ}/Mq

    ϕ_sys =  [ϕ_t[i][1] for i in 1:length(tlist)]*sqrt(Z_sys)
    q_sys =  [q_t[i][1] for i in 1:length(tlist)]/ sqrt(Z_sys);



    if energy
        E = 0.5 * [q_t[t]' * (Mq*Z_sys) * q_t[t] + ϕ_t[t]' * (Mϕ/Z_sys) * ϕ_t[t] for t in 1:npoints]
        return(ϕ_sys, q_sys, sol.t, E)
    end
    return(ϕ_sys, q_sys, tlist)
end

"""
INPUT :
ω_sys : system frequency in GHz (default: 5 * 2π)
κ : damping rate in GHz
C_inv : inverse capacitance in GHz

OUTPUT :
ϕ_analytic : array of the time evolution of the phase ϕ(t)
q_analytic : array of the time evolution of the charge q(t)
t : time array
"""
function CL_linear(t_max; ω_sys=5*2pi, κ=1/20, C_inv=0.147*2*pi)

    ϕ_init = 1.0
    q_init = 1.0
    X_0=[ϕ_init,q_init]
    
    tspan = (0.0, t_max)
    dt=1/(20*ω_sys)
    npoints = Int(round((tspan[2]-tspan[1])/dt))  # number of time points
    # Time evolution in normal coordinates
    t = range(tspan[1], tspan[2], length=npoints)

    Ω=sqrt(ω_sys^2-κ^2)
    q_analytic= @. exp(-κ*t)*(X_0[2]*cos(Ω*t) - (κ*X_0[2]+ω_sys^2*X_0[1]/C_inv)*sin(Ω*t)/Ω)
    ϕ_analytic= @. exp(-κ*t)*(X_0[1]*cos(Ω*t) + (X_0[2]*C_inv+κ*X_0[1])*sin(Ω*t)/Ω);

    return(ϕ_analytic, q_analytic, t)
end

## Non-linear oscillator solver ##

"""
INPUT :
n : number of oscillators in the chain
bath : function to define the spectral density J(ω) of the bath
bath_params : parameters of the bath function
E_J : Josephson energy in GHz (default: 90 * 2π / 2)
κ : decay rate of the bath in GHz (default: 1/20)
C_inv : inverse of the capacitance in GHz (default: 0.147 * 2π)
β : inverse temperature (default: ∞ for zero temperature T=0)
energy : boolean to indicate if the energy should be calculated (default: false)

OUTPUT :
ϕ_sys : array of the system's ϕ_i values at each time step
q_sys : array of the system's q_i values at each time step
sol.t : time vector for the simulation
E : array of the energy values at each time step if energy=true
"""
function One_d_non_linear(n, bath, bath_params, t_max;E_J=90 *2pi /2, κ=1/20, C_inv=0.147*2*pi, β=∞, energy=false)


    AB, J = bath(bath_params=bath_params, κ=κ, C_inv=C_inv)
    onsite_energies,hopping_energies,system_chain_coupling = chaincoeffs_finiteT(n, β, false; J=J, mc=size(AB)[1], mp=0, AB=AB, iq=1, idelta=2, procedure=:Lanczos, Mmax=5000, save=false)  # chain parameters, i.e. on-site energies ϵ_i, hopping energies t_i, and system-chain coupling c_0

    function damped_non_linear_chain_2ODE(ddu,du,u,p,t)
        ddu[1] = - E_J*ω_nl * sqrt(Z_nl) * sin(u[1] * sqrt(Z_nl)/ϕ_quantum) / ϕ_quantum + ω_nl * system_chain_coupling * sqrt(Z_nl) * u[2]
        temp = Mϕ_chain*u[2:end]
        temp[1] -= system_chain_coupling * sqrt(Z_nl) * u[1]
        ddu[2:end]=-Mq_chain*temp[1:end]
        #ddu[1:end] = - Mq_bis*Mϕ_bis*u[1:end]
    end

    ϕ_quantum = 1/2 # ħ/e 
    ω_nl = sqrt(4*C_inv*E_J)
    Z_nl=C_inv/ω_nl
    gq=vcat(-hopping_energies)
    gϕ=vcat(hopping_energies)
    Mq_chain=SymTridiagonal(onsite_energies, gq)
    Mϕ_chain=SymTridiagonal(onsite_energies, gϕ)
    Mq_bis=SymTridiagonal(vcat([ω_nl], onsite_energies), vcat([0],gq))


    #Initial conditions
    ϕ_init = 1.0
    q_init = 1.0

    ϕ_0 = vcat(ϕ_init/sqrt(Z_nl), fill(0.0, n))
    q_0 = vcat(q_init * sqrt(Z_nl), fill(0.0, n))
    ϕ_dot_0 = Mq_bis*q_0


    #Run the simulation
    tspan = (0.0, t_max) # Time span for the simulation
    dt = 1 / (20 * maximum(vcat(ω_nl,onsite_energies))) # Time step for the simulation
    prob = SecondOrderODEProblem(damped_non_linear_chain_2ODE, ϕ_dot_0, ϕ_0, tspan)
    sol = solve(prob, VelocityVerlet(),dt=dt)
    ϕ_dot = [getindex.(sol.u,i) for i in 1:n+1]
    ϕ = [getindex.(sol.u,i+n+1) for i in 1:n+1]

    ϕ_sys = ϕ[1] * sqrt(Z_nl)
    inv_Mq_bis = inv(Mq_bis)
    q = [inv_Mq_bis*getindex.(ϕ_dot,t) for t in 1:length(sol.t)]
    q = [getindex.(q,i) for i in 1:n+1]
    q_sys = q[1] / sqrt(Z_nl)

    if energy
        E=q_sys.^2 * C_sys_inv /2 - E_J*cos.(ϕ_sys/ϕ_quantum) - system_chain_coupling * ϕ_sys .* ϕ[2] + 1/2*[getindex.(q,i)[2:end]'*Mq_chain*getindex.(q,i)[2:end] + getindex.(ϕ,i)[2:end]'*Mϕ_chain*getindex.(ϕ,i)[2:end] for i in 1:length(sol.t)];
        return(ϕ_sys, q_sys, sol.t, E)
    end
    return(ϕ_sys, q_sys, sol.t)
end

"""
INPUT :
E_J : Josephson energy in GHz
κ : damping rate in GHz
C_inv : inverse capacitance in GHz

OUTPUT :
ϕ_CL : array of the time evolution of the phase ϕ(t)
q_CL : array of the time evolution of the charge q(t)
t : time array
"""
function CL_non_linear(t_max; E_J=90 *2pi /2, κ=1/20, C_inv=0.147*2*pi)

    # Definition of the second order ODE
    function damped_non_linear(ddu, du, u, p, t)
        ddu[1] = -2 * κ * du[1] - 4 * ϕ_quantum * E_J * C_inv *sin(u[1] / ϕ_quantum)
    end

    ϕ_quantum = 1/2 # ħ/e 
    ω_nl = sqrt(4*C_inv*E_J)
    Z_nl=C_inv/ω_nl


    # Initial conditions
    ϕ_0 = [1.]
    q_0 = [1.]
    ϕ_dot_0 = q_0 * C_inv

    tspan = (0.0, t_max) # Time span for the simulation
    dt = 1 / (20 * ω_nl) # Time step for the simulation

    prob = SecondOrderODEProblem(damped_non_linear,ϕ_dot_0, ϕ_0, tspan)
    sol = solve(prob, VelocityVerlet(), dt=dt)

    # Extract solution arrays
    ϕ_dot_CL = [u[1] for u in sol.u]
    ϕ_CL = [u[2] for u in sol.u]
    q_CL = ϕ_dot_CL / C_inv

    return(ϕ_CL, q_CL, sol.t)
end

## Non-linear driven oscillator solver ##

function One_d_non_linear_driven(n, bath, bath_params, t_max;E_J=90 *2pi /3, κ=1/20, C_inv=0.147*2*pi, ω=sqrt(4*C_inv*E_J), A=5, β=∞, energy=false,fast_no_chain=true,emission=false)
    """
    INPUT :
    n : number of oscillators in the chain
    bath : function to define the spectral density J(ω) of the bath
    bath_params : parameters of the bath function
    E_J : Josephson energy in GHz (default: 90 * 2π / 3)
    κ : decay rate of the bath in GHz (default: 1/20)
    C_inv : inverse of the capacitance in GHz (default: 0.147 * 2π)
    ω : frequency of the driving force in GHz (resonant by default : ω_nl = sqrt(4*C_inv*E_J))
    A : amplitude of the driving force (default: 5)
    β : inverse temperature (default: ∞ for zero temperature T=0)
    energy : boolean to indicate if the energy should be calculated (default: false)
    fast_no_chain : boolean to indicate if the chain state should not be saved (default: true)

    OUTPUT :
    ϕ_sys : array of the system's ϕ_i values at each time step
    q_sys : array of the system's q_i values at each time step
    sol.t : time vector for the simulation
    E : array of the energy values at each time step if energy=true
    """

    AB, J = bath(bath_params=bath_params, κ=κ, C_inv=C_inv)
    onsite_energies,hopping_energies,system_chain_coupling = chaincoeffs_finiteT(n, β, false; J=J, mc=size(AB)[1], mp=0, AB=AB, iq=1, idelta=2, procedure=:Lanczos, Mmax=5000, save=false)  # chain parameters, i.e. on-site energies ϵ_i, hopping energies t_i, and system-chain coupling c_0

    temp_buf = zeros(n)
    function damped_non_linear_chain_2ODE_driven(ddu,du,u,p,t)
        ϕ=u[1]
        Φ=u[2:end]
        ddu[1] = - E_J*ω_nl * sqrt(Z_nl) * (ϕ * sqrt(Z_nl)/ϕ_quantum) / ϕ_quantum + ω_nl * system_chain_coupling * sqrt(Z_nl) * Φ[1] + A*cos(ω * t) *sqrt(2) *ω_nl
        #ddu[1] = - E_J*ω_nl * sqrt(Z_nl) * u[1] * sqrt(Z_nl)/ϕ_quantum / ϕ_quantum + ω_nl * system_chain_coupling * sqrt(Z_nl) * u[2] + A*cos(ω * t) / sqrt(Z_nl) #if we want to use the linearized equation
        mul!(temp_buf, Mϕ_chain, Φ)
        @. temp_buf = -temp_buf
        temp_buf[1] += system_chain_coupling * sqrt(Z_nl) * ϕ
        mul!(view(ddu, 2:length(ddu)), Mq_chain, temp_buf)
        #ddu[1:end] = - Mq_bis*Mϕ_bis*u[1:end] # if we want to use the linearized equation in a more robust way
    end


    ϕ_quantum = 1/2 # ħ/e 
    ω_nl = sqrt(4*C_inv*E_J)
    Z_nl=C_inv/ω_nl
    gq=vcat(-hopping_energies)
    gϕ=vcat(hopping_energies)
    Mq_chain=SymTridiagonal(onsite_energies, gq)
    Mϕ_chain=SymTridiagonal(onsite_energies, gϕ)
    Mq_bis=SymTridiagonal(vcat([ω_nl], onsite_energies), vcat([0],gq))
    #Mϕ_bis=SymTridiagonal(vcat([ω_nl], onsite_energies), vcat([-system_chain_coupling * sqrt(Z_nl)],gϕ))

    #Initial conditions
    ϕ_init = 1.0
    q_init = 1.0

    ϕ_0 = vcat(ϕ_init/sqrt(Z_nl), fill(0.0, n))
    q_0 = vcat(q_init * sqrt(Z_nl), fill(0.0, n))
    ϕ_dot_0 = Mq_bis*q_0


    #Run the simulation
    tspan = (0.0, t_max) # Time span for the simulation
    dt = 1 / (40 * maximum(vcat(ω_nl,onsite_energies))) # Time step for the simulation
    prob = SecondOrderODEProblem(damped_non_linear_chain_2ODE_driven, ϕ_dot_0, ϕ_0, tspan)

    if emission
        save_idxs=vcat(2:n+1,n+3:n+2+n) # We save the chain DoFs (Φ)
        sol = solve(prob, VelocityVerlet(),dt=dt,
        saveat=t_max,                      # how often to save (or choose a coarser step)
        save_idxs=save_idxs,            # only store system DoFs
        save_everystep=false            # do not store intermediate solver steps
        )
        ϕ_dot = [getindex.(sol.u,i)[end] for i in 1:n]
        ϕ = [getindex.(sol.u,i+n)[end] for i in 1:n]
        return(ϕ,ϕ_dot,sol.t,Mq_chain, Mϕ_chain)
    
    end

    if fast_no_chain
        save_idxs=[1,n+2] # We only save the system DoFs (ϕ_a, ϕ_b) and not the chain DoFs (Φ)
        sol = solve(prob, VelocityVerlet(),dt=dt,
        saveat=dt,                      # how often to save (or choose a coarser step)
        save_idxs=save_idxs,            # only store system DoFs
        save_everystep=false            # do not store intermediate solver steps
        )

        ϕ_dot = getindex.(sol.u,1)
        ϕ = getindex.(sol.u,2) * sqrt(Z_nl)
    
        
        q = ϕ_dot/ω_nl / sqrt(Z_nl)
        return(ϕ,q,sol.t)
    end
    sol = solve(prob, VelocityVerlet(),dt=dt)
    ϕ_dot = [getindex.(sol.u,i) for i in 1:n+1]
    ϕ = [getindex.(sol.u,i+n+1) for i in 1:n+1]

    ϕ_sys = ϕ[1] * sqrt(Z_nl)
    inv_Mq_bis = inv(Mq_bis)
    q = [inv_Mq_bis*getindex.(ϕ_dot,t) for t in 1:length(sol.t)]
    q = [getindex.(q,i) for i in 1:n+1]
    q_sys = q[1] / sqrt(Z_nl)

    if energy
        E=q_sys.^2 * C_inv /2 - E_J*cos.(ϕ_sys/ϕ_quantum) - system_chain_coupling * ϕ_sys .* ϕ[2] + 1/2*[getindex.(q,i)[2:end]'*Mq_chain*getindex.(q,i)[2:end] + getindex.(ϕ,i)[2:end]'*Mϕ_chain*getindex.(ϕ,i)[2:end] for i in 1:length(sol.t)];
        return(ϕ_sys, q_sys, sol.t, E)
    end
    return(ϕ_sys, q_sys, sol.t)
end

function CL_non_linear_driven(t_max;E_J=90 *2pi /3, κ=1/20, C_inv=0.147*2*pi, ω=sqrt(4*C_inv*E_J), A=5)
    """
    INPUT :
    E_J : Josephson energy in GHz
    κ : damping rate in GHz
    C_inv : inverse capacitance in GHz
    ω : frequency of the driving force in GHz (resonant by default : ω_nl = sqrt(4*C_inv*E_J))
    A : amplitude of the driving force (default: 5)

    OUTPUT :
    ϕ_CL : array of the time evolution of the phase ϕ(t)
    q_CL : array of the time evolution of the charge q(t)
    t : time array
    """
    # Definition of the second order ODE
    function damped_non_linear_driven(ddu, du, u, p, t)
        ddu[1] = -2 * κ * du[1] - E_J * C_inv *sin(u[1] / ϕ_quantum) / ϕ_quantum + A * cos(ω * t)
        #ddu[1] = -2 * κ * du[1] - E_J * C_inv *u[1] / ϕ_quantum / ϕ_quantum + A * cos(ω * t)
    end

    
    ϕ_quantum = 1/2 # ħ/e 
    ω_nl = sqrt(4*C_inv*E_J)
    Z_nl=C_inv/ω_nl


    # Initial conditions
    ϕ_0 = [1.]
    q_0 = [1.]
    ϕ_dot_0 = q_0 * C_inv

    tspan = (0.0, t_max) # Time span for the simulation
    dt = 1 / (60 * ω_nl) # Time step for the simulation

    prob = SecondOrderODEProblem(damped_non_linear_driven,ϕ_dot_0, ϕ_0, tspan)
    sol = solve(prob, VelocityVerlet(), dt=dt)

    # Extract solution arrays
    ϕ_dot_CL = [u[1] for u in sol.u]
    ϕ_CL = [u[2] for u in sol.u]
    q_CL = ϕ_dot_CL / C_inv

    return(ϕ_CL, q_CL, sol.t)
end

## Two-photon process solver
"""
INPUT
n : numer of sites on the chain
bath : type of bath modeled (Ohmic_bath, Exp_decay, Drude_lorentz)
bath_params : parameters for the bath
t_max : maximum time for the simulation
circuit_params : parameters of the circuit
init_mode_a : initial parameter of the flux and the charge for the mode a
init_mode_b : initial parameter of the flux and the charge for the mode b
β : inverse temperature
minimal_a : boolean if you want only the flux and charge of the mode a
emission : boolean if you need to compute the emission spectrum

OUTPUT
ϕ_a : flux for the mode a
ϕ_b : flux for the mode b
Φ : flux for the chain
q_a : charge for the mode a
q_b : charge for the mode b
Q : charge for the chain
t_list : time vector for the simulation
Mq_chain : matrix for the chain Hamiltonian for the charge variable if emission=true
Mϕ_chain : matrix for the chain Hamiltonian for the flux variable if emission=true
"""
function two_photon_1d_chain(n, t_max; chain_params=nothing, bath=nothing, bath_params=nothing,
     circuit_params=(ω_a=4*2*pi,ω_b=7.05*2*pi,E_J = 37*2*pi, E_Ca=0.0121, E_Cb=0.0705, ϵ_0 = 0.6*pi,κ=1/10,ϵ_d=0.24,ω_d=7.0*2*pi),
     init_mode_a=(ϕ_a_0=0.0,q_a_0 = 0.0), init_mode_b=(ϕ_b_0=0.0,q_b_0 = 0.0), β=∞,cat_phase=0.0,minimal_a=false,emission=false, fast_no_chain=true)



    # Intialization of the circuit parameters
    ω_a, ω_b, E_J, E_Ca, E_Cb, ϵ_0, κ, ϵ_d, ω_d = circuit_params
    C_a_inv=2*E_Ca # hbar/e^2 GHz
    C_b_inv=2*E_Cb # hbar/e^2 GHz
    Z_a=C_a_inv/ω_a # Impedance of the resonator a
    Z_b=C_b_inv/ω_b # Impedance of the resonator b
    

    ϕ_quantum=1/2 #flux quantum in hbar/e


    # Initialization of the bath parameters
    if chain_params===nothing
        AB, J = bath(bath_params=bath_params,κ=κ, C_inv=C_b_inv)
        onsite_energies,hopping_energies,system_chain_coupling = chaincoeffs_finiteT(n, β, false; J=J, mc=size(AB)[1], mp=0, AB=AB, iq=1, idelta=2, procedure=:Lanczos, Mmax=5000, save=false)  # chain parameters, i.e. on-site energies ϵ_i, hopping energies t_i, and system-chain coupling c_0
    else
        onsite_energies, hopping_energies, system_chain_coupling = chain_params
    end

    function drive(t)
        return ϵ_d * cos(ω_d * t + cat_phase) #We consider a drive resonant with the mode b
    end

    ω_p=2ω_a-ω_d
    function epsilon(t)
        return ϵ_0 * cos(ω_p * t) # We consider a drive resonant with the two-photon process
    end


    temp_buf = zeros(n)
    function two_photon_ODE(ddu,du,u,p,t)
        ϕ_a = u[1]
        ϕ_b = u[2]
        Φ=u[3:end]

        
        ddu[1] = -ω_a^2*ϕ_a + 2*E_J*ω_a * sqrt(Z_a) * sin(epsilon(t))*(cos(ϕ_a * sqrt(Z_a)/ϕ_quantum + ϕ_b * sqrt(Z_b)/ϕ_quantum) - 1) / ϕ_quantum
        ddu[2] = -ω_b^2*ϕ_b + ω_b * system_chain_coupling * sqrt(Z_b) * Φ[1] - ω_b *drive(t)*sqrt(2) + 2*E_J*ω_b * sqrt(Z_b) * sin(epsilon(t))*(cos(ϕ_a * sqrt(Z_a)/ϕ_quantum + ϕ_b * sqrt(Z_b)/ϕ_quantum) - 1) / ϕ_quantum
        mul!(temp_buf, Mϕ_chain, Φ)
        @. temp_buf = -temp_buf
        temp_buf[1] += system_chain_coupling * sqrt(Z_b) * ϕ_b
        mul!(view(ddu, 3:length(ddu)), Mq_chain, temp_buf)
        return nothing
    end


    gq=vcat(- hopping_energies)
    gϕ=vcat(hopping_energies)

    Mq_chain=SymTridiagonal(onsite_energies, gq)
    Mϕ_chain=SymTridiagonal(onsite_energies, gϕ)

    Mq_full=SymTridiagonal(vcat([ω_a,ω_b], onsite_energies), vcat([0,0],gq))

    #Initial conditions
    ϕ_a_0,q_a_0 = init_mode_a

    ϕ_b_0,q_b_0 = init_mode_b


    ϕ_0 = vcat(ϕ_a_0/sqrt(Z_a),ϕ_b_0/sqrt(Z_b), fill(0.0, n))
    q_0 = vcat(q_a_0 * sqrt(Z_a),q_b_0 * sqrt(Z_b), fill(0.0, n))
    ϕ_dot_0 = Mq_full*q_0


    #Run the simulation
    tspan = (0.0, t_max) # Time span for the simulation
    dt = 1 / (5 * maximum(vcat(ω_p,ω_a,ω_b,onsite_energies))) # Time step for the simulation
    prob = SecondOrderODEProblem(two_photon_ODE, ϕ_dot_0, ϕ_0, tspan)

    if emission
        save_idxs=vcat(3:n+2,n+5:n+4+n) # We save the chain DoFs (Φ)
        sol = solve(prob, VelocityVerlet(),dt=dt,
        saveat=t_max,                      # how often to save (or choose a coarser step)
        save_idxs=save_idxs,            # only store system DoFs
        save_everystep=false            # do not store intermediate solver steps
        )
        ϕ_dot = [getindex.(sol.u,i)[end] for i in 1:n]
        ϕ = [getindex.(sol.u,i+n)[end] for i in 1:n]
        return(ϕ,ϕ_dot,sol.t,Mq_chain, Mϕ_chain)
    
    end

    if fast_no_chain
        save_idxs=[1,2,n+3,n+4] # We only save the system DoFs (ϕ_a, ϕ_b) and not the chain DoFs (Φ)
        sol = solve(prob, VelocityVerlet(),dt=dt,
        saveat=dt,                      # how often to save (or choose a coarser step)
        save_idxs=save_idxs,            # only store system DoFs
        save_everystep=false            # do not store intermediate solver steps
        )

        ϕ_dot = [getindex.(sol.u,i) for i in 1:2]
        ϕ = [getindex.(sol.u,i+2) for i in 1:2]
    
        ϕ_a = ϕ[1] * sqrt(Z_a)
        q_a = ϕ_dot[1]/ω_a / sqrt(Z_a)
        if minimal_a
            return(ϕ_a,q_a,sol.t)
        end

        ϕ_b = ϕ[2] * sqrt(Z_b)
        q_b = ϕ_dot[2]/ω_b / sqrt(Z_b)

        t_list = sol.t
        return(ϕ_a,ϕ_b,q_a,q_b,t_list)
    end
    
    sol = solve(prob, VelocityVerlet(),dt=dt)
    
    ϕ_dot = [getindex.(sol.u,i) for i in 1:n+2]
    ϕ = [getindex.(sol.u,i+n+2) for i in 1:n+2]
    ϕ_a = ϕ[1] * sqrt(Z_a)
    ϕ_b = ϕ[2] * sqrt(Z_b)

    fac = lu(Mq_full)
    q = [fac \ getindex.(ϕ_dot,t) for t in 1:length(sol.t)]
    q = [getindex.(q,i) for i in 1:n+2]
    q_a = q[1] / sqrt(Z_a)
    q_b = q[2] / sqrt(Z_b)
    if minimal_a
        return(ϕ_a,q_a,sol.t)
    end

    t_list = sol.t
    return(ϕ_a,ϕ_b,ϕ[3:end],q_a,q_b,q[3:end],t_list)
end


"""
INPUT
n_a : numer of sites on the chain of the mode a
n_b : numer of sites on the chain of the mode b
chain_params_a : parameters of the chain for the mode a (optional)
chain_params_b : parameters of the chain for the mode b (optional)
bath_a : type of bath modeled (Ohmic_bath, Exp_decay, Drude_lorentz) for the mode a (optional)
bath_b : type of bath modeled (Ohmic_bath, Exp_decay, Drude_lorentz) for the mode b (optional)
bath_params_a : parameters for the bath of the mode a (optional)
bath_params_b : parameters for the bath of the mode b (optional)
t_max : maximum time for the simulation
circuit_params : parameters of the circuit
init_mode_a : initial parameter of the flux and the charge for the mode a
init_mode_b : initial parameter of the flux and the charge for the mode b
β : inverse temperature
minimal_a : boolean if you want only the flux and charge of the mode a

OUTPUT
ϕ_a : flux for the mode a
ϕ_b : flux for the mode b
Φ : flux for the chain
q_a : charge for the mode a
q_b : charge for the mode b
Q : charge for the chain
t_list : time vector for the simulation
Mq_chain : matrix for the chain Hamiltonian for the charge variable if emission=true
Mϕ_chain : matrix for the chain Hamiltonian for the flux variable if emission=true
"""
function two_plus_one_photon_chain(n_a, n_b, t_max; chain_params_a=nothing,chain_params_b=nothing, bath_a=nothing, bath_b=nothing, bath_params_a=nothing, bath_params_b=nothing,
     circuit_params=(ω_a=4*2*pi,ω_b=7.05*2*pi,E_J = 37*2*pi, E_Ca=0.0121, E_Cb=0.0705, ϵ_0 = 0.6*pi,κ_a= 0.01284,κ_b=1/10,ϵ_d=0.24,ω_d=7.0*2*pi),
     init_mode_a=(ϕ_a_0=0.0,q_a_0 = 0.0), init_mode_b=(ϕ_b_0=0.0,q_b_0 = 0.0), β=∞,minimal_a=false, fast_no_chain=true)


    # Intialization of the circuit parameters
    ω_a, ω_b, E_J, E_Ca, E_Cb, ϵ_0, κ_a, κ_b, ϵ_d, ω_d = circuit_params
    C_a_inv=2*E_Ca # hbar/e^2 GHz
    C_b_inv=2*E_Cb # hbar/e^2 GHz
    Z_a=C_a_inv/ω_a # Impedance of the resonator a
    Z_b=C_b_inv/ω_b # Impedance of the resonator b
    

    ϕ_quantum=1/2 #flux quantum in hbar/e


    # Initialization of the bath parameters for the mode a
    if chain_params_a===nothing
        AB, J = bath(bath_params=bath_params_a,κ=κ_a, C_inv=C_a_inv)
        onsite_energies_a,hopping_energies_a,system_chain_coupling_a = chaincoeffs_finiteT(n_a, β, false; J=J, mc=size(AB)[1], mp=0, AB=AB, iq=1, idelta=2, procedure=:Lanczos, Mmax=5000, save=false)  # chain parameters, i.e. on-site energies ϵ_i, hopping energies t_i, and system-chain coupling c_0
    else
        onsite_energies_a, hopping_energies_a, system_chain_coupling_a = chain_params_a
    end

    # Initialization of the bath parameters for the mode b
    if chain_params_b===nothing
        AB, J = bath(bath_params=bath_params_b,κ=κ_b, C_inv=C_b_inv)
        onsite_energies_b,hopping_energies_b,system_chain_coupling_b = chaincoeffs_finiteT(n_b, β, false; J=J, mc=size(AB)[1], mp=0, AB=AB, iq=1, idelta=2, procedure=:Lanczos, Mmax=5000, save=false)  # chain parameters, i.e. on-site energies ϵ_i, hopping energies t_i, and system-chain coupling c_0
    else
        onsite_energies_b, hopping_energies_b, system_chain_coupling_b = chain_params_b
    end

    function drive(t)
        return ϵ_d * cos(ω_d * t) #We consider a drive resonant with the mode b
    end

    ω_p=2ω_a-ω_d
    function epsilon(t)
        return ϵ_0 * cos(ω_p * t) # We consider a drive resonant with the two-photon process
    end


    temp_buf_a = zeros(n_a)
    temp_buf_b = zeros(n_b)
    function two_photon_ODE(ddu,du,u,p,t)
        ϕ_a = u[1]
        ϕ_b = u[2]
        Φ_a=u[3:n_a+2]
        Φ_b=u[n_a+3:end]

        ddu[1] = -ω_a^2*ϕ_a + 2*E_J*ω_a * sqrt(Z_a) * sin(epsilon(t))*(cos(ϕ_a * sqrt(Z_a)/ϕ_quantum + ϕ_b * sqrt(Z_b)/ϕ_quantum) - 1) / ϕ_quantum + ω_a * system_chain_coupling_a * sqrt(Z_a) * Φ_a[1]
        ddu[2] = -ω_b^2*ϕ_b + ω_b * system_chain_coupling_b * sqrt(Z_b) * Φ_b[1] - ω_b *drive(t)*sqrt(2) + 2*E_J*ω_b * sqrt(Z_b) * sin(epsilon(t))*(cos(ϕ_a * sqrt(Z_a)/ϕ_quantum + ϕ_b * sqrt(Z_b)/ϕ_quantum) - 1) / ϕ_quantum
        mul!(temp_buf_a, Mϕ_chain_a, Φ_a)
        @. temp_buf_a = -temp_buf_a
        temp_buf_a[1] += system_chain_coupling_a * sqrt(Z_a) * ϕ_a
        mul!(view(ddu, 3:n_a+2), Mq_chain_a, temp_buf_a)

        mul!(temp_buf_b, Mϕ_chain_b, Φ_b)
        @. temp_buf_b = -temp_buf_b
        temp_buf_b[1] += system_chain_coupling_b * sqrt(Z_b) * ϕ_b
        mul!(view(ddu, n_a+3:length(ddu)), Mq_chain_b, temp_buf_b)
        return nothing
    end



    gq_a=vcat(- hopping_energies_a)
    gϕ_a=vcat(hopping_energies_a)

    Mq_chain_a=SymTridiagonal(onsite_energies_a, gq_a)
    Mϕ_chain_a=SymTridiagonal(onsite_energies_a, gϕ_a)

    gq_b=vcat(- hopping_energies_b)
    gϕ_b=vcat(hopping_energies_b)
    Mq_chain_b=SymTridiagonal(onsite_energies_b, gq_b)
    Mϕ_chain_b=SymTridiagonal(onsite_energies_b, gϕ_b)

    Mq_full=SymTridiagonal(vcat([ω_a,ω_b], onsite_energies_a, onsite_energies_b), vcat([0,0],gq_a,[0] ,gq_b))

    #Initial conditions
    ϕ_a_0,q_a_0 = init_mode_a

    ϕ_b_0,q_b_0 = init_mode_b


    ϕ_0 = vcat(ϕ_a_0/sqrt(Z_a),ϕ_b_0/sqrt(Z_b), fill(0.0, n_a+n_b))
    q_0 = vcat(q_a_0 * sqrt(Z_a),q_b_0 * sqrt(Z_b), fill(0.0, n_a+n_b))
    ϕ_dot_0 = Mq_full*q_0


    #Run the simulation
    tspan = (0.0, t_max) # Time span for the simulation
    dt = 1 / (5 * maximum(vcat(ω_p,ω_a,ω_b,onsite_energies_a,onsite_energies_b))) # Time step for the simulation
    prob = SecondOrderODEProblem(two_photon_ODE, ϕ_dot_0, ϕ_0, tspan)
    if fast_no_chain
        save_idxs=[1,2,n_a+n_b+3,n_a+n_b+4] # We only save the system DoFs (ϕ_a, ϕ_b) and not the chain DoFs (Φ)
        sol = solve(prob, VelocityVerlet(),dt=dt,
        saveat=dt,                      # how often to save (or choose a coarser step)
        save_idxs=save_idxs,            # only store system DoFs
        save_everystep=false            # do not store intermediate solver steps
        )

        ϕ_dot = [getindex.(sol.u,i) for i in 1:2]
        ϕ = [getindex.(sol.u,i+2) for i in 1:2]
    
        ϕ_a = ϕ[1] * sqrt(Z_a)
        q_a = ϕ_dot[1]/ω_a / sqrt(Z_a)
        if minimal_a
            return(ϕ_a,q_a,sol.t)
        end

        ϕ_b = ϕ[2] * sqrt(Z_b)
        q_b = ϕ_dot[2]/ω_b / sqrt(Z_b)

        t_list = sol.t
        return(ϕ_a,ϕ_b,q_a,q_b,t_list)
    end
    sol = solve(prob, VelocityVerlet(),dt=dt)

    ϕ_dot = [getindex.(sol.u,i) for i in 1:n_a+n_b+2]
    ϕ = [getindex.(sol.u,i+n_a+n_b+2) for i in 1:n_a+n_b+2]

    ϕ_a = ϕ[1] * sqrt(Z_a)
    ϕ_b = ϕ[2] * sqrt(Z_b)

    fac = lu(Mq_full)
    q = [fac \ getindex.(ϕ_dot,t) for t in 1:length(sol.t)]
    q = [getindex.(q,i) for i in 1:n_a+n_b+2]
    q_a = q[1] / sqrt(Z_a)
    q_b = q[2] / sqrt(Z_b)

    if minimal_a
        return(ϕ_a,q_a,sol.t)
    end


    t_list = sol.t
    return(ϕ_a,ϕ_b,ϕ[3:end],q_a,q_b,q[3:end],t_list)
end

"""
INPUT
n : numer of sites on the chain
bath : type of bath modeled (Ohmic_bath, Exp_decay, Drude_lorentz)
bath_params : parameters for the bath
t_max : maximum time for the simulation
circuit_params : parameters of the circuit
init_mode_a : initial parameter of the flux and the charge for the mode a
β : inverse temperature
minimal_a : boolean if you want only the flux and charge of the mode a

OUTPUT
ϕ_a : flux for the mode a
ϕ_b : flux for the mode b
q_a : charge for the mode a
q_b : charge for the mode b
sol.t : time vector for the simulation
"""
function two_photon_CL(t_max;
        circuit_params=(ω_a=4*2*pi, ω_b=7.0*2*pi, E_J = 37*2*pi, E_Ca=0.0121, E_Cb=0.0705, ϵ_0 = 0.6*pi, κ_a=0, κ_b=1/10, ϵ_d=0.24, ω_d=ω_b),
        init_mode_a=(ϕ_a_0=0.0,q_a_0 = 0.0), init_mode_b=(ϕ_b_0=0.0,q_b_0 = 0.0), β=∞, cat_phase=0.0)

    
    ω_a,ω_b, E_J, E_Ca, E_Cb, ϵ_0, κ_a, κ_b, ϵ_d,ω_d = circuit_params
    C_a_inv=2*E_Ca # hbar/e^2 GHz
    C_b_inv=2*E_Cb # hbar/e^2 GHz
    Z_a=C_a_inv/ω_a # Impedance of the resonator a
    Z_b=C_b_inv/ω_b # Impedance of the resonator b

    ϕ_quantum = 1/2 # ħ/e 

    function drive(t)
        return ϵ_d * cos(ω_d * t+ cat_phase) #We consider a drive resonant with the mode b
    end

    ω_p=2ω_a-ω_d
    function epsilon(t)
        return ϵ_0 * cos(ω_p * t) # We consider a drive resonant with the two-photon process
    end

    # Definition of the second order ODE
    function damped_two_photon_2(ddu, du, u, p, t)
        ϕ_a = u[1]
        ϕ_b = u[2]
        
        ddu[1] = -ω_a^2*ϕ_a + 2*E_J*ω_a * sqrt(Z_a) * sin(epsilon(t))*(cos(ϕ_a * sqrt(Z_a)/ϕ_quantum + ϕ_b * sqrt(Z_b)/ϕ_quantum) - 1) / ϕ_quantum  - 2 * κ_a * du[1]
        ddu[2] = -ω_b^2*ϕ_b - ω_b *drive(t)*sqrt(2) - 2 * κ_b * du[2] + 2*E_J*ω_b * sqrt(Z_b) * sin(epsilon(t))*(cos(ϕ_a * sqrt(Z_a)/ϕ_quantum + ϕ_b * sqrt(Z_b)/ϕ_quantum) - 1) / ϕ_quantum 
    end

    # Initial conditions
    ϕ_a_0, q_a_0 = init_mode_a

    ϕ_b_0, q_b_0 = init_mode_b
    
    ϕ_0 = vcat(ϕ_a_0/sqrt(Z_a),ϕ_b_0/sqrt(Z_b))
    ϕ_dot_0 = vcat(q_a_0 * sqrt(Z_a)*ω_a,q_b_0 * sqrt(Z_b)*ω_b)

    tspan = (0.0, t_max) # Time span for the simulation
    dt = 1 / (50 * ω_a) # Time step for the simulation
    print(join([ω_a, ω_b, E_J, E_Ca, E_Cb, κ_a, κ_b, Z_a, Z_b, ϕ_quantum, ϵ_d, ϵ_0, ω_d, ω_p]," "))
    prob = SecondOrderODEProblem(damped_two_photon_2,ϕ_dot_0, ϕ_0, tspan)
    sol = solve(prob, VelocityVerlet(), dt=dt)

    # Extract solution arrays
    ϕ_dot_CL_a = [u[1] for u in sol.u]*sqrt(Z_a)
    ϕ_CL_a = [u[3] for u in sol.u]*sqrt(Z_a)
    q_CL_a = ϕ_dot_CL_a / C_a_inv

    ϕ_dot_CL_b = [u[2] for u in sol.u]*sqrt(Z_b)
    ϕ_CL_b = [u[4] for u in sol.u]*sqrt(Z_b)
    q_CL_b = ϕ_dot_CL_b / C_b_inv

    return ϕ_CL_a, ϕ_CL_b, q_CL_a, q_CL_b, sol.t
end
#########################
# Reverse chain mapping #
#########################
"""
INPUT :
ϕ_chain : vector of the field ϕ in the chain. Each vector is an oscillator and contains its dynamics for each time step.
ϕ_dot_chain : vector of the time derivative of the field ϕ in the chain. Each vector is an oscillator and contains its dynamics for each time step.
Mϕ_chain : matrix of the field ϕ in the chain
Mq_chain : matrix of the field q in the chain

OUTPUT :
ωk_chain : vector of the normal mode frequencies in the chain
Energies_normal_modes_chain : matrix of the energy of the normal modes in the chain
"""
function emission_spectrum(ϕ_chain, ϕ_dot_chain, Mϕ_chain, Mq_chain)

    #Transforming the vectors into matrices
    ϕ_chain = reduce(hcat, ϕ_chain) # Remove the first and second elements which are the system field
    ϕ_dot_chain = reduce(hcat, ϕ_dot_chain) # Remove the first and second elements which are the system field

    #Diagonalization
    inv_Mq_chain= inv(Mq_chain)

    A_chain=Mq_chain * Mϕ_chain
    vals_chain, vecs_chain = eigen(A_chain)

    ωk²_chain = real(vals_chain)
    ωk_chain = sqrt.(ωk²_chain)
    P_chain = real(vecs_chain)
    P_chain_inv=inv(P_chain)

    Dq_chain= diag(P_chain' * inv_Mq_chain * P_chain)
    Dϕ_chain= diag(P_chain' * Mϕ_chain * P_chain)

    Energies_normal_modes_chain = Matrix{Float64}(undef,size(ϕ_chain))

    # Compute S_chain and S_chain_dot on the fly, avoid storing large intermediates
    for t in 1:size(ϕ_chain, 1)
        S = P_chain_inv * ϕ_chain[t,:]
        S_dot = P_chain_inv * ϕ_dot_chain[t,:]
        Energies_normal_modes_chain[t,:] = Dq_chain .* S_dot.^2 .+ Dϕ_chain .* S.^2
    end
    return(ωk_chain,Energies_normal_modes_chain)
end
;