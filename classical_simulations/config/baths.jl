include("/Users/tmalas/Desktop/Dev/Simulations_local/classical_simulations/utils/smoothstep.jl")

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
    Ω : characteristic frequency of the exponential decay in GHz
    ω_cut_off : cut-off frequency of the spectral density J(ω) in GHz

OUTPUT :
AB : Array of the segments of the frequency ranges where the spectral density J(ω) is defined.
        Each segment is an array with two elements [ω_min ω_max] in GHz.
Jω_fct : spectral density J(ω)
"""
function Exp_decay(;bath_params=(Ω=50*2pi, ω_cut_off=90), κ=1/20, C_inv=0.147*2*pi, plot_bool= false)


    Ω, ω_cut_off = bath_params.Ω, bath_params.ω_cut_off
    ω_up_zero = ω_cut_off + 0.5*2*pi # Upper cut-off of the spectral density J(ω), GHz
    
    ### Caracterization of the impedance #####
    α = 2*κ/C_inv # no unit

    AB=[[-Inf 0];[0 ω_cut_off];[ω_cut_off ω_up_zero];[ω_up_zero Inf]]

    function J_exp_decay(ω,i)
        if i==1 # First segment of AB (here [-Inf 0]), zero for negative frequencies when β == Inf
            y = 0
        elseif i==2 # Second segment of AB (here [0 ω_cut_off])
            y = @.α*ω*exp(-ω/Ω)
        elseif i==3 # Third segment of AB (here [ω_cut_off ω_up_zero])
            y = @. α*ω*exp(-ω_cut_off/Ω)*(1 - smoothstep((ω - ω_cut_off) / (ω_up_zero - ω_cut_off)))
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