function Shannon_entropy(F::Vector{ComplexF64})
    power_density = abs.(F).^2
    power_density_normalized = power_density ./sum(power_density)
    Shannon_entropy = -sum(power_density_normalized.*log.(power_density_normalized))
    
    return Shannon_entropy
end