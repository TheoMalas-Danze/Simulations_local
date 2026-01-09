using MPSDynamics

ωR = 7.5
κ = 0.1
αchain = κ / (4π*ωR)
ωc = 15.0 
N = 300

 import MPSDynamics: chaincoeffs_ohmic

cpars = chaincoeffs_ohmic(N, αchain, 1; ωc = ωc)

λreorg_ohmic = 2 * αchain * ωc 
# H_cav = ωR a^\dagger a + λreorg_ohmic (a + a^\dagger)^2

using LinearAlgebra

function colpa_diagonalize(A::Matrix{ComplexF64}, B::Matrix{ComplexF64}; tol=1e-10)
    """
    Diagonalizes a bosonic quadratic Hamiltonian according to Colpa's method.
    Arguments:
    - A: (N, N) Hermitian matrix.
    - B: (N, N) Symmetric matrix.
    - tol: Tolerance for small eigenvalues.
    Returns:
    - omega: Vector of positive excitation energies.
    - T: (2N, N) Bogoliubov transformation matrix.
    """
    N = size(A, 1)
    @assert size(A) == (N, N)
    @assert size(B) == (N, N)
    @assert ishermitian(A) "Matrix A must be Hermitian."
    @assert norm(B - transpose(B)) < tol "Matrix B must be symmetric."
    # Build full Hamiltonian matrix
    H = [A B;
         B' transpose(A)]
    # Metric for bosons
    η = [I(N) zeros(N,N);
         zeros(N,N) -I(N)]
    # Solve the generalized eigenvalue problem: H v = ω η v
    eigvals, eigvecs = eigen(H, η)
    # Sort eigenvalues and eigenvectors
    idx = sortperm(map(real,eigvals))
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    # Keep only positive eigenvalues (excitation energies)
    pos_idx = findall(x -> real(x) > tol, eigvals)
    omega = eigvals[pos_idx]
    T = eigvecs[:, pos_idx]
    # Normalize T such that T† η T = η
    norm_matrix = T' * η * T
    for i in 1:length(omega)
        scale = sqrt(abs(norm_matrix[i,i]))
        T[:,i] /= scale
    end
    return omega, T
end
# Example usage
function main()
    # Define example A and B matrices
    A = ComplexF64[2.0 0.5;
                   0.5 3.0]
    B = ComplexF64[0.1 0.2;
                   0.2 0.3]
    omega, T = colpa_diagonalize(A, B)
    println("Excitation energies (omega):")
    println(omega)
    println("\nBogoliubov transformation matrix (T):")
    println(T)
end


function normal_modes_chain()
    # Define example A and B matrices
    ωs=vcat([ωR+2*λreorg_ohmic],cpars[1])
    ts=vcat([cpars[3]],cpars[2])
    A = Array{Complex{Float64},2}(Tridiagonal(ts, ωs, ts));
    B = Array{Complex{Float64},2}(Tridiagonal(vcat([cpars[3]],zeros(N-1)) , vcat(λreorg_ohmic,zeros(N)), vcat([cpars[3]],zeros(N-1))));
    return colpa_diagonalize(A, B)    
end


function normal_modes_chain_RWA()
    # Define example A and B matrices
    ωs=vcat([ωR+2*λreorg_ohmic],cpars[1])
    ts=vcat([cpars[3]],cpars[2])
    A = Array{Complex{Float64},2}(Tridiagonal(ts, ωs, ts));
    B = Array{Complex{Float64},2}(Tridiagonal(vcat([cpars[3]],zeros(N-1)) ,zeros(N+1), vcat([cpars[3]],zeros(N-1))));
    return colpa_diagonalize(A, B)    
end


function normal_modes_chain_dec_resonator()
    # Define example A and B matrices
    ωs=vcat([ωR],cpars[1])
    ts=vcat([0],cpars[2])
    A = Array{Complex{Float64},2}(Tridiagonal(ts, ωs, ts));
    B = Array{Complex{Float64},2}(Tridiagonal(vcat([0],zeros(N-1)) , zeros(N+1), vcat([0],zeros(N-1))));
    return colpa_diagonalize(A, B)    
end


# main()

ωNN, T = normal_modes_chain();


function find_closest_mode(T::Matrix{ComplexF64},ωNN::Array{Float64})
    N = size(T, 1) ÷ 2
    U = T[1:N, :]  # top half of T
    # Look at first row
    overlaps = abs.(U[1, :])
    # Find the index of maximum overlap
    max_idx = argmax(overlaps)
    return max_idx, ωNN[max_idx], overlaps
end

# Usage:
max_idx, overlaps = find_closest_mode(T)
println("Normal mode closest to first bare mode is mode number: ", max_idx)
# println("Overlap amplitudes (|U[1,j]|) are:")
# println(overlaps)

using Plots

plot(overlaps,xlim=[148,152])
