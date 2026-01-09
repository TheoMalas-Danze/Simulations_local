using FFTW

function FFT_perso(ϕ,t)
    F = fftshift(fft(ϕ))
    fs = length(t)/(maximum(t)-minimum(t)) # sampling frequency
    freqs = fftshift(fftfreq(length(t), fs))
    return F, freqs
end