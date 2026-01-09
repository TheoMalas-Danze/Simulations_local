from scipy.fft import fft, fftfreq
from jax.numpy import abs

def fft_perso(signal, t_list):
    N = len(signal)
    T = t_list[1] - t_list[0]  # Assuming uniform time steps
    yf = fft(signal)
    xf = fftfreq(N, T)[:N//2]
    return xf, 2.0/N * abs(yf[:N//2])