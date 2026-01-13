import numpy as np
import scipy.signal as signal
import sys

def design_fir_lowpass(fS, fL, N):
    #Compute sinc filter.
    h = np.sinc(2 * fL / fS * (np.arange(N)-(N-1)/2))

    #optionally apply window
    h *= np.hamming(N)
    # h *= np.blackman(N)
    # beta = 4.0
    # h *= np.kaiser(N,beta)

    #normalize for unity gain
    h /= np.sum(h)

    #print(f'{{{", ".join(map(str,h))}}}')
    print(f'{{{", ".join(f"{x:.20f}" for x in h)}}}')

    w, H = signal.freqz(h, worN = fS)
    f = w * fS / (2 * np.pi)

    return h, f, H



# ===============================
# SIGNAL GENERATOR
# ===============================

def generate_test_signal(fS, duration):
    """
    Generates a signal with:
      - 50 kHz  (should pass)
      - 200 kHz (should be attenuated)
      - noise
    """
    t = np.arange(0, duration, 1/fS)

    x  = 1.0 * np.sin(2*np.pi*50_000*t)    # in-band
    x += 0.8 * np.sin(2*np.pi*200_000*t)   # out-of-band
    x += 0.2 * np.random.randn(len(t))    # noise

    return t, x


# ===============================
# FIR FILTER (time-domain MAC)
# ===============================

def fir_filter(x, h):
    """
    Time-domain FIR filter (like HDL MAC chain)
    """
    N = len(h)
    y = np.zeros(len(x))

    for n in range(len(x)):
        acc = 0.0
        for k in range(N):
            if n-k >= 0:
                acc += h[k] * x[n-k]
        y[n] = acc

    return y


# ===============================
# FFT utility
# ===============================

def compute_fft(x, fS):
    X = np.fft.fft(x)
    f = np.fft.fftfreq(len(x), 1/fS)
    return f, np.abs(X)



# Print coefficients in your Verilog-friendly format
# print("{", ", ".join(f"{x:.20f}" for x in h), "}")