import numpy as np
import scipy.signal as signal
import sys
from scipy.fft import fft, fftfreq, fftshift

# def design_fir_lowpass(fS, fL, N):
#     #Compute sinc filter.
#     h = np.sinc(2 * fL / fS * (np.arange(N)-(N-1)/2))

#     #optionally apply window
#     h *= np.hamming(N)
#     # h *= np.blackman(N)
#     # beta = 4.0
#     # h *= np.kaiser(N,beta)

#     #normalize for unity gain
#     h /= np.sum(h)

#     #print(f'{{{", ".join(map(str,h))}}}')
#     #print(f'{{{", ".join(f"{x:.20f}" for x in h)}}}')

#     w, H = signal.freqz(h, worN = fS) #w = Filter computed frequencies, H = complex frequency response.
#     f = w * fS / (2 * np.pi)

#     return h, f, H

def design_fir_lowpass(fS, fL, N):
    #Compute sinc filter.

    h = signal.firwin(N,fL,width=None,window='hamming',fs=fS)

    return h




# ===============================
# SIGNAL GENERATOR
# ===============================

## Add 2 signals
def generate_test_signal_add(f1,f2,duration,fS):
    """
    Generates a signal with:
        - 50 kHz  (should pass)
        - 200 kHz (should be attenuated)
        - noise
    """
    time_int = 1/(fS) #30 points per oscillation. Based on higher frequency.
    t = np.arange(0, duration, time_int)

    x  = np.sin(2*np.pi*f1*t)    # in-band
    x += np.sin(2*np.pi*f2*t)   # out-of-band
    #x += 0.2 * np.random.randn(len(t))    # noise


    return t, x


def generate_test_signal_mult(f1,f2,duration,fS):
    """
    Generates a signal with:
      - 50 kHz  (should pass)
      - 200 kHz (should be attenuated)
      - noise
    """
    time_int = 1/(fS) #30 points per oscillation. Based on higher frequency.
    t = np.arange(0, duration, time_int)

    x  = np.sin(2*np.pi*f1*t)    # in-band
    x *= np.sin(2*np.pi*f2*t)   # out-of-band
    #x += 0.2 * np.random.randn(len(t))    # noise

    return t, x


# ===============================
# FIR FILTER (time-domain MAC)
# ===============================

def fir_filter(x, h):
    """
    Time-domain FIR filter (like HDL MAC chain)
    """
    a0 = 1
    b = h

    y = signal.lfilter(b,a0,x)
    # N = len(h)
    # y = np.zeros(len(x))

    # for n in range(len(x)):
    #     acc = 0.0
    #     for k in range(N):
    #         if n-k >= 0:
    #             acc += h[k] * x[n-k]
    #     y[n] = acc

    return y


# ===============================
# FFT utility
# ===============================

def compute_fft(x,Runtime,fS):
    T = Runtime/fS
    X = fft(x)
    f = fftfreq(len(x), T)
    f = fftshift(f)
    X = fftshift(X)
    return f, np.abs(X)



# Print coefficients in your Verilog-friendly format
# print("{", ", ".join(f"{x:.20f}" for x in h), "}")