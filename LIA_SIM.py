import numpy as np
import scipy.signal as signal
from LIA_Test import *
import matplotlib.pyplot as plt


fS = 1_000_000
fL = 110_000
N  = 63

# Design FIR
h, f, H = design_fir_lowpass(fS, fL, N)
print(len(h))

# Generate test signal
t, x = generate_test_signal(fS, duration=0.002)   # 2ms. Check this function implementation. Write 2 functions to test LIA.

# Filter it
y = fir_filter(x, h)

# FFTs
fx, X = compute_fft(x, fS)
fy, Y = compute_fft(y, fS)


# ===============================
# PLOTS
# ===============================

plt.figure(figsize=(12,8)) 

# Time domain
plt.subplot(2,1,1)
plt.plot(t[:2000], x[:2000], label="Input")
plt.plot(t[:2000], y[:2000], label="Filtered")
plt.title("Time Domain")
plt.legend()

# Frequency domain
plt.subplot(2,1,2)
plt.plot(fx, X, label="Input spectrum")
plt.plot(fy, Y, label="Filtered spectrum")
plt.xlim(0, 500_000)
plt.title("Frequency Domain")
plt.legend()

plt.tight_layout()
plt.show()