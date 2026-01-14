import numpy as np
import scipy.signal as signal
from LIA_Test import *
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift

#underscores are visual seperators
Runtime = 100/1000000 # s/1000000 for conversion to us. This also affects FIR plot. Peaks look bigger with longer runtime.

# Generate test signal
f1 = 100_000
f2 = 300_000
#t, x = generate_test_signal_add(100_000,300_000, duration=Runtime) 
t, x = generate_test_signal_mult(f1,f2, duration=Runtime)

#print(t,x)
#print(len(x))

fS = 10_000_000  #Input Signal Samples. The more samples there are the filter works better too
fL = 210_000    #cutoff
N  = 63        #taps


# Design FIR
h = design_fir_lowpass(fS, fL, N)
#print(h)
print("{", ", ".join(f"{x:.20f}" for x in h), "}")
plot_LPF_freq_response(1500000,fS,h)



# # Filter it
y = fir_filter(x, h)
# plt.plot(t, y)
# plt.xlabel("t")
# plt.ylabel("x")
# plt.title("x vs t")
# plt.grid(True)
# plt.show()


# FFTs
T = 1/(30*max(f1,f2))
Nx = len(x)
Fx = fft(x)
freqX = fftfreq(Nx,T)


Ny = len(y)
Fy = fft(y)
freqY = fftfreq(Ny,T)
yplot = fftshift(Fy)


# # ===============================
# # PLOTS
# # ===============================

plt.figure(figsize=(12,8)) 

### Time domain
plt.subplot(2,1,1)
plt.plot(t, x, label="Input")
plt.plot(t, y, label="Filtered")
plt.title("Time Domain")
plt.legend()

### Frequency domain
plt.subplot(2,1,2)
plt.plot(freqX, 1/Nx*np.abs(Fx), label="Input spectrum")
plt.plot(freqX, 1/Ny*np.abs(Fy), label="Filtered spectrum")
plt.xlim(0, 500_000)
plt.title("Frequency Domain")
plt.legend()

plt.tight_layout()
plt.show()