import numpy as np
import scipy.signal as signal
from LIA_Test import *
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift


fS = 4_000_000  #System Sampling Rate

#underscores are visua
# l seperators
Runtime = 100/1000000 # s/1000000 for conversion to us. This also affects FIR plot. Peaks look bigger with longer runtime.

# Generate test signal
f1 = 100_000
f2 = 300_000
#t, x = generate_test_signal_add(100_000,300_000, duration=Runtime) 
t, x = generate_test_signal_mult(f1,f2, duration=Runtime,fS=fS)

#print(t,x)
#print(len(x))


# Design FIR
fL = 240_000    #Filter cutoff. Note that filter efficiency works on fL/fS.
N  = 63        #Filter taps for steeper cutoff slope

h = design_fir_lowpass(fS, fL, N)

# print("{", ", ".join(f"{x:.20f}" for x in h), "}")  #prints coefficients to copy to Vivado
# plot_LPF_freq_response(1500000,fS,h)

# Apply the Filter
y = fir_filter(x, h)





# # ===============================
# # PLOTS
# # ===============================



# FFTs
# T = 1/(fS)
# Nx = len(x)
# Fx = fft(x)
# freqX = fftfreq(Nx,T)


# Ny = len(y)
# Fy = fft(y)
# freqY = fftfreq(Ny,T)
# yplot = fftshift(Fy)
# plt.figure(figsize=(12,8)) 

# ### Time domain FIR
# plt.subplot(2,1,1)
# plt.plot(t, x, label="Input")
# plt.plot(t, y, label="Filtered")
# plt.title("Time Domain")
# plt.legend()

# ### Frequency domain FIR
# plt.subplot(2,1,2)
# plt.plot(freqX, 1/Nx*np.abs(Fx), label="Input spectrum")
# plt.plot(freqX, 1/Ny*np.abs(Fy), label="Filtered spectrum")
# plt.xlim(0, 500_000)
# plt.title("Frequency Domain")
# plt.legend()

# plt.tight_layout()
# plt.show()



