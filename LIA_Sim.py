import numpy as np
import scipy.signal as signal
from LIA_Test import *
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift
import math


fS = 3_000_000  #System Sampling Rate

#underscores are visual aid
# l seperators
Runtime = 100/1000000 # s/1000000 for conversion to us. This also affects FIR plot. Peaks look bigger with longer runtime.

# Generate test signals
f1 = 80000
f2 = 250_000
fref = 80_000
t_in, x_in = generate_test_signal_add(f1,f2, duration=Runtime,fS=fS) #LIA_Input
#t, x = generate_test_signal_mult(f1,f2, duration=Runtime,fS=fS)

#Mixing for both Quadratures
Tx, X = mult_ref(x_in,fref,duration=Runtime, fS=fS, ref = 'sin') #X quad
Ty, Y = mult_ref(x_in,fref,duration=Runtime, fS=fS, ref = 'cos') #Y quad

# FIR X-Quadrature
fL = 120000    #Filter cutoff. Note that filter efficiency works on fL/fS.
N  = 63        #Filter taps for steeper cutoff slope

h = design_fir_lowpass(fS, fL, N)

# print("{", ", ".join(f"{x:.20f}" for x in h), "}")  #prints coefficients to copy to Vivado
# plot_LPF_freq_response(1500000,fS,h)

# Apply the Filter
x = fir_filter(X, h)
y = fir_filter(Y, h)

z = x**2+y**2

R = [math.sqrt(i) for i in z]

#print(R)


# # ===============================
# # PLOTS
# # ===============================



# # FFTs
T = 1/(fS)
Nx = len(x)
Fx = fft(x)
freqX = fftfreq(Nx,T)
#print(freqX[:Nx//2])

# Ny = len(y)
# Fy = fft(y)
# freqY = fftfreq(Ny,T)
# yplot = fftshift(Fy)
# plt.figure(figsize=(12,8)) 

### Time domain FIR
plt.subplot(2,1,1)
plt.plot(Tx, R, label="Input")
#plt.plot(Ty, Y, label="Filtered")
plt.title("Time Domain")
plt.legend()

# ### Frequency domain FIR
plt.subplot(2,1,2)
plt.plot(freqX[:Nx//2], 1/Nx*np.abs(Fx)[:Nx//2], label="Input spectrum")
#plt.plot(freqX, 1/Ny*np.abs(Fy), label="Filtered spectrum")
plt.xlim(0, 500_000)
plt.title("Frequency Domain")
plt.legend()

plt.tight_layout()
plt.show()



