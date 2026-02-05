import numpy as np
import scipy.signal as signal
from LIA_Test import *
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift
import math

fS = 5_000_000
#underscores are visual seperators
Runtime = 100/1000000 # s/1000000 for conversion to us. This also affects FIR plot. Peaks look bigger with longer runtime.

# Generate test signal
f1 = 100_000
f2 = 412_000
fref= 100_000
tin, xin = generate_test_signal_add(f1,f2, duration=Runtime,fS=fS,Ampl1 = 1) 
#tin, xin = generate_test_signal_mult(f1,f2, duration=Runtime,fS=fS,Ampl1 = 2)

#print(t,x)
#print(len(x))
 #Input Signal Samples. The more samples there are the filter works better too
fL = 110_000    #cutoff
N  = 254          #taps


Tx, X = mult_ref(xin,fref,duration=Runtime,fS=fS,ref = 'cos')
Ty, Y = mult_ref(xin,fref,duration=Runtime,fS=fS,ref = 'sin')

# Design FIR
h = design_fir_lowpass(fS, fL, N)
#print(h)``
print("{", ", ".join(f"{x:.20f}" for x in h), "}")
plot_LPF_freq_response(1500000,fS,h)
#Find a way to convert signed fixed point for simulation.





# # Filter it
x = fir_filter(X, h)
y = fir_filter(Y, h)
# plt.plot(t, y)
# plt.xlabel("t")
# plt.ylabel("x")
# plt.title("x vs t")
# plt.grid(True)
# plt.show()


R = np.sqrt(x**2+y**2)
theta = np.atan2(x,y)
#print(theta)

# FFTs
T = 1/(fS)
Nx = len(xin)
Fx = fft(xin)
freqX = fftfreq(Nx,T)


Ny = len(x)
Fy = fft(x)
freqY = fftfreq(Ny,T)




# # ===============================
# # PLOTS
# # ===============================

plt.figure(figsize=(12,8)) 

### Time domain
plt.subplot(2,1,1)
plt.plot(tin, xin, label="Input")
plt.plot(tin, y, label="Filtered")
plt.title("Time Domain")
plt.legend()

### Frequency domain
plt.subplot(2,1,2)
plt.plot(freqX, 1/Nx*np.abs(Fx), label="Input spectrum")
plt.plot(freqY, 1/Ny*np.abs(Fy), label="Filtered spectrum")
plt.xlim(0, 600_000)
plt.title("Frequency Domain")
plt.legend()

plt.tight_layout()
plt.show()