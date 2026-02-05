import numpy as np
import scipy.signal as signal
from LIA_Test import *
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift
import math

fS = 1_000_000 
#underscores are visual seperators
Runtime = 300/1000000 # s/1000000 for conversion to us. This also affects FIR plot. Peaks look bigger with longer runtime.

#print(len(x))
 #Input Signal Samples. The more samples there are the filter works better too
fL = 140_000    #cutoff
N  = 127        #taps



# Design FIR
h = design_fir_lowpass(fS, fL, N)
#print(h)
print("{", ", ".join(f"{x:.20f}" for x in h), "}")
#plot_LPF_freq_response(1500000,fS,h)

