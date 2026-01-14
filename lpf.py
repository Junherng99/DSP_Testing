import numpy as np
import sys
import matplotlib.pyplot as plt
import scipy.signal as signal

if (len(sys.argv)-1) not in (3, ):
    #config
    fS = 1000000 #sampling rate
    fL = 110000 #cutoff frequecy in Hz
    N = 63 #Filter length. Must be odd
else:
    fS = int(sys.argv[1])
    fL = int(sys.argv[2])
    N = int(sys.argv[3])

#Compute sinc filter.
h = np.sinc(2 * fL / fS * (np.arange(N)-(N-1)/2))

#optionally apply window
h *= np.hamming(N)
# h *= np.blackman(N)
# beta = 4.0
# h *= np.kaiser(N,beta)

#normalize for unity gain
h /= np.sum(h)

print(len(h))
#print(f'{{{", ".join(map(str,h))}}}')
print(f'{{{", ".join(f"{x:.20f}" for x in h)}}}')

w, H = signal.freqz(h, worN = fS, fs=fS)

# Plot the frequency response
plt.figure(figsize=(10, 6))
plt.plot(w, 20* np.log10(abs(H)))
plt.xlim(0,1500000)
plt.ylim(-25,0)
plt.title('Filter Frequency Response')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (dB)')
plt.grid(True)
plt.show()

