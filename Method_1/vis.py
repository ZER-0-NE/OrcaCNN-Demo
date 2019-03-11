import aifc
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from matplotlib import mlab, cm
from scipy.stats import skew
import numpy as np
#import pandas as pd

# ReadAIFF function
def ReadAIFF(file):
# Reads the frames from the audio clip and returns the uncompressed data
    s = aifc.open(file,'r')
    nFrames = s.getnframes()
    strSig = s.readframes(nFrames)
    return np.fromstring(strSig, np.short).byteswap()

# Read one file as an example
params = {'NFFT':256, 'Fs':200, 'noverlap':192} 
s = ReadAIFF('train6.aiff')
P, freqs, bins = mlab.specgram(s, **params)

print(P.shape)
print(freqs.shape)
print(bins.shape)
# Spectrogram plotting function long_sample_01.aiff
def plot_spectrogram(ax, P):
    plt.imshow(P, origin='lower', extent=[-6,6,-1,1], aspect=4, cmap = cm.get_cmap('bwr'))
    loc = plticker.MultipleLocator(base=3.0) # this locator puts ticks at regular intervals
    ax.xaxis.set_major_locator(loc)
    ax.set_xticklabels(np.arange(1.0,12.5,0.5))
    ax.set_yticklabels(range(0,10000,250))
    ax.set_xlabel('Time (seconds)', fontsize = 12)
    ax.set_ylabel('Frequency (Hz)', fontsize = 12)
    cbar = plt.colorbar()
    cbar.set_label('Amplitude', fontsize = 12)

fig = plt.figure(figsize = (7,4))
ax1 = plt.subplot(111)
plot_spectrogram(ax1, P)
plt.show()

f1, f2 = 25, 5

# Reassign time bins returned by specgram (convenience so variables match winning submission)
b = bins
# Limit statistics to about 470 Hz.
maxM = 60 
fig = plt.figure(figsize = (14,4))
ax1 = plt.subplot(121)
ax1.plot(b, P[f1,:])
ax1.set_xlabel('Time (seconds)', fontsize = 12)
ax1.set_ylabel('Amplitude', fontsize = 12)
ax2 = plt.subplot(122)
ax2.plot(b, P[f2,:])
ax2.set_xlabel('Time (seconds)', fontsize = 12)
ax2.set_ylabel('Amplitude', fontsize = 12)
plt.suptitle('Comparision of 200 Hz to 40 Hz Frequency Bins', fontsize = 16)
plt.show()

cf_ = [np.sum(P[i,:]*b)/np.sum(P[i,:]) for i in range(maxM)]
centroid = lambda i: np.sum(P[i,:]*b)/np.sum(P[i,:])

print("Peak of whale call at {}s".format(centroid(f1)/10))