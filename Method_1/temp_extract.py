'''
Some of the templates selected are logical: they correspond to the subregion 
of the spectrogram where we find the shape of a whale sound.
'''

import aifc
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from matplotlib import mlab, cm
import numpy as np
import pandas as pd

# ReadAIFF function
def ReadAIFF(file):
# Reads the frames from the audio clip and returns the uncompressed data
    s = aifc.open(file,'r')
    nFrames = s.getnframes()
    strSig = s.readframes(nFrames)
    return np.fromstring(strSig, np.short).byteswap()

# Define vertical sliding window for pre-processing images
def slidingWindowV(P,inner=3,outer=64,maxM = 50,norm=True):
    Q = P.copy()
    m, n = Q.shape
    if norm:
        mval, sval = np.mean(Q[:maxM,:]), np.std(Q[:maxM,:])
        fact_ = 1.5
        Q[Q > mval + fact_*sval] = mval + fact_*sval
        Q[Q < mval - fact_*sval] = mval - fact_*sval
    wInner = np.ones(inner)
    wOuter = np.ones(outer)
    for i in range(n):
        Q[:,i] = Q[:,i] - (np.convolve(Q[:,i],wOuter,'same') - np.convolve(Q[:,i],wInner,'same'))/(outer - inner)
    return Q[:maxM,:]


'''
The following script extracts and plots all the templates, showing 3 images per row:

1. The original image (from where the template is extracted)
2. The image after being pre-processed
3. The template extracted, where the background is blue for whale sounds and 
red for non-whale sounds
'''


# Define a function to reuse plot format
def plot_spectrogram(ax, P, y_label = None):
    plt.imshow(P, origin='lower', extent=[-6,6,-1,1], aspect=4, cmap = cm.get_cmap('bwr'))
    loc = plticker.MultipleLocator(base=3.0)
    ax.xaxis.set_major_locator(loc)
    ax.set_xticklabels(np.arange(-0.5,2.5,0.5))
    ax.set_yticklabels(range(0,1001,250))
    ax.set_xlabel('Time (seconds)', fontsize = 12)
    if y_label:
        ax.set_ylabel('Frequency (Hz)', fontsize = 12)


# Extract templates
for row in range(len(list_tmpl)):
    file_num, x0, xn, y0, yn, wclass = list_tmpl.ix[row,:]
    wclass = 'Whale' if wclass=='H0' else 'Non-whale'
    # Read one file
    params = {'NFFT':256, 'Fs':2000, 'noverlap':192} 
    s = ReadAIFF(path_data + 'train/train%d.aiff' % file_num)
    P, freqs, bins = mlab.specgram(s, **params)
    m, n = P.shape
    
    # Pre-process image
    Q = slidingWindowV(P,maxM=m)
    R = Q.copy()

    # Set background based on the class of the sound
    set_val = Q.max() + 1 if wclass != 'Whale' else Q.min() - 1
    R[0:x0,:], R[xn:130,:] = set_val, set_val
    R[:,0:y0], R[:,yn:60] = set_val, set_val
    
    # Set border around image
    R[x0:xn,y0], R[x0:xn,yn] = -set_val, -set_val
    R[x0,y0:yn], R[xn,y0:yn] = -set_val, -set_val
    
    # Plot
    fig = plt.figure(figsize = (14,4))
    ax1 = plt.subplot(131)
    plot_spectrogram(ax1, P, 1)
    plt.title('Original')
    ax2 = plt.subplot(132)
    plot_spectrogram(ax2, Q)
    plt.title('Enhanced')
    ax3 = plt.subplot(133)
    plot_spectrogram(ax3, R)
    plt.title('Template')
    plt.suptitle('Template - %s' % wclass, fontsize = 16)
    plt.subplots_adjust(top=1.05)
    plt.show()