import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
#from IPython.display import Image
import aifc

# Match template function
def match_template(template_to_match, image_where_to_search):
    # Read the images
    img = cv2.imread(image_where_to_search,0)
    img2 = img.copy()
    template = cv2.imread(template_to_match,0)
    w, h = template.shape[::-1]

    # Set method 
    meth = 'cv2.TM_CCOEFF_NORMED'
    img = img2.copy()
    method = eval(meth)

    # Apply template Matching
    res = cv2.matchTemplate(img,template,method)
    # Obtain vals and locs (max_lox corresponds to the best match)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    print('Start time=',max_loc[0] , 'End time=', max_loc[0] + w)

    # Add rectangle with match
    bottom_right = (max_loc[0] + w, max_loc[1] + h)
    cv2.rectangle(img,max_loc, bottom_right, 255, 3)

    # Plot matching and image with best match in rectangle
    plt.figure(figsize = (14,8))
    plt.subplot(121),plt.imshow(res,cmap = 'gray', interpolation='nearest')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img, cmap = 'gray', interpolation='nearest')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.show()

# ReadAIFF function
def ReadAIFF(file):
    s = aifc.open(file,'r')
    nFrames = s.getnframes()
    strSig = s.readframes(nFrames)
    return np.fromstring(strSig, np.short).byteswap()

# Save image of spectrogram (we intentionally remove labels from the chart in this case)
def save_spectrogram(filename):
    sound = ReadAIFF(filename)
    fig = plt.figure(figsize = (10,6));
    my_cmap = matplotlib.cm.get_cmap('hsv_r');
    params = {'NFFT':256, 'Fs':2000, 'noverlap':192, 'cmap' : my_cmap};
    plot1 = plt.specgram(sound, **params);
    return fig;


# Save image of whale sound
plot1 = save_spectrogram('whale.aiff');
plot1.savefig('assets/whale1.png');
plt.close(plot1)

# Save image of another whale sound
plot1 = save_spectrogram('whale_.aiff');
plot1.savefig('assets/whale2.png');
plt.close(plot1)

# Save image of a non-whale sound
plot1 = save_spectrogram('no_whale.aiff');
plot1.savefig('assets/nonwhale1.png');
plt.close(plot1)

plot1 = save_spectrogram('long_sample_03.aiff');
plot1.savefig('assets/whale3.png');
plt.close(plot1)


match_template('assets/whale_template.png', 'assets/whale1.png') #whale
match_template('assets/whale_template.png', 'assets/whale2.png') #whale
match_template('assets/whale_template.png', 'assets/nonwhale1.png') #non-whale
match_template('assets/whale_template.png', 'assets/whale3.png')