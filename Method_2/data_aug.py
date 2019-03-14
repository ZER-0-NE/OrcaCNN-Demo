import numpy as np
import random
import itertools
import librosa
import matplotlib.pyplot as plt

def load_audio_file(file_path):
    input_length = 44100
    data = librosa.core.load(file_path)[0] #, sr=16000
    if len(data)>input_length:
        data = data[:input_length]
    else:
        data = np.pad(data, (0, max(0, input_length - len(data))), "constant")
    return data

def plot_time_series(data):
    fig = plt.figure(figsize=(14, 8))
    plt.title('Raw wave ')
    plt.ylabel('Amplitude')
    plt.plot(np.linspace(0, 1, len(data)), data)
    plt.show()

def write_audio_file(file, data, sample_rate=44100):
    librosa.output.write_wav(file, data, sample_rate)

data = load_audio_file("wavfiles/pos_00.wav")
plot_time_series(data)


# Adding white noise 
wn = np.random.randn(len(data))
data_wn = data + 0.005*wn
plot_time_series(data_wn)

data_roll = np.roll(data, 1600)
plot_time_series(data_roll)

write_audio_file('wavfiles/pos_002.wav',data_roll, sample_rate=44100)