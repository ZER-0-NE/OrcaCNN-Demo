import os
from scipy.io import wavfile
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from python_speech_features import mfcc
import pickle
import pandas as pd
from keras.callbacks import ModelCheckpoint
from cfg import Config
from keras.models import load_model

def build_predictions(audio_dir):
	y_true = []
	y_pred = []
	fn_prob = {}

	print('Extracting features from audio')
	for fn in tqdm(os.listdir(audio_dir)):
		rate, wav = wavfile.read(os.path.join(audio_dir, fn))
		# label = fn2class[fn]
		# c = classes.index(label) # pull out classes
		y_prob = []

		for i in range(0, wav.shape[0] - config.step, config.step):
			sample = wav[i:i+config.step]
			x = mfcc(sample, rate,
							numcep=config.nfeat, nfilt=config.nfilt, nfft=config.nfft)
			x = (x - config.min)/ (config.max - config.min)
			print(x.shape)
			if config.mode == 'conv':
				x = x.reshape(1, x.shape[0], x.shape[1], 1)
				print(x.shape)
			elif config.mode == 'time':
				x = np.expand_dims(x, axis=0)
			#print(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
			x.reshape([1, 9, x.shape[2]])
			#x = np.expand_dims(x, axis=1)
			# np.resize(x, (-1, (9,13, 1)))
			y_hat = model.predict(x)
			print(np.argmax(y_hat))
			y_prob.append(y_hat)
			y_pred.append(np.argmax(y_hat))
			y_true.append(c)
			
		#fn_prob[fn] = np.mean(y_prob, axis=0).flatten()
		#print(fn_prob[fn])
	return y_true, y_pred, fn_prob


df = pd.read_csv('whales.csv')
classes = list(np.unique(df.label))
#print(classes)
fn2class = dict(zip(df.fname, df.label))
#print(fn2class)
p_path = os.path.join('pickles', 'conv.p')

with open(p_path, 'rb') as handle:
	config = pickle.load(handle)

model = load_model(config.model_path)
y_true, y_pred, fn_prob = build_predictions('clean1/')
#print(y_true, y_pred, fn_prob)
# acc_score = accuracy_score(y_true=y_true, y_pred = y_pred)

# y_probs = []

# for i, row in df.iterrows():
# 	y_prob = fn_prob[row.fname]
# 	y_probs.append(y_prob)
# 	for c, p in zip(classes, y_prob):
# 		df.at[i, c] = p

# y_pred = [classes[np.argmax(y)] for y in y_probs]
# df['y_pred'] = y_pred

# df.to_csv('predictions1.csv', index=False)