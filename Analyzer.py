import librosa
import numpy as np
import argparse
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score

SAVE_PATH = 'model'

def extract_features(fn, bands=60, frames=41):
    def _windows(data, window_size):
        start = 0
        while start < len(data):
            yield int(start), int(start + window_size)
            start += (window_size // 2)
            
    window_size = 512 * (frames - 1)
    features = []
    segment_log_specgrams = []
    sound_clip,sr = librosa.load(fn)

    for (start,end) in _windows(sound_clip,window_size):
        if(len(sound_clip[start:end]) == window_size):
            signal = sound_clip[start:end]
            melspec = librosa.feature.melspectrogram(signal,n_mels=bands)
            logspec = librosa.amplitude_to_db(melspec)
            logspec = logspec.T.flatten()[:, np.newaxis].T
            segment_log_specgrams.append(logspec)
        
    segment_log_specgrams = np.asarray(segment_log_specgrams).reshape(
        len(segment_log_specgrams),bands,frames,1)
    segment_features = np.concatenate((segment_log_specgrams, np.zeros(
        np.shape(segment_log_specgrams))), axis=3)
    for i in range(len(segment_features)): 
        segment_features[i, :, :, 1] = librosa.feature.delta(
            segment_features[i, :, :, 0])
    
    if len(segment_features) > 0: # check for empty segments 
        return segment_features

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--file', type=str, default='')
    opt = parser.parse_args()

    X = extract_features(opt.file)
	
    print(X.shape)
    model = keras.models.load_model(SAVE_PATH)

    avg_p = np.mean(model.predict(X), axis = 0)
    print(avg_p)
