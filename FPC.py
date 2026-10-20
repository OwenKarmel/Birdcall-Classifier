# 2024 Final Programming Competition Submission
# Owen Karmel

# imports
import time
import pandas as pd
import numpy as np
import librosa
from scipy.io import wavfile
from tqdm import tqdm
from sklearn import preprocessing
from sklearn import ensemble

train_x_size = 12000  # should be 12000
test_x_size = 3000  # should be 3000
wavfile_length = 24000

# Set up training set
t1 = time.time()
df = pd.read_csv('train.csv')  # Read train.csv
train_x = np.empty((train_x_size, wavfile_length))  # initialize train_x
path = 'train/train/'  # path from root directory to training .wav files
print('creating variable train_x')
for i in tqdm(range(train_x_size)):  # Creates train_x from wav files
    train_x[i] = np.array(wavfile.read(path+f'train_{i}.wav')[1])
train_y = df.Label  # train y is just the labels column from the csv

print('train_x to np.array time: ', time.time()-t1)  # time to run
sr = wavfile.read(f'train/train/train_{i}.wav')[0]  # determines sample rate

# Set up test set
test_x = np.empty((test_x_size, wavfile_length))  # initialize test_x
test_ids = np.empty(test_x_size,dtype='int32')
path = 'test/test/'  # path from root directory to training .wav files
print('creating variable test_x')
for i in tqdm(range(test_x_size)):  # creates test_x from wav files
    test_x[i] = np.array(wavfile.read(path+f'test_{i}.wav')[1])
    test_ids[i] = i  # ID column for the submission .csv

print('test_x to np.array time: ', time.time()-t1)  # time to run


# Feature Extraction
def extract_librosa_features(y, sr):
    """
    Extract auditory features from wavfiles and flatten them into
    training data for model training.
    :param y: ndarray of wavfile
    :param sr: sampling rate of wavfile
    :return: flattened string of the features mfcc,
    rms spectral_centroid, spectral_bandwidth, spectral_flatness and
    spectral_rolloff
    """
    return np.concatenate((
        np.ndarray.flatten(librosa.feature.mfcc(y=y,sr=sr)),
        np.ndarray.flatten(librosa.feature.rms(y=y)),
        np.ndarray.flatten(librosa.feature.spectral_centroid(y=y,sr=sr)),
        np.ndarray.flatten(librosa.feature.spectral_contrast(y=y,sr=sr,fmin=100)),
        np.ndarray.flatten(librosa.feature.spectral_bandwidth(y=y,sr=sr)),
        np.ndarray.flatten(librosa.feature.spectral_flatness(y=y)),
        np.ndarray.flatten(librosa.feature.spectral_rolloff(y=y,sr=sr))

    ))


print('extracting librosa features for train_x:')
train_x_features = np.array([extract_librosa_features(sample, sr)
                             for sample in tqdm(train_x)])
# normalize train_x
train_x_features = preprocessing.StandardScaler().fit_transform(train_x_features)

print('extracting librosa features for test_x:')
test_x_features = np.array([extract_librosa_features(sample, sr)
                            for sample in tqdm(test_x)])
# normalize test_x
test_x_features = preprocessing.StandardScaler().fit_transform(test_x_features)

print('#######Gradient Boosting on Librosa Features########')
GBC = ensemble.HistGradientBoostingClassifier(
    learning_rate=.01, max_iter=1500, verbose=True)  # initialize classifier
start = time.time()
GBC.fit(train_x_features,train_y[:train_x_size])  # fit the data
interval1 = time.time()
print(f'GBC model fit time: {interval1-start:.2f} s')
print(f'training accuracy: {GBC.score(train_x_features,train_y[:train_x_size])}')
interval2 = time.time()
print(f'train predict time: {interval2-interval1:.2f} s')
prediction = GBC.predict(test_x_features)  # make prediction using testing data
interval3 = time.time()
print(f'test predict time: {interval3-interval2:.2f} s')

# Make output CSV
output_df = pd.DataFrame({'# ID': test_ids, 'Label': prediction})
output_df.to_csv('submission.csv', index=False)
tf = time.time()

print('final time: ', tf-t1, ' s')
