import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy import signal
from scipy.signal import butter, filtfilt
from coughvid.src.feature_class import features
#import pickle
import os
from sklearn.utils import shuffle
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data_folder = 'D:\\Zhang\\public_dataset_wav\\public_dataset_wav_json'
var1='filename'
var2='label'
#scaler = pickle.load(open(os.path.join('C:\\Users\\kaicy\\PycharmProjects\\coughvid_', 'cough_classification_scaler'), 'rb'))

resultX = []
resultY = []


def preprocess_cough(x, fs, cutoff=6000, normalize=True, filter_=True, downsample=True):
    """
    Normalize, lowpass filter, and downsample cough samples in a given data folder

    Inputs: x*: (float array) time series cough signal
    fs*: (int) sampling frequency of the cough signal in Hz
    cutoff: (int) cutoff frequency of lowpass filter
    normalize: (bool) normailzation on or off
    filter: (bool) filtering on or off
    downsample: (bool) downsampling on or off
    *: mandatory input

    Outputs: x: (float32 array) new preprocessed cough signal
    fs: (int) new sampling frequency
    """

    fs_downsample = cutoff * 2

    # Preprocess Data
    if len(x.shape) > 1:
        x = np.mean(x, axis=1)  # Convert to mono
    if normalize:
        x = x / (np.max(np.abs(x)) + 1e-17)  # Norm to range between -1 to 1
    if filter_:
        b, a = butter(4, fs_downsample / fs, btype='lowpass')  # 4th order butter lowpass filter
        x = filtfilt(b, a, x)
    if downsample:
        x = signal.decimate(x, int(fs / fs_downsample))  # Downsample for anti-aliasing

    fs_new = fs_downsample

    return np.float32(x), fs_new


def segment_cough(x, fs, cough_padding=0.2, min_cough_len=0.2, th_l_multiplier=0.1, th_h_multiplier=2):
    """Preprocess the data by segmenting each file into individual coughs using a hysteresis comparator on the signal power

    Inputs:
    *x (np.array): cough signal
    *fs (float): sampling frequency in Hz
    *cough_padding (float): number of seconds added to the beginning and end of each detected cough to make sure coughs are not cut short
    *min_cough_length (float): length of the minimum possible segment that can be considered a cough
    *th_l_multiplier (float): multiplier of the RMS energy used as a lower threshold of the hysteresis comparator
    *th_h_multiplier (float): multiplier of the RMS energy used as a high threshold of the hysteresis comparator

    Outputs:
    *coughSegments (np.array of np.arrays): a list of cough signal arrays corresponding to each cough
    cough_mask (np.array): an array of booleans that are True at the indices where a cough is in progress"""

    cough_mask = np.array([False] * len(x))

    # Define hysteresis thresholds
    rms = np.sqrt(np.mean(np.square(x)))
    seg_th_l = th_l_multiplier * rms
    seg_th_h = th_h_multiplier * rms

    # Segment coughs
    coughSegments = []
    padding = round(fs * cough_padding)
    min_cough_samples = round(fs * min_cough_len)
    cough_start = 0
    cough_end = 0
    cough_in_progress = False
    tolerance = round(0.01 * fs)
    below_th_counter = 0

    for i, sample in enumerate(x ** 2):
        if cough_in_progress:
            if sample < seg_th_l:
                below_th_counter += 1
                if below_th_counter > tolerance:
                    cough_end = i + padding if (i + padding < len(x)) else len(x) - 1
                    cough_in_progress = False
                    if (cough_end + 1 - cough_start - 2 * padding > min_cough_samples):
                        coughSegments.append(x[cough_start:cough_end + 1])
                        cough_mask[cough_start:cough_end + 1] = True
            elif i == (len(x) - 1):
                cough_end = i
                cough_in_progress = False
                if (cough_end + 1 - cough_start - 2 * padding > min_cough_samples):
                    coughSegments.append(x[cough_start:cough_end + 1])
            else:
                below_th_counter = 0
        else:
            if sample > seg_th_h:
                cough_start = i - padding if (i - padding >= 0) else 0
                cough_in_progress = True

    return coughSegments, cough_mask


csv_data = pd.read_csv('dataset_xgboost.csv')
csv_data_ = shuffle(csv_data)
csv_data_.to_csv("shuffle_data.csv", index=0)
csv_data = pd.read_csv('shuffle_data.csv')
#print(type(csv_data[var2][0]))

for i in range(len(csv_data)):
    fs, cough = wavfile.read(data_folder+"//"+csv_data[var1][i])
    cough_, fs = preprocess_cough(cough, fs)
    #cough_list, cough_mask = segment_cough(cough_, fs)
    data = (fs, cough_)
    FREQ_CUTS = [(0,200),(300,425),(500,650),(950,1150),(1400,1800),(2300,2400),(2850,2950),(3800,3900)]
    features_fct_list = ['spectral_features', 'SSL_SD', 'MFCC', 'PSD']
    feature_values_vec = []
    obj = features(FREQ_CUTS)
    for feature in features_fct_list:
        feature_values, feature_names = getattr(obj, feature)(data)
        for value in feature_values:
            if isinstance(value, np.ndarray):
                feature_values_vec.append(value[0])
            else:
                feature_values_vec.append(value)
    # scaler = StandardScaler().fit(np.array(feature_values_vec).reshape(-1, 1))
    # feature_values_scaled = scaler.transform(np.array(feature_values_vec).reshape(-1, 1))
    feature_values_scaled = np.array(feature_values_vec)
    resultX.append(feature_values_scaled)
    resultY.append(csv_data[var2][i])

X = np.array(resultX)
print(X.shape)
Y = np.array(resultY)
print(Y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
xgb = XGBClassifier(n_estimators=100)
eval_set = [(X_test, y_test)]
xgb.fit(X_train, y_train, eval_metric="logloss", verbose=True, eval_set=eval_set)
#eval_metric="logloss"
#early_stopping_rounds=20
#xgb.save_model("1.model")
pre_test = xgb.predict(X_test)
predictions = [round(value) for value in pre_test]

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# pre_prob = xgb.predict_proba(X_test)
# print(pre_prob[:, 1])

