"""
Created on Wed Mar 25 12:54:28 2020

@author: gallicch
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

from scipy.io import loadmat

# from sktime.utils.data_io import load_from_tsfile_to_dataframe
from sktime.datasets._data_io import load_from_tsfile_to_dataframe  # V 0.10
# from sktime.utils.load_data import load_from_tsfile_to_dataframe  # Change this for old version

import _pickle as cPickle


def load_waf_dataset():
    # ecg
    data = loadmat('WAF.mat')
    X_train = data['X']
    X_test = data['Xte']
    y_train = data['Y'] - 1
    y_test = data['Yte'] - 1
    return (X_train, y_train, X_test, y_test, X_train.shape[-1], 2)


def load_ecg_dataset():
    # ecg
    data = loadmat('ECG.mat')
    X_train = data['X']
    X_test = data['Xte']
    y_train = data['Y'] - 1
    y_test = data['Yte'] - 1
    return (X_train, y_train, X_test, y_test, X_train.shape[-1], 2)


def load_blood_dataset():
    data = loadmat('BLOOD.mat')
    X_train = data['X']
    X_test = data['Xte']
    y_train = data['Y']
    y_test = data['Yte']
    return (X_train, y_train, X_test, y_test, X_train.shape[-1], 2)


def load_char_dataset():
    data = loadmat('CHAR.mat')
    X_train = data['X']
    X_test = data['Xte']
    y_train = data['Y'] - 1
    y_test = data['Yte'] - 1
    return (X_train, y_train, X_test, y_test, X_train.shape[-1], 20)


def load_jap_dataset():
    data = loadmat('JAP.mat')
    X_train = data['X']
    X_test = data['Xte']
    y_train = data['Y']
    y_test = data['Yte']
    return (X_train, y_train, X_test, y_test, X_train.shape[-1], 10)


# to properly use padding in the reservoir state computation, use the following
# function to replace every row where all elements are equal to the padding value
# with an np.nan value    
def nan_padding(X, value=0):
    # X is a 3d tensor with shape [num_sequences, num_timesteps, num_features]
    for sequence in range(X.shape[0]):
        for t in range(X.shape[1]):
            if (np.sum(X[sequence, t, :] == 0) == X.shape[2]):
                X[sequence, t, :] = np.repeat(np.nan, X.shape[2])
    return X


def convert_input_from_sktime(X):
    all_data = []
    lengths = []

    # X.shape[0] is the number of time-series in the dataset_name
    # X.shape[1] is the number of features in each time-series

    num_time_series = X.shape[0]
    num_features = X.shape[1]

    for i in range(num_time_series):
        time_series = []  # this is going to contain the i-th time-series
        num_timesteps = 0
        for j in range(num_features):
            # X.values[i][j] contains the elements of the j-th feature
            # in the i-th time-series
            if j == 0:
                num_timesteps = X.values[i][j].shape[0]
            else:
                if num_timesteps > X.values[i][j].shape[0]:
                    num_timesteps = X.values[i][j].shape[0]
        lengths.append(num_timesteps)
        for j in range(num_features):
            dimension = []
            l = 0
            for x in X.values[i][j]:
                dimension.append(x)
                l = l + 1
                if l == num_timesteps:
                    break
            if j == 0:
                time_series = np.column_stack([dimension])
            else:
                time_series = np.column_stack([time_series, dimension])
        all_data.append(time_series)
    max_length = max(lengths)
    X_c = np.zeros(shape=(len(all_data), max_length, num_features))
    for i in range(len(all_data)):
        X_c[i, :lengths[i], :] = all_data[i]
    return X_c, lengths


def convert_class_from_sktime(y):
    _, y_c = np.unique(y, return_inverse=True)
    return y_c


def load_sktime_dataset(filename):
    x, y = load_from_tsfile_to_dataframe(filename)
    return convert_input_from_sktime(x)[0], convert_class_from_sktime(y)


def convert_polyphonic(data):
    num_sequences = len(data)
    num_notes = 88

    lengths = []
    for s in data:
        lengths.append(len(s))

    time_steps = max(lengths)
    all_data = np.zeros(shape=(num_sequences, time_steps, num_notes))
    for s in range(len(data)):
        for t in range(len(data[s])):
            for note in data[s][t]:
                # print(datasets[s][t])
                # print(note)
                all_data[s, t, note - 21] = 1

    X = all_data[:, :-1, :]
    Y = all_data[:, 1:, :]

    return X, Y


def load_polyphonic_data(filename, path='time_series/Polyphonic/'):
    file = open(os.path.join(path, filename), 'rb')
    dataset = cPickle.load(file)
    file.close()

    X_train, y_train = convert_polyphonic(dataset['train'])
    X_valid, y_valid = convert_polyphonic(dataset['valid'])
    X_test, y_test = convert_polyphonic(dataset['test'])

    return X_train, y_train, X_valid, y_valid, X_test, y_test


class FrameLevelAccuracy(keras.metrics.Metric):
    def __init__(self, name="frame_level_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.value = self.add_weight(name="fla", initializer="zeros")

    def update_state(self, y_true, y_pred):
        TP = 0
        FP = 0
        FN = 0

        prediction = tf.round(y_pred)

        prediction_played = tf.cast(prediction == 1, 'int32')
        prediction_unplayed = tf.cast(prediction == 0, 'int32')
        target_played = tf.cast(y_true == 1, 'int32')
        target_unplayed = tf.cast(y_true == 0, 'int32')

        TP = tf.reduce_sum(tf.multiply(prediction_played, target_played))
        FP = tf.reduce_sum(tf.multiply(prediction_played, target_unplayed))
        FN = tf.reduce_sum(tf.multiply(prediction_unplayed, target_played))

        den = tf.reduce_sum([TP, FP, FN])
        computed_value = tf.cast(tf.divide(TP, den), 'float32')

        self.value.assign_add(tf.cast(computed_value, 'float32'))

        # self.value.assign_add(music_accuracy(y_true,y_pred))

    def result(self):
        return self.value

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.value.assign(0.0)


def music_accuracy(y_true, y_pred):
    threshold = 0.5;

    X = y_pred
    Y = y_true

    X = K.transpose(X)
    Y = K.transpose(Y)

    Nsys = K.sum(K.cast(K.greater(X, threshold), y_pred.dtype), axis=0)
    Nref = K.sum(K.cast(K.greater(Y, threshold), y_pred.dtype), axis=0)
    Ncorr = K.sum(K.cast(K.greater(X, threshold), y_pred.dtype) * K.cast(K.greater(Y, threshold), y_pred.dtype), axis=0)

    TP = K.sum(Ncorr)
    FP = K.sum(Nsys - Ncorr)
    FN = K.sum(Nref - Ncorr)
    ACCURACY = TP / (TP + FP + FN)
    return ACCURACY


def my_loss(y_true, y_pred):
    return tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(y_true, 'float32'),
                                                   logits=tf.cast(y_pred, 'float32'))
