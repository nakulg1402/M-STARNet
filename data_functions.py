import scipy.io as sio
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re, pickle
from sklearn.preprocessing import StandardScaler, label_binarize
import torch
import torch.nn.functional as F
from functools import partial
from torch.utils.data import DataLoader, TensorDataset, Sampler, Dataset
import os, time, json, math, random, mne
from copy import deepcopy
from collections import Counter, defaultdict
import torch.nn as nn
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import LeaveOneOut, train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, f1_score, precision_score
from torch_geometric.utils import dense_to_sparse
from matplotlib.patches import Ellipse, Arc
from scipy.stats import mode
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="numpy")
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc
import torch.optim as optim
from torch.nn import CrossEntropyLoss

def extract_and_sort_features(features_dict, prefix, exclude_prefix=None):
    metadata_keys = ['__header__', '__version__', '__globals__']
    for key in metadata_keys:
        features_dict.pop(key, None)

    if exclude_prefix:
        feature_keys = [key for key in features_dict.keys() 
                       if key.startswith(prefix) and not key.startswith(exclude_prefix)]
    else:
        feature_keys = [key for key in features_dict.keys() if key.startswith(prefix)]
    
    if not feature_keys:
        print(f"No keys found with prefix '{prefix}'")
        return {}
        
    trial_nums = []
    for key in feature_keys:
        num_str = key[len(prefix):]
        match = re.match(r'\d+', num_str)
        if match:
            trial_nums.append(int(match.group()))
        else:
            raise ValueError(f"Could not extract trial number from key: {key}")

    sorted_keys = [key for _, key in sorted(zip(trial_nums, feature_keys))]

    sorted_features = {}
    for new_key, old_key in enumerate(sorted_keys, start=1):
        sorted_features[str(new_key)] = features_dict[old_key]
    
    return sorted_features

def rearrange_eyes(eyes_dict):
    metadata_keys = ['__header__', '__version__', '__globals__']
    for key in metadata_keys:
        eyes_dict.pop(key, None)

    trial_numbers = sorted([int(key) for key in eyes_dict.keys()])

    sorted_eyes = {}
    for new_key, old_key in enumerate(trial_numbers, start=1):
        old_key_str = str(old_key)
        new_key_str = str(new_key)
        sorted_eyes[new_key_str] = eyes_dict[old_key_str]
    
    return sorted_eyes

def rearrange_labels(labels_dict):
    metadata_keys = ['__header__', '__version__', '__globals__']
    for key in metadata_keys:
        labels_dict.pop(key, None)

    trial_numbers = sorted([int(key) for key in labels_dict.keys()])

    sorted_labels = {}
    for new_key, old_key in enumerate(trial_numbers, start=1):
        old_key_str = str(old_key)
        new_key_str = str(new_key)
        sorted_labels[new_key_str] = labels_dict[old_key_str]
    
    return sorted_labels

def filter_frequency_bands(data, band_indices):
    if len(data.shape) != 3:
        raise ValueError("Data must be a 3D array.")
    return data[:, band_indices, :]

def extract_eye_features(eye_features, video_index):
    video_key = str(video_index)
    if video_key not in eye_features:
        raise ValueError(f"Video index {video_index} not found in eye_features.")

    features_array = eye_features[video_key]  # Shape: (n, 33)

    feature_ranges = {
        'pupil_diameters': (0, 12),      # Features 1-12: columns 0-11
        'fixation_duration': (12, 14),   # Features 13-14: columns 12-13
        'dispersion': (14, 18),          # Features 15-18: columns 14-17
        'saccade_duration': (18, 20),    # Features 19-20: columns 18-19
        'saccade_amplitude': (20, 22),   # Features 21-22: columns 20-21
        'blink_duration': (22, 24),      # Features 23-24: columns 22-23
        'event_statistics': (24, 33)     # Features 25-33: columns 24-32
    }

    extracted_features = {}
    for category, (start, end) in feature_ranges.items():
        extracted_features[category] = features_array[:, start:end]

    return extracted_features

def combine_eye_features_by_category(sorted_eyes):
    subjects_features = {}
    
    subject_features = {
        'pupil_diameters': {},
        'fixation_duration': {},
        'dispersion': {},
        'saccade_duration': {},
        'saccade_amplitude': {},
        'blink_duration': {},
        'event_statistics': {}
    }

    for j in range(1, 81):
        features = extract_eye_features(sorted_eyes, j)
        for category, data in features.items():
            subject_features[category][str(j)] = data

    subjects_features = subject_features
    
    return subjects_features

def prepare_emotion_de(sorted_de_lds_delta, sorted_de_lds_theta, sorted_de_lds_alpha, sorted_de_lds_beta, 
                         sorted_de_lds_gamma, sorted_labels, video_to_emotion, emotion_to_int):

    all_de_lds_delta = np.concatenate([sorted_de_lds_delta[f'{i}'] for i in range(1, 81)], axis=0)  # (total_samples, 62)
    all_de_lds_theta = np.concatenate([sorted_de_lds_theta[f'{i}'] for i in range(1, 81)], axis=0)  # (total_samples, 62)
    all_de_lds_alpha = np.concatenate([sorted_de_lds_alpha[f'{i}'] for i in range(1, 81)], axis=0)  # (total_samples, 62)
    all_de_lds_beta = np.concatenate([sorted_de_lds_beta[f'{i}'] for i in range(1, 81)], axis=0)  # (total_samples, 62)
    all_de_lds_gamma = np.concatenate([sorted_de_lds_gamma[f'{i}'] for i in range(1, 81)], axis=0)  # (total_samples, 62)

    de_lds_mean_delta = np.mean(all_de_lds_delta, axis=0)  # (,62)
    de_lds_mean_theta = np.mean(all_de_lds_theta, axis=0)  # (,62)
    de_lds_mean_alpha = np.mean(all_de_lds_alpha, axis=0)  # (,62)
    de_lds_mean_beta = np.mean(all_de_lds_beta, axis=0)  # (,62)
    de_lds_mean_gamma = np.mean(all_de_lds_gamma, axis=0)  # (,62)
    
    de_lds_std_delta = np.std(all_de_lds_delta, axis=0)  # (,62)
    de_lds_std_theta = np.std(all_de_lds_theta, axis=0)  # (,62)
    de_lds_std_alpha = np.std(all_de_lds_alpha, axis=0)  # (,62)
    de_lds_std_beta = np.std(all_de_lds_beta, axis=0)  # (,62)
    de_lds_std_gamma = np.std(all_de_lds_gamma, axis=0)  # (,62)

    standardized_de_lds_delta = {f'{i}': (sorted_de_lds_delta[f'{i}'] - de_lds_mean_delta) / (de_lds_std_delta + 1e-8) for i in range(1, 81)}
    standardized_de_lds_theta = {f'{i}': (sorted_de_lds_theta[f'{i}'] - de_lds_mean_theta) / (de_lds_std_theta + 1e-8) for i in range(1, 81)}
    standardized_de_lds_alpha = {f'{i}': (sorted_de_lds_alpha[f'{i}'] - de_lds_mean_alpha) / (de_lds_std_alpha + 1e-8) for i in range(1, 81)}
    standardized_de_lds_beta = {f'{i}': (sorted_de_lds_beta[f'{i}'] - de_lds_mean_beta) / (de_lds_std_beta + 1e-8) for i in range(1, 81)}
    standardized_de_lds_gamma = {f'{i}': (sorted_de_lds_gamma[f'{i}'] - de_lds_mean_gamma) / (de_lds_std_gamma + 1e-8) for i in range(1, 81)}
    
    X = []
    y_discrete = []
    y_continuous = []

    for video_index in range(1, 81):
        emotion = video_to_emotion[video_index]
        label_discrete = emotion_to_int[emotion]

        de_lds_delta = standardized_de_lds_delta[f'{video_index}']  # (n_samples, 62)
        de_lds_theta = standardized_de_lds_theta[f'{video_index}']  # (n_samples, 62)
        de_lds_alpha = standardized_de_lds_alpha[f'{video_index}']  # (n_samples, 62)
        de_lds_beta = standardized_de_lds_beta[f'{video_index}']  # (n_samples, 62)
        de_lds_gamma = standardized_de_lds_gamma[f'{video_index}']  # (n_samples, 62)
        labels_continuous = sorted_labels[f'{video_index}'][0]  # (n_samples,)
        
        n_samples = de_lds_alpha.shape[0]

        for i in range(n_samples):
            de_lds_flat_delta = de_lds_delta[i].flatten()  # (62,) = 1 * 62
            de_lds_flat_theta = de_lds_theta[i].flatten()  # (62,) = 1 * 62
            de_lds_flat_alpha = de_lds_alpha[i].flatten()  # (62,) = 1 * 62
            de_lds_flat_beta = de_lds_beta[i].flatten()  # (62,) = 1 * 62
            de_lds_flat_gamma = de_lds_gamma[i].flatten()  # (62,) = 1 * 62
            
            features = np.concatenate([de_lds_flat_delta, de_lds_flat_theta, de_lds_flat_alpha, de_lds_flat_beta, de_lds_flat_gamma])
            
            X.append(features)
            y_discrete.append(label_discrete)
            y_continuous.append(labels_continuous[i])

    X = np.array(X)  # (total_samples, 310)
    y_discrete = np.array(y_discrete)  # (total_samples,)
    y_continuous = np.array(y_continuous)  # (total_samples,)

    return X, y_discrete, y_continuous

def prepare_emotion_eye(pupil_diameters, fixation_duration, dispersion, saccade_duration, saccade_amplitude,
                        blink_duration, event_statistics, sorted_labels, video_to_emotion, emotion_to_int):

    all_pupils = np.concatenate([pupil_diameters[f'{i}'] for i in range(1, 81)], axis=0)  # (total_samples, 12)
    all_fixation = np.concatenate([fixation_duration[f'{i}'] for i in range(1, 81)], axis=0)  # (total_samples, 2)
    all_dispersion = np.concatenate([dispersion[f'{i}'] for i in range(1, 81)], axis=0)  # (total_samples, 4)
    all_saccade_duration = np.concatenate([saccade_duration[f'{i}'] for i in range(1, 81)], axis=0)  # (total_samples, 2)
    all_saccade_amplitude = np.concatenate([saccade_amplitude[f'{i}'] for i in range(1, 81)], axis=0)  # (total_samples, 2)
    all_blink_duration = np.concatenate([blink_duration[f'{i}'] for i in range(1, 81)], axis=0)  # (total_samples, 2)
    all_event_stats = np.concatenate([event_statistics[f'{i}'] for i in range(1, 81)], axis=0)  # (total_samples, 9)

    pupil_mean = np.mean(all_pupils, axis=0)  # (12,)
    pupil_std = np.std(all_pupils, axis=0)  # (12,)
    
    fixation_mean = np.mean(all_fixation, axis=0)  # (2,)
    fixation_std = np.std(all_fixation, axis=0)  # (2,)
    
    dispersion_mean = np.mean(all_dispersion, axis=0)  # (4,)
    dispersion_std = np.std(all_dispersion, axis=0)  # (4,)
    
    saccade_duration_mean = np.mean(all_saccade_duration, axis=0)  # (2,)
    saccade_duration_std = np.std(all_saccade_duration, axis=0)  # (2,)
    
    saccade_amplitude_mean = np.mean(all_saccade_amplitude, axis=0)  # (2,)
    saccade_amplitude_std = np.std(all_saccade_amplitude, axis=0)  # (2,)
    
    blink_duration_mean = np.mean(all_blink_duration, axis=0)  # (2,)
    blink_duration_std = np.std(all_blink_duration, axis=0)  # (2,)
    
    event_stats_mean = np.mean(all_event_stats, axis=0)  # (9,)
    event_stats_std = np.std(all_event_stats, axis=0)  # (9,)

    standardized_pupils = {f'{i}': (pupil_diameters[f'{i}'] - pupil_mean) / (pupil_std + 1e-8) for i in range(1, 81)}
    standardized_fixation = {f'{i}': (fixation_duration[f'{i}'] - fixation_mean) / (fixation_std + 1e-8) for i in range(1, 81)}
    standardized_dispersion = {f'{i}': (dispersion[f'{i}'] - dispersion_mean) / (dispersion_std + 1e-8) for i in range(1, 81)}
    standardized_saccade_duration = {f'{i}': (saccade_duration[f'{i}'] - saccade_duration_mean) / (saccade_duration_std + 1e-8) for i in range(1, 81)}
    standardized_saccade_amplitude = {f'{i}': (saccade_amplitude[f'{i}'] - saccade_amplitude_mean) / (saccade_amplitude_std + 1e-8) for i in range(1, 81)}
    standardized_blink_duration = {f'{i}': (blink_duration[f'{i}'] - blink_duration_mean) / (blink_duration_std + 1e-8) for i in range(1, 81)}
    standardized_event_stats = {f'{i}': (event_statistics[f'{i}'] - event_stats_mean) / (event_stats_std + 1e-8) for i in range(1, 81)}

    X = []
    y_discrete = []
    y_continuous = []

    for video_index in range(1, 81):
        emotion = video_to_emotion[video_index]
        label_discrete = emotion_to_int[emotion]

        pupil = standardized_pupils[f'{video_index}']  # (n_samples, 12)
        fixation = standardized_fixation[f'{video_index}']  # (n_samples, 2)
        dispersion = standardized_dispersion[f'{video_index}']  # (n_samples, 4)
        saccade_duration = standardized_saccade_duration[f'{video_index}']  # (n_samples, 2)
        saccade_amplitude = standardized_saccade_amplitude[f'{video_index}']  # (n_samples, 2)
        blink_duration = standardized_blink_duration[f'{video_index}']  # (n_samples, 2)
        event_stats = standardized_event_stats[f'{video_index}']  # (n_samples, 9)
        labels_continuous = sorted_labels[f'{video_index}'][0]  # (n_samples,)

        n_samples = pupil.shape[0]

        for i in range(n_samples):
            pupil_features = pupil[i]  # (12,)
            fixation_features = fixation[i]  # (2,)
            dispersion_features = dispersion[i]  # (4,)
            saccade_duration_features = saccade_duration[i]  # (2,)
            saccade_amplitude_features = saccade_amplitude[i]  # (2,)
            blink_duration_features = blink_duration[i]  # (2,)
            event_stats_features = event_stats[i]  # (9,)

            features = np.concatenate([pupil_features, fixation_features, dispersion_features, saccade_duration_features, saccade_amplitude_features,
                                       blink_duration_features, event_stats_features])
            
            X.append(features)
            y_discrete.append(label_discrete)
            y_continuous.append(labels_continuous[i])

    X = np.array(X)  # (total_samples, 33)
    y_discrete = np.array(y_discrete)  # (total_samples,)
    y_continuous = np.array(y_continuous)  # (total_samples,)

    return X, y_discrete, y_continuous

video_to_emotion = {
    1: 'Happy', 2: 'Neutral', 3: 'Disgust', 4: 'Sad', 5: 'Anger', 6: 'Anger', 7: 'Sad', 8: 'Disgust', 9: 'Neutral', 10: 'Happy',
    11: 'Happy', 12: 'Neutral', 13: 'Disgust', 14: 'Sad', 15: 'Anger', 16: 'Anger', 17: 'Sad', 18: 'Disgust', 19: 'Neutral', 20: 'Happy',
    21: 'Anger', 22: 'Sad', 23: 'Fear', 24: 'Neutral', 25: 'Surprise', 26: 'Surprise', 27: 'Neutral', 28: 'Fear', 29: 'Sad', 30: 'Anger',
    31: 'Anger', 32: 'Sad', 33: 'Fear', 34: 'Neutral', 35: 'Surprise', 36: 'Surprise', 37: 'Neutral', 38: 'Fear', 39: 'Sad', 40: 'Anger',
    41: 'Happy', 42: 'Surprise', 43: 'Disgust', 44: 'Fear', 45: 'Anger', 46: 'Anger', 47: 'Fear', 48: 'Disgust', 49: 'Surprise', 50: 'Happy',
    51: 'Happy', 52: 'Surprise', 53: 'Disgust', 54: 'Fear', 55: 'Anger', 56: 'Anger', 57: 'Fear', 58: 'Disgust', 59: 'Surprise', 60: 'Happy',
    61: 'Disgust', 62: 'Sad', 63: 'Fear', 64: 'Surprise', 65: 'Happy', 66: 'Happy', 67: 'Surprise', 68: 'Fear', 69: 'Sad', 70: 'Disgust',
    71: 'Disgust', 72: 'Sad', 73: 'Fear', 74: 'Surprise', 75: 'Happy', 76: 'Happy', 77: 'Surprise', 78: 'Fear', 79: 'Sad', 80: 'Disgust'
}

emotion_to_int = {
    'Happy': 0, 'Neutral': 1, 'Disgust': 2, 'Sad': 3, 'Anger': 4, 'Fear': 5, 'Surprise': 6
}

def make_batches(data, batch_size):
    N = data.shape[0]

    remainder = N % batch_size
    if remainder != 0:
        pad_len = batch_size - remainder
        pad_data = np.zeros((pad_len, 62, 5))
        data = np.concatenate([data, pad_data], axis=0)
        N = data.shape[0]
        
    num_batches = N // batch_size
    data = data.reshape(num_batches, batch_size, 62, 5)
    return data

def make_eye_batches(data, batch_size):
    N = data.shape[0]

    remainder = N % batch_size
    if remainder != 0:
        pad_len = batch_size - remainder
        pad_data = np.zeros((pad_len, 33))
        data = np.concatenate([data, pad_data], axis=0)
        N = data.shape[0]
        
    num_batches = N // batch_size
    data = data.reshape(num_batches, batch_size, 33)
    return data
