#!/usr/bin/env python
# coding: utf-8

# 1. Import Libraries
import scipy.io as sio
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
from sklearn.preprocessing import StandardScaler, label_binarize
import torch
import torch.nn.functional as F
from functools import partial
from torch.utils.data import DataLoader, TensorDataset, Sampler, Dataset, WeightedRandomSampler
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
from torch.optim.lr_scheduler import OneCycleLR
import shap
from shap import Explanation
import lime
import lime.lime_tabular
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="numpy")
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, f1_score, roc_auc_score, roc_curve, auc
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import CosineAnnealingLR


# 2. Featurization
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


# 3. Filter frequency bands in EEG
def filter_frequency_bands(data, band_indices):
    if len(data.shape) != 3:
        raise ValueError("Data must be a 3D array.")
    return data[:, band_indices, :]


# 4.EYES Feature extraction
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


# 5. Prepare Emotion Data - EEG (DE)
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


# 6. Prepare Emotion Data - EYE
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

# 7. video-to-emotion-to-integer mapping from the tabl
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

# 8. Make data batches
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


# 9. Multimodal Data Formation
for i in range(1, 21):
    eeg_path = f'/.../SEED VII Dataset/EEG_features/{i}.mat'
    eye_path = f'/.../SEED VII Dataset/EYE_features/{i}.mat'
    labels_path = f'/.../SEED VII Dataset/continuous_labels/{i}.mat'

    matf_1 = sio.loadmat(eeg_path)
    mat_eye = sio.loadmat(eye_path)
    labels_data = sio.loadmat(labels_path)

    band_names = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    band_indices = {'delta': 0, 'theta': 1, 'alpha': 2, 'beta': 3, 'gamma': 4}

    sorted_de_lds = extract_and_sort_features(matf_1, 'de_LDS_')

    delta_data = {}
    theta_data = {}
    alpha_data = {}
    beta_data = {}
    gamma_data = {}

    band_storage_de_lds = {
        'delta': delta_data,
        'theta': theta_data,
        'alpha': alpha_data,
        'beta': beta_data,
        'gamma': gamma_data
    }
    
    for key in sorted_de_lds:
        original_data = sorted_de_lds[key]  # Shape: (n_samples, 5, 62)
        for band in band_names:
            band_index = band_indices[band]
            filtered_data = filter_frequency_bands(original_data, [band_index])  # Shape: (n_samples, 1, 62)
            band_data = filtered_data[:, 0, :]
            band_storage_de_lds[band][key] = band_data
    
    sorted_de_lds_delta = band_storage_de_lds['delta']
    sorted_de_lds_theta = band_storage_de_lds['theta']
    sorted_de_lds_alpha = band_storage_de_lds['alpha']
    sorted_de_lds_beta = band_storage_de_lds['beta']
    sorted_de_lds_gamma = band_storage_de_lds['gamma']
    
    sorted_eyes = rearrange_eyes(mat_eye)
    all_features = combine_eye_features_by_category(sorted_eyes)
    subject_features = all_features
    
    pupil_diameters = subject_features['pupil_diameters']
    fixation_duration = subject_features['fixation_duration']
    dispersion = subject_features['dispersion']
    saccade_duration = subject_features['saccade_duration']
    saccade_amplitude = subject_features['saccade_amplitude']
    blink_duration = subject_features['blink_duration']
    event_statistics = subject_features['event_statistics']
    
    sorted_labels = rearrange_labels(labels_data)
    
    X_de, y_discrete_de, y_continuous_de = prepare_emotion_de(sorted_de_lds_delta, sorted_de_lds_theta, sorted_de_lds_alpha,
                                                              sorted_de_lds_beta, sorted_de_lds_gamma, sorted_labels,
                                                              video_to_emotion, emotion_to_int)
    
    X_eye, _, _ = prepare_emotion_eye(pupil_diameters, fixation_duration, dispersion, saccade_duration, saccade_amplitude,
                                      blink_duration, event_statistics, sorted_labels, video_to_emotion, emotion_to_int)
    
    if X_de.shape[0] == 3487:
        index = torch.randint(0, X_de.shape[0], (1,))
        x_sample = X_de[index].reshape(1, -1)
        x_eye_sample = X_eye[index].reshape(1, -1)
        
        y_sample = y_discrete_de[index].reshape(-1)
        y_cont   = y_continuous_de[index].reshape(-1)
            
        X_de = np.concatenate([X_de, x_sample], axis=0)
        X_eye = np.concatenate([X_eye, x_eye_sample], axis=0)
        
        y_discrete_de = np.concatenate([y_discrete_de, y_sample], axis=0)
        y_continuous_de = np.concatenate([y_continuous_de, y_cont], axis=0)
        
    if X_de.shape[0] > 3488:
        X_de = X_de[:3488]
        X_eye = X_eye[:3488]
        y_discrete_de = y_discrete_de[:3488]
        y_continuous_de = y_continuous_de[:3488]
        
    if X_de.shape[1] != 310:
        raise ValueError("Input data must have 310 features per sample")
    else:
        X_de = X_de.reshape(X_de.shape[0], 62, 5)
        
    batch_size = 16
    X_de = make_batches(X_de, batch_size)
    X_eye = make_eye_batches(X_eye, batch_size)
    
    globals()[f'X_de_{i}'] = X_de
    globals()[f'y_discrete_de_{i}'] = y_discrete_de
    globals()[f'y_continuous_de_{i}'] = y_continuous_de

    print(f"Processed file EEG DE {i}: X shape: {X_de.shape}, y_discrete shape: {y_discrete_de.shape}, y_continuous shape: {y_continuous_de.shape}")
    
    globals()[f'X_eye_{i}'] = X_eye
    print(f"Processed file EYE {i}: X shape: {X_eye.shape}")

# 10. M-STARNet Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEATURE_DIM = 256
NUM_CLASSES = 7

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (seq_len, batch_size, d_model)
        x = x + self.pe[:x.size(0), :]
        return x
        
class SMTFeatureEncoder(nn.Module):
    def __init__(self, feature_dim=FEATURE_DIM, nhead=8, num_layers=3):
        super().__init__()
        
        # 1. Spatial Branch (9x9 grid)
        self.spatial_patch_embed = nn.Linear(80, feature_dim)
        self.spatial_pos_embed = PositionalEncoding(feature_dim, max_len=25)
        spatial_encoder_layer = nn.TransformerEncoderLayer(feature_dim, nhead=nhead, dim_feedforward=feature_dim*4, dropout=0.1)
        self.spatial_transformer = nn.TransformerEncoder(spatial_encoder_layer, num_layers=num_layers)
        
        # 2. Temporal Branch (1D Conv + Transformer)
        self.temporal_conv = nn.Conv1d(310, feature_dim, kernel_size=1)
        self.temporal_pos_embed = PositionalEncoding(feature_dim, max_len=16)
        temporal_encoder_layer = nn.TransformerEncoderLayer(feature_dim, nhead=nhead, dim_feedforward=feature_dim*4, dropout=0.1)
        self.temporal_transformer = nn.TransformerEncoder(temporal_encoder_layer, num_layers=num_layers)
        
        # 3. Ocular Branch (1D Conv + Transformer)
        self.eye_conv = nn.Conv1d(33, feature_dim, kernel_size=1)
        self.eye_pos_embed = PositionalEncoding(feature_dim, max_len=16)
        eye_encoder_layer = nn.TransformerEncoderLayer(feature_dim, nhead=nhead, dim_feedforward=feature_dim*4, dropout=0.1)
        self.eye_transformer = nn.TransformerEncoder(eye_encoder_layer, num_layers=num_layers)
        
        # 4. Multimodal Fusion (with CLS token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, feature_dim))
        fusion_encoder_layer = nn.TransformerEncoderLayer(feature_dim, nhead=nhead, dim_feedforward=feature_dim*2, dropout=0.2)
        self.fusion_transformer = nn.TransformerEncoder(fusion_encoder_layer, num_layers=2)
        
        # Classification Head
        self.fc_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(0.4),
            nn.Linear(feature_dim, NUM_CLASSES)
        )
        
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
    
    def _spatial_forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        B = x.size(0)
        x = x.view(B, x.size(1), -1).permute(2, 0, 1)
        x = self.spatial_patch_embed(x) 
        x = self.spatial_pos_embed(x)
        x = self.spatial_transformer(x)
        return x.mean(0) # (B, 256)

    def _temporal_forward(self, x):
        B = x.size(0)
        if x.dim() > 3:
            x = x.reshape(B, -1, x.size(-1)) 
        x = x.reshape(B, 16, 310)
        
        B, T, F = x.size()
        if F != self.temporal_conv.in_channels:
            raise ValueError(f"Temporal feature dimension mismatch. Expected {self.temporal_conv.in_channels} features, but got {F}.")
        x = x.permute(0, 2, 1).contiguous()
        
        x = self.temporal_conv(x)
        
        x = x.permute(2, 0, 1)
        x = self.temporal_pos_embed(x)
        x = self.temporal_transformer(x)
        return x.mean(0)

    def _eye_forward(self, x):        
        B = x.size(0)
        if x.dim() > 3:
            x = x.reshape(B, -1, x.size(-1))
            
        x = x.reshape(B, 16, 33)

        B, T, F = x.size()
        if F != self.eye_conv.in_channels:
            raise ValueError(f"Eye feature dimension mismatch. Expected {self.eye_conv.in_channels} features, but got {F}.")
        x = x.permute(0, 2, 1).contiguous()
        
        x = self.eye_conv(x)
        
        x = x.permute(2, 0, 1)
        x = self.eye_pos_embed(x)
        x = self.eye_transformer(x)
        return x.mean(0)
    
    def forward(self, x_spatial, x_temporal, x_eye):
        spatial_feat = self._spatial_forward(x_spatial)
        temporal_feat = self._temporal_forward(x_temporal)
        eye_feat = self._eye_forward(x_eye)
        
        modality_features = torch.stack([spatial_feat, temporal_feat, eye_feat], dim=0) # (3, B, 256)

        # ---Random Modality Dropout for Subject Generalization---
        if self.training:
            p_dropout = 0.3
            mask = torch.ones(3, 1, 1, device=modality_features.device)
            for i in range(3):
                if torch.rand(1).item() < p_dropout:
                    mask[i] = 0.0
            if mask.sum() > 0:
                modality_features = modality_features * mask
        # -------------------------------------------------------------------------- 
        cls_token = self.cls_token.expand(-1, spatial_feat.size(0), -1) 
        x = torch.cat([cls_token, modality_features], dim=0)
        fused_output = self.fusion_transformer(x) 
        cls_output = fused_output[0, :, :]
        out = self.fc_head(cls_output)
        return out
        
def convert_phi_to_degrees(input_path, output_path):
    with open(input_path, 'r') as f:
        lines = f.readlines()
    with open(output_path, 'w') as f:
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 4 and parts[0].isdigit():
                num, theta_str, phi_str, label = parts
                theta = float(theta_str)
                phi_deg = float(phi_str) * (180 / np.pi)
                f.write(f"{num}\t{theta}\t{phi_deg}\t{label}\n")
            else:
                f.write(line)

def load_positions(locs_path):
    temp_path = os.path.splitext(locs_path)[0] + '_deg.locs'
    if os.path.exists(temp_path):
        os.remove(temp_path)
    convert_phi_to_degrees(locs_path, temp_path)
    montage = mne.channels.read_custom_montage(temp_path)
    position = montage.get_positions()['ch_pos']
    channel_names = list(position.keys())
    pos_array = np.array([position[ch] for ch in channel_names])
    norms = np.linalg.norm(pos_array, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        pos_array /= norms[:, np.newaxis]
    max_abs = np.max(np.abs(pos_array[:, :2]))
    positions = []
    for x, y, _ in pos_array:
        col = int(np.round((x + max_abs) / (2 * max_abs) * 8))
        row = int(np.round((max_abs - y) / (2 * max_abs) * 8))
        positions.append((row, col))
    if len(positions) != 62:
        raise ValueError(f"Expected 62 positions, got {len(positions)}")
    os.remove(temp_path)
    return positions

def map_to_9x9(data, positions):
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.float32, device=device) 
    TW, C, F = data.shape 
    if C != 62:
        raise ValueError(f"Expected 62 channels, got {C}")
    
    data_flat = data.permute(1, 0, 2).reshape(C, TW * F)
    
    grid = torch.zeros(9, 9, TW * F, device=data.device, dtype=torch.float32)
    for i, (row, col) in enumerate(positions):
        if 0 <= row <= 8 and 0 <= col <= 8:
            grid[row, col, :] = data_flat[i, :]
        else:
            raise ValueError(f"Position out of bounds: ({row}, {col})")
    return grid

def aggregate_labels(y_discrete, num_samples=218):
    y_discrete = y_discrete.numpy()
    num_labels = len(y_discrete)
    labels_per_sample = num_labels // num_samples
    remainder = num_labels % num_samples
    if remainder != 0:
        y_discrete = np.pad(y_discrete, (0, num_samples * labels_per_sample - num_labels), 
                           mode='edge')
    y_reshaped = y_discrete.reshape(num_samples, labels_per_sample)
    y_agg = mode(y_reshaped, axis=1, keepdims=False)[0].flatten() 
    return torch.tensor(y_agg, dtype=torch.long)

def load_all_data(locs_path):
    print("Starting data loading and feature grid creation...")
    positions = load_positions(locs_path)
    
    all_spatial_grids = []
    all_temporal_raw = []
    all_eye_raw = []
    all_labels = []

    for i in range(1, 21): 
        eeg_var = f"X_de_{i}"
        eye_var = f"X_eye_{i}"
        label_var = f"y_discrete_de_{i}"
        
        try:
            eeg_x_np = globals()[eeg_var] 
            eye_x_np = globals()[eye_var]
            y_discrete_np = globals()[label_var]
        except KeyError:
            print(f"Data for subject {i} not found. Stopping extraction.")
            break

        eeg_x = torch.tensor(eeg_x_np, dtype=torch.float32)
        eye_x = torch.tensor(eye_x_np, dtype=torch.float32)
        y_discrete = torch.tensor(y_discrete_np, dtype=torch.long)

        N_samples = eeg_x.shape[0]
        TW = 16
        C = 62
        F_per_C = 5 
        
        spatial_grids = []
        for sample_idx in range(N_samples):
            sample_data = eeg_x[sample_idx].view(TW, C, F_per_C) 
            grid = map_to_9x9(sample_data, positions)
            spatial_grids.append(grid.cpu())
            
        all_spatial_grids.append(torch.stack(spatial_grids))

        all_temporal_raw.append(eeg_x)
        all_eye_raw.append(eye_x)
        y_agg = aggregate_labels(y_discrete) 
        all_labels.append(y_agg)
        
    all_spatial_grids = torch.cat(all_spatial_grids, dim=0)
    all_temporal_raw = torch.cat(all_temporal_raw, dim=0)
    all_eye_raw = torch.cat(all_eye_raw, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    print(f"Total concatenated samples: {all_spatial_grids.shape[0]}")
    return all_spatial_grids, all_temporal_raw, all_eye_raw, all_labels

def plot_metrics(history, fold_num):
    metrics = ['Loss', 'Accuracy']
    plt.figure(figsize=(12, 5))
    for i, metric in enumerate(metrics):
        plt.subplot(1, 2, i + 1)
        plt.plot(history[f'train_{metric.lower()}'], label=f'Train {metric}')
        plt.plot(history[f'val_{metric.lower()}'], label=f'Validation {metric}')
        plt.title(f'Fold {fold_num}: {metric} vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def train_and_evaluate(model, train_loader, val_loader, test_data, num_epochs):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = OneCycleLR(optimizer, max_lr=8e-4, steps_per_epoch=len(train_loader), epochs=num_epochs)

    history = {'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_accuracy': []}
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20

    for epoch in range(num_epochs):
        model.train()
        total_loss, total_correct, total_samples = 0, 0, 0
        for s_batch, t_batch, e_batch, l_batch in train_loader:
            s_batch, t_batch, e_batch, l_batch = s_batch.to(device), t_batch.to(device), e_batch.to(device), l_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(s_batch, t_batch, e_batch)
            loss = criterion(outputs, l_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item() * s_batch.size(0)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == l_batch).sum().item()
            total_samples += s_batch.size(0)

        train_loss = total_loss / total_samples
        train_accuracy = total_correct / total_samples
        
        model.eval()
        total_val_loss, total_val_correct, total_val_samples = 0, 0, 0
        with torch.no_grad():
            for s_batch, t_batch, e_batch, l_batch in val_loader:
                s_batch, t_batch, e_batch, l_batch = s_batch.to(device), t_batch.to(device), e_batch.to(device), l_batch.to(device)
                outputs = model(s_batch, t_batch, e_batch)
                loss = criterion(outputs, l_batch)
                
                total_val_loss += loss.item() * s_batch.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val_correct += (predicted == l_batch).sum().item()
                total_val_samples += s_batch.size(0)

        val_loss = total_val_loss / total_val_samples
        val_accuracy = total_val_correct / total_val_samples

        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model, 'best_model_k_fold.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
                
    return 'best_model_k_fold.pth', history

# 11. Main Function
if __name__ == '__main__':
    locs_path = "/.../CODE SEED VII/channel_62_pos.locs"
    
    all_spatial_grids, all_temporal_raw, all_eye_raw, all_labels = load_all_data(locs_path)

    num_subjects = 20
    samples_per_subject = 218
    num_samples = all_spatial_grids.shape[0]
    accuracies = []
    fold_reports = []
    fold_conf_matrices = []
    fold_auc_scores = []
    all_y_true = []
    all_y_proba = []
    all_y_pred = []

    K_FOLDS = 5
    BATCH_SIZE = 64
    NUM_EPOCHS = 100
    all_histories = []
    all_model_paths = []
    
    for subject_idx in range(num_subjects):
        print(f"\n--- Starting Fold {subject_idx + 1} (Test Subject {subject_idx + 1}) ---")

        test_start = subject_idx * samples_per_subject
        test_end = (subject_idx + 1) * samples_per_subject
        test_idx = torch.arange(test_start, test_end)
        train_val_idx = torch.cat((torch.arange(0, test_start), torch.arange(test_end, num_samples)))

        test_data = (
            all_spatial_grids[test_idx].to(device),
            all_temporal_raw[test_idx].to(device),
            all_eye_raw[test_idx].to(device),
            all_labels[test_idx].to(device)
        )

        train_val_spatial = all_spatial_grids[train_val_idx]
        train_val_temporal = all_temporal_raw[train_val_idx]
        train_val_eye = all_eye_raw[train_val_idx]
        train_val_labels = all_labels[train_val_idx]

        skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
        best_model_paths = []
        
        for fold_k, (train_k_idx, val_k_idx) in enumerate(skf.split(train_val_spatial, train_val_labels)):
            print(f"  Internal K-Fold {fold_k + 1}/{K_FOLDS}...")

            train_k_set = TensorDataset(
                train_val_spatial[train_k_idx], 
                train_val_temporal[train_k_idx], 
                train_val_eye[train_k_idx], 
                train_val_labels[train_k_idx]
            )
            val_k_set = TensorDataset(
                train_val_spatial[val_k_idx], 
                train_val_temporal[val_k_idx], 
                train_val_eye[val_k_idx], 
                train_val_labels[val_k_idx]
            )
            
            train_loader = DataLoader(train_k_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
            val_loader = DataLoader(val_k_set, batch_size=BATCH_SIZE, shuffle=False)
            
            model = SMTFeatureEncoder(feature_dim=FEATURE_DIM).to(device)
            best_model_file, history = train_and_evaluate(model, train_loader, val_loader, test_data, num_epochs=NUM_EPOCHS)

            unique_model_path = f'best_model_s{subject_idx+1}_k{fold_k+1}.pth'
            os.rename(best_model_file, unique_model_path)
            best_model_paths.append(unique_model_path)
            all_histories.append(history)
            all_model_paths.append(unique_model_path)  
            print(f"  K-Fold {fold_k + 1} Best Validation Loss Reached. Model Saved.")
            plot_metrics(history, subject_idx + 1)

        s_test, t_test, e_test, l_test = test_data
        ensemble_logits = 0.0

        for model_path in best_model_paths:
            model = torch.load(model_path, weights_only=False)
            model.to(device)
            model.eval()

            with torch.no_grad():
                ensemble_logits = ensemble_logits + model(s_test.to(device), t_test.to(device), e_test.to(device))

        _, predicted_ensemble = torch.max(ensemble_logits, 1)       
        final_test_accuracy = (predicted_ensemble == l_test.to(device)).float().mean().item()
        accuracies.append(final_test_accuracy)
        print(f"Fold {subject_idx + 1} (Test Subject {subject_idx + 1}): Final ENSEMBLE Test Accuracy = {final_test_accuracy:.4f}")

        ensemble_probs = F.softmax(ensemble_logits, dim=1)
        y_true = l_test.cpu().numpy()
        y_pred = predicted_ensemble.cpu().numpy()
        y_proba = ensemble_probs.cpu().numpy()
        fold_report = classification_report(y_true, y_pred, output_dict=True)
        fold_reports.append(fold_report)
        print(f"Fold {subject_idx + 1} Classification Report:\n{classification_report(y_true, y_pred)}")

        # Confusion Matrix (numerics)
        fold_cm = confusion_matrix(y_true, y_pred)
        fold_conf_matrices.append(fold_cm)
        plt.figure(figsize=(8, 6))
        sns.heatmap(fold_cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Fold {subject_idx + 1} Confusion Matrix (Numerics)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'cm_fold_{subject_idx + 1}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Confusion Matrix (percentages)
        cm_percent_fold = fold_cm.astype('float') / fold_cm.sum(axis=1)[:, np.newaxis] * 100
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_percent_fold, annot=True, fmt='.2f', cmap='Blues', cbar_kws={'format': '%.0f%%'})
        plt.title(f'Fold {subject_idx + 1} Confusion Matrix (Percentages)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'cm_fold_percent_{subject_idx + 1}.png', dpi=300, bbox_inches='tight')
        plt.show()

        # ROC-AUC Score (macro-averaged)
        fold_auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
        fold_auc_scores.append(fold_auc)
        print(f"Fold {subject_idx + 1} ROC-AUC Score (macro): {fold_auc:.4f}")

        # Collect for overall ROC
        all_y_true.append(y_true)
        all_y_pred.append(y_pred)
        all_y_proba.append(y_proba)
    
    # Compute Average Accuracy
    mean_accuracy = sum(accuracies) / len(accuracies)
    print(f"\n--- Final LOSO-CV Result ---")
    print(f"Average LOSO-CV Test Accuracy: {mean_accuracy:.4f}")

    mean_auc = sum(fold_auc_scores) / len(fold_auc_scores)
    print(f"Average LOSO-CV ROC-AUC Score (macro): {mean_auc:.4f}")

    # Overall Classification Report (averaged)
    avg_precision = np.mean([report['weighted avg']['precision'] for report in fold_reports])
    avg_recall = np.mean([report['weighted avg']['recall'] for report in fold_reports])
    avg_f1 = np.mean([report['weighted avg']['f1-score'] for report in fold_reports])
    print(f"\nOverall (Averaged) Metrics:")
    print(f"Precision (weighted avg): {avg_precision:.4f}")
    print(f"Recall (weighted avg): {avg_recall:.4f}")
    print(f"F1-Score (weighted avg): {avg_f1:.4f}")

    # Overall Confusion Matrix (numbers)
    all_y_true_concat = np.concatenate(all_y_true)
    all_y_pred_concat = np.concatenate(all_y_pred)
    overall_cm = confusion_matrix(all_y_true_concat, all_y_pred_concat)
    plt.figure(figsize=(8, 6))
    sns.heatmap(overall_cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Overall Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('cm_overall_total.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Overall Confusion Matrix (percentages)
    cm_percent = overall_cm.astype('float') / overall_cm.sum(axis=1)[:, np.newaxis] * 100
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percent, annot=True, fmt='.2f', cmap='Blues', cbar_kws={'format': '%.0f%%'}, xticklabels=class_names, yticklabels=class_names)
    plt.title('Overall Confusion Matrix (Percentages)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('cm_overall_percent.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Overall ROC Plot
    all_y_proba_concat = np.concatenate(all_y_proba, axis=0)

    plt.figure(figsize=(10, 8))
    n_classes = NUM_CLASSES
    y_true_bin = label_binarize(all_y_true_concat, classes=range(n_classes))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], all_y_proba_concat[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # macro-average ROC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    plt.plot(fpr["macro"], tpr["macro"], label=f'Macro-average ROC (AUC = {roc_auc["macro"]:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Overall LOSO-CV ROC Curve (Macro-Average)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('roc_overall_macro.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Per-class ROC
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink']
    for i, color in zip(range(n_classes), colors):
        plt.figure(figsize=(8, 6))
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC Class {i} (AUC = {roc_auc[i]:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for Class {i}')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(f'roc_class_{i}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
#  12. Model Summary
model_path = 'best_model_s1_k1.pth'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}. Ensure training ran.")

model = torch.load(model_path, weights_only=False)
model.to(device)
model.eval()

dummy_spatial = all_spatial_grids[0:1].to(device)
dummy_temporal = all_temporal_raw[0:1].to(device)
dummy_eye = all_eye_raw[0:1].to(device)

print(f"Input Shapes: Spatial {dummy_spatial.shape}, Temporal {dummy_temporal.shape}, Eye {dummy_eye.shape}")
assert dummy_spatial.shape == (1, 9, 9, 80), f"Expected [1, 9, 9, 80], got {dummy_spatial.shape}"
assert dummy_temporal.shape == (1, 16, 62, 5), f"Expected [1, 16, 62, 5], got {dummy_temporal.shape}"
assert dummy_eye.shape == (1, 16, 33), f"Expected [1, 16, 33], got {dummy_eye.shape}"

with torch.no_grad():
    test_output = model(dummy_spatial, dummy_temporal, dummy_eye)
    print(f"Output Shape: {test_output.shape} (Expected: [1, {NUM_CLASSES}])")
    assert test_output.shape == (1, NUM_CLASSES), f"Expected [1, {NUM_CLASSES}], got {test_output.shape}"

def get_model_summary_to_csv(model, dummy_inputs, csv_filename='model_summary.csv'):
    summary_data = []
    hook_inputs = {}
    hook_outputs = {}
    
    def register_hooks(module):
        def hook_fn_fn(module, input, output):
            in_shapes = [list(i.shape) for i in input] if isinstance(input, (tuple, list)) else [list(input.shape)]
            if isinstance(output, (tuple, list)):
                out_shapes = [list(o.shape) for o in output if isinstance(o, torch.Tensor)]
            else:
                out_shapes = [list(output.shape)] if isinstance(output, torch.Tensor) else ["N/A"]
            hook_inputs[module] = in_shapes
            hook_outputs[module] = out_shapes
        return hook_fn_fn
    hooks = []
    for name, module in model.named_modules():
        if len(name) > 0 and not name.startswith('.'):
            hook = module.register_forward_hook(register_hooks(module))
            hooks.append(hook)
            
    with torch.no_grad():
        _ = model(*dummy_inputs)
        
    for hook in hooks:
        hook.remove()
        
    summary_data = []
    for name, module in model.named_modules():
        if len(name) > 0 and not name.startswith('.'):
            trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad and len(list(module.children())) == 0)
            in_shape = str(hook_inputs.get(module, ["N/A"]))
            out_shape = str(hook_outputs.get(module, ["N/A"]))
            summary_data.append({
                'Layer (type)': f"{name} ({type(module).__name__})",
                'Input Shape': in_shape,
                'Output Shape': out_shape,
                'Param #': f"{trainable_params:,}" if trainable_params else 0
            })
            
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_size_mb = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad) / (1024**2)
    
    df = pd.DataFrame(summary_data)
    print("=== Model Summary (TensorFlow-like Schema with Shapes) ===")
    print(df.to_string(index=False, justify='left'))
    
    footer_data = [
        {'Layer (type)': 'Total params', 'Input Shape': '', 'Output Shape': '', 'Param #': f"{total_params:,} ({total_params / 1e6:.2f}M)"},
        {'Layer (type)': 'Trainable params', 'Input Shape': '', 'Output Shape': '', 'Param #': f"{total_params:,}"},
        {'Layer (type)': 'Non-trainable params', 'Input Shape': '', 'Output Shape': '', 'Param #': '0'},
        {'Layer (type)': 'Model size', 'Input Shape': '', 'Output Shape': '', 'Param #': f"{total_size_mb:.2f}MB"}
    ]
    footer_df = pd.DataFrame(footer_data)
    final_df = pd.concat([df, footer_df], ignore_index=True)
    final_df.to_csv(csv_filename, index=False)
    print(f"\nModel summary saved to {csv_filename}")
    
    # Print total stats
    print(f"\nTotal params: {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"Trainable params: {total_params:,}")
    print(f"Non-trainable params: 0")
    print(f"Model size: {total_size_mb:.2f}MB")
    print(f"_________________________________________________________________")
    print(f"Total params in model: {total_params:,} ({total_params / 1e6:.2f}M)")
    
    return final_df
    
csv_output = get_model_summary_to_csv(
    model, 
    [dummy_spatial, dummy_temporal, dummy_eye], 
    csv_filename='model_summary_detailed.csv'
)
