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
from data_functions import extract_and_sort_features, filter_frequency_bands, rearrange_eyes, combine_eye_features_by_category
from data_functions import rearrange_labels, prepare_emotion_de, prepare_emotion_eye, make_batches, make_eye_batches

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEATURE_DIM = 256
NUM_CLASSES = 7

# Defining video-to-emotion mapping from the table
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

# Defining emotion-to-integer mapping
emotion_to_int = {
    'Happy': 0, 'Neutral': 1, 'Disgust': 2, 'Sad': 3, 'Anger': 4, 'Fear': 5, 'Surprise': 6
}

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
    # plt.show()

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
