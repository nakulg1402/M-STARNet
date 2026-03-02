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
from model import load_all_data, SMTFeatureEncoder, train_and_evaluate, plot_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEATURE_DIM = 256
NUM_CLASSES = 7

if __name__ == '__main__':
    locs_path = "/.../channel_62_pos.locs"
    
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
        # plt.show()
        
        # Confusion Matrix (percentages)
        cm_percent_fold = fold_cm.astype('float') / fold_cm.sum(axis=1)[:, np.newaxis] * 100
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_percent_fold, annot=True, fmt='.2f', cmap='Blues', cbar_kws={'format': '%.0f%%'})
        plt.title(f'Fold {subject_idx + 1} Confusion Matrix (Percentages)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'cm_fold_percent_{subject_idx + 1}.png', dpi=300, bbox_inches='tight')
        # plt.show()

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
    sns.heatmap(overall_cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Overall Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('cm_overall_total.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Overall Confusion Matrix (percentages)
    cm_percent = overall_cm.astype('float') / overall_cm.sum(axis=1)[:, np.newaxis] * 100
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percent, annot=True, fmt='.2f', cmap='Blues', cbar_kws={'format': '%.0f%%'})
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
