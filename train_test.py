"""
TFN_train_test.py

This module contains the core training and testing pipelines for the ChaosSpec framework.
It handles multimodal data loading, forward/backward propagation, and comprehensive
evaluation of classification metrics (Accuracy, Precision, Sensitivity, Specificity, F1, AUC).

Author: [Your Name]
Date: [Current Date]
"""

import os
import gc
import random
import pickle
import warnings
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import label_binarize

# Local environment imports
from GAI import define_optimizer
from data_loaders import graph_fusion_DatasetLoader
from inter_fusion import MultimodalModel
from utils import count_parameters

warnings.filterwarnings("ignore")


def train(opt, data, device, k):
    """
    Trains the multimodal disease diagnosis model.

    Args:
        opt: Parsed command-line arguments.
        data: Loaded dataset dictionary or object.
        device: Computation device (CPU or GPU).
        k: Current fold number in k-fold cross-validation.

    Returns:
        model: Trained PyTorch model.
        optimizer: Trained optimizer.
        metric_logger: Dictionary recording training and testing metrics per epoch.
    """
    # Ensure reproducibility
    cudnn.deterministic = True
    torch.cuda.manual_seed_all(opt.seed)
    torch.manual_seed(opt.seed)
    random.seed(opt.seed)

    # Initialize main model
    model = MultimodalModel(
        opt.input_size1, opt.input_size2, opt.label_dim, opt.dropout_rate
    ).to(device)

    cross_entropy_loss = nn.CrossEntropyLoss().to(device)

    # Initialize optimizer
    optimizer = define_optimizer(opt, model)

    print(model)
    num_params = count_parameters(model)
    print(f"Number of Trainable Parameters: {num_params / (1024 * 1024):.4f} MB")

    # Initialize data loader
    custom_data_loader = graph_fusion_DatasetLoader(data, split='train')
    train_loader = DataLoader(
        dataset=custom_data_loader, batch_size=len(custom_data_loader), num_workers=0, shuffle=True
    )

    # Initialize metrics logger
    metric_logger = {
        'train': {'loss': [], 'Accuracy': [], 'Precision': [], 'Sensitivity': [], 'Specificity': [], 'F1-score': [],
                  'AUC': []},
        'test': {'loss': [], 'Accuracy': [], 'Precision': [], 'Sensitivity': [], 'Specificity': [], 'F1-score': [],
                 'AUC': []}
    }

    accuracy_best = 0.0
    num_classes = opt.label_dim
    all_classes = list(range(num_classes))

    for epoch in tqdm(range(opt.epoch_count, opt.niter + opt.niter_decay + 1), desc=f"Fold {k} Training"):
        model.train()
        pred_all, label_all = np.array([]), np.array([])
        pred_scores_all = []

        loss_epoch = 0.0
        gc.collect()

        # Iterate over batches
        for batch_idx, (x_ir, x_raman, x_ir_chao, x_raman_chao, label) in enumerate(train_loader):
            # Flatten inputs
            x_ir = x_ir.view(x_ir.size(0), -1).to(device)
            x_raman = x_raman.view(x_raman.size(0), -1).to(device)
            x_ir_chao = x_ir_chao.view(x_ir_chao.size(0), -1).to(device)
            x_raman_chao = x_raman_chao.view(x_raman_chao.size(0), -1).to(device)
            label = label.to(device)

            # Forward pass
            pred = model(x_ir, x_raman, x_ir_chao, x_raman_chao)

            # Compute loss
            loss = cross_entropy_loss(pred, label.squeeze(dim=1).long())
            loss_epoch = loss.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Record predictions and scores
            pred_scores = torch.softmax(pred, dim=1).detach().cpu().numpy()
            pred_labels = np.argmax(pred_scores, axis=1)

            pred_scores_all.append(pred_scores)
            pred_all = np.concatenate((pred_all, pred_labels))
            label_all = np.concatenate((label_all, label.detach().cpu().numpy().reshape(-1)))

        # Aggregate batch predictions
        if pred_scores_all:
            pred_scores_all = np.concatenate(pred_scores_all, axis=0)
        else:
            pred_scores_all = np.empty((0, num_classes))

        # Evaluate and log metrics at specified intervals or at the last epoch
        if opt.measure or epoch == (opt.niter + opt.niter_decay - 1):
            loss_epoch /= train_loader.batch_size

            y_pred = pred_all
            y_true = label_all
            y_pred_scores = pred_scores_all

            # Calculate Macro-Average Metrics
            acc_train = accuracy_score(y_true, y_pred)
            prec_train = precision_score(y_true, y_pred, average='macro', zero_division=0)
            sens_train = recall_score(y_true, y_pred, average='macro', zero_division=0)
            f1_train = f1_score(y_true, y_pred, average='macro', zero_division=0)

            # Calculate Specificity
            try:
                cm = confusion_matrix(y_true, y_pred, labels=all_classes)
                all_specificity = []
                for i in range(num_classes):
                    if i >= len(cm):
                        all_specificity.append(0.0)
                        continue
                    tp = cm[i, i]
                    fp = cm[:, i].sum() - tp
                    fn = cm[i, :].sum() - tp
                    tn = cm.sum() - (tp + fp + fn)
                    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0.0
                    all_specificity.append(specificity)
                spec_train = np.mean(all_specificity)
            except Exception:
                spec_train = 0.0

            # Calculate AUC (One-vs-Rest)
            try:
                y_true_binarized = label_binarize(y_true, classes=all_classes)
                if num_classes == 2 and y_true_binarized.shape[1] == 1:
                    y_true_binarized = np.hstack((1 - y_true_binarized, y_true_binarized))
                auc_train = roc_auc_score(y_true_binarized, y_pred_scores, multi_class='ovr', average='macro')
            except ValueError:
                auc_train = 0.0

            # Evaluate on the test set
            (loss_test, acc_test, prec_test, sens_test, spec_test, f1_test, auc_test,
             pred_test, _) = Test(opt, model, data, 'test', device, num_classes, all_classes)

            # Log metrics
            metric_logger['train']['loss'].append(loss_epoch)
            metric_logger['train']['Accuracy'].append(acc_train)
            metric_logger['train']['Precision'].append(prec_train)
            metric_logger['train']['Sensitivity'].append(sens_train)
            metric_logger['train']['Specificity'].append(spec_train)
            metric_logger['train']['F1-score'].append(f1_train)
            metric_logger['train']['AUC'].append(auc_train)

            metric_logger['test']['loss'].append(loss_test)
            metric_logger['test']['Accuracy'].append(acc_test)
            metric_logger['test']['Precision'].append(prec_test)
            metric_logger['test']['Sensitivity'].append(sens_test)
            metric_logger['test']['Specificity'].append(spec_test)
            metric_logger['test']['F1-score'].append(f1_test)
            metric_logger['test']['AUC'].append(auc_test)

            # Save test predictions for current epoch
            save_path = os.path.join(opt.results, opt.exp_name, opt.model_name, f'{k}_fold')
            with open(os.path.join(save_path, f'{opt.model_name}_{epoch}_pred_test.pkl'), 'wb') as f:
                pickle.dump(pred_test, f)

            if acc_test > accuracy_best:
                accuracy_best = acc_test

    return model, optimizer, metric_logger


def Test(opt, model, data, split, device, num_classes, all_classes):
    """
    Evaluates the model on a specified dataset split.

    Args:
        opt: Parsed command-line arguments.
        model: Trained PyTorch model.
        data: Loaded dataset dictionary or object.
        split: String indicating the data split (e.g., 'test' or 'train').
        device: Computation device.
        num_classes: Total number of classes.
        all_classes: List of all class indices.

    Returns:
        Tuple containing loss, macro-averaged metrics (Accuracy, Precision,
        Sensitivity, Specificity, F1, AUC), raw prediction data, and per-class metrics.
    """
    if split == 'test':
        model.eval()

    custom_data_loader = graph_fusion_DatasetLoader(data, split)
    test_loader = DataLoader(
        dataset=custom_data_loader, batch_size=len(custom_data_loader), num_workers=0, shuffle=True
    )

    pred_all, label_all = np.array([]), np.array([])
    pred_scores_all = []
    loss_test = 0.0
    cross_entropy_loss = nn.CrossEntropyLoss().to(device)

    total_inference_time = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, (x_ir, x_raman, x_ir_chao, x_raman_chao, label) in enumerate(test_loader):
            # Flatten inputs
            x_ir = x_ir.view(x_ir.size(0), -1).to(device)
            x_raman = x_raman.view(x_raman.size(0), -1).to(device)
            x_ir_chao = x_ir_chao.view(x_ir_chao.size(0), -1).to(device)
            x_raman_chao = x_raman_chao.view(x_raman_chao.size(0), -1).to(device)
            label = label.to(device)

            # Measure inference time
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

            # Forward pass
            pred = model(x_ir, x_raman, x_ir_chao, x_raman_chao)

            end_event.record()
            torch.cuda.synchronize()
            total_inference_time += start_event.elapsed_time(end_event)
            total_samples += x_ir.size(0)

            # Compute loss
            loss_cox = cross_entropy_loss(pred, label.squeeze(dim=1).long())
            loss_test += loss_cox.item()

            # Record predictions and scores
            pred_scores = torch.softmax(pred, dim=1).detach().cpu().numpy()
            pred_labels = np.argmax(pred_scores, axis=1)

            pred_all = np.concatenate((pred_all, pred_labels))
            pred_scores_all.append(pred_scores)
            label_all = np.concatenate((label_all, label.detach().cpu().numpy().reshape(-1)))

    loss_test /= test_loader.batch_size

    if pred_scores_all:
        pred_scores_all = np.concatenate(pred_scores_all, axis=0)
    else:
        pred_scores_all = np.empty((0, num_classes))

    y_pred = pred_all
    y_true = label_all
    y_pred_scores = pred_scores_all

    # ------------------------------------------------------------------
    # Macro-Average Metrics Calculation
    # ------------------------------------------------------------------
    acc_test = accuracy_score(y_true, y_pred)
    prec_test = precision_score(y_true, y_pred, average='macro', zero_division=0)
    sens_test = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_test = f1_score(y_true, y_pred, average='macro', zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=all_classes)

    # Specificity Calculation
    try:
        all_specificity = []
        for i in range(num_classes):
            if i >= len(cm):
                all_specificity.append(0.0)
                continue
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - (tp + fp + fn)
            spec = tn / (tn + fp) if (tn + fp) != 0 else 0.0
            all_specificity.append(spec)
        spec_test = np.mean(all_specificity)
    except Exception:
        spec_test = 0.0

    # AUC Calculation
    try:
        y_true_binarized = label_binarize(y_true, classes=all_classes)
        if num_classes == 2 and y_true_binarized.shape[1] == 1:
            y_true_binarized = np.hstack((1 - y_true_binarized, y_true_binarized))
        auc_test = roc_auc_score(y_true_binarized, y_pred_scores, multi_class='ovr', average='macro')
    except ValueError:
        auc_test = 0.0

    # ------------------------------------------------------------------
    # Per-Class Metrics Calculation
    # ------------------------------------------------------------------
    prec_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    sens_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    spec_per_class = []
    acc_per_class = []

    for i in range(num_classes):
        try:
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - (tp + fp + fn)

            spec = tn / (tn + fp) if (tn + fp) != 0 else 0.0
            acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0.0

            spec_per_class.append(spec)
            acc_per_class.append(acc)
        except Exception:
            spec_per_class.append(0.0)
            acc_per_class.append(0.0)

    try:
        auc_per_class = roc_auc_score(y_true_binarized, y_pred_scores, multi_class='ovr', average=None)
    except ValueError:
        auc_per_class = [0.0] * num_classes

    per_class_results = {
        'Precision': prec_per_class,
        'Sensitivity': sens_per_class,
        'Specificity': spec_per_class,
        'F1': f1_per_class,
        'AUC': auc_per_class,
        'Accuracy': acc_per_class
    }

    pred_test_data = [pred_all, label_all, pred_scores_all]

    return loss_test, acc_test, prec_test, sens_test, spec_test, f1_test, auc_test, pred_test_data, per_class_results