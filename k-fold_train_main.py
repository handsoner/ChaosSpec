# -*- coding: utf-8 -*-
from IPython.core.tests.test_display import test_display_id

from MFM1.utils import calculate_model_efficiency
from train_test import train, Test
import os

import logging
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
import torch
import pandas as pd
# Env
from data_loaders import *
from options import parse_args
import scipy.io as sio

# --- [导入依赖] ---
from itertools import cycle
from sklearn.metrics import roc_curve, auc
# 修正：label_binarize 必须从 preprocessing 导入
from sklearn.preprocessing import label_binarize
from numpy import interp

opt = parse_args()  # 解析参数
device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
print("Using device:", device)
print(torch.cuda.device_count())  # 打印gpu数量
if not os.path.exists(os.path.join(opt.model_save, opt.exp_name, opt.model_name)):
    os.makedirs(os.path.join(opt.model_save, opt.exp_name, opt.model_name))

results = []

loss_train_results = []
loss_test_results = []

average_acc_results = []
average_Precision_results = []
average_Sensitivity_results = []
average_Specificity_results = []
average_F1_score_results = []
average_auc_results = []


def train_display(trian_loss, trian_acc, trian_precision, train_sensitivity, train_specificity, train_f1_score,
                  train_auc):
    print("\nLoss epoch on train set is {:.4f}".format(trian_loss))
    print("Accuracy score on train set is {:.4f}".format(trian_acc))
    print("Precision on train set is {:.4f}".format(trian_precision))
    print("Sensitivity on train set is {:.4f}".format(train_sensitivity))
    print("Specificity on train set is {:.4f}".format(train_specificity))
    print("F1-score on train set is {:.4f}".format(train_f1_score))
    print("AUC on train set is {:.4f}".format(train_auc))


def Test_display(test_loss, test_acc, test_precision, test_sensitivity, test_specificity, test_f1_score, test_auc):
    print("\nLoss epoch on test set is {:.4f}".format(test_loss))
    print("Accuracy score on test set is {:.4f}".format(test_acc))
    print("Precision on test set is {:.4f}".format(test_precision))
    print("Sensitivity on test set is {:.4f}".format(test_sensitivity))
    print("Specificity on test set is {:.4f}".format(test_specificity))
    print("F1-score on test set is {:.4f}".format(test_f1_score))
    print("AUC on test set is {:.4f}".format(test_auc))


### [绘图函数部分] ###

def plot_combined_curves(train_loss, test_loss, train_acc, test_acc, k, save_path):
    """
    在一张图上绘制训练和测试的 Loss 与 Accuracy。
    左轴(红色): Loss (实线=Train, 虚线=Test)
    右轴(蓝色): Accuracy (实线=Train, 虚线=Test)
    """
    fig, ax1 = plt.subplots(figsize=(12, 8))

    epochs = range(1, len(train_loss) + 1)

    # --- 左侧 Y 轴 (Loss) ---
    color_loss = 'tab:red'
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', color=color_loss, fontweight='bold', fontsize=12)

    # 绘制 Loss 曲线
    l1 = ax1.plot(epochs, train_loss, color=color_loss, linestyle='-', linewidth=2, label='Train Loss')
    l2 = ax1.plot(epochs, test_loss, color=color_loss, linestyle='--', linewidth=2, label='Test Loss')

    ax1.tick_params(axis='y', labelcolor=color_loss)
    ax1.grid(True, linestyle='--', alpha=0.3)

    # --- 右侧 Y 轴 (Accuracy) ---
    ax2 = ax1.twinx()  # 共享 X 轴
    color_acc = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color_acc, fontweight='bold', fontsize=12)

    # 绘制 Accuracy 曲线
    l3 = ax2.plot(epochs, train_acc, color=color_acc, linestyle='-', linewidth=2, label='Train Accuracy')
    l4 = ax2.plot(epochs, test_acc, color=color_acc, linestyle='--', linewidth=2, label='Test Accuracy')

    ax2.tick_params(axis='y', labelcolor=color_acc)
    # 设置精度范围通常在 0-1 之间，如果有需要可以注释掉下面这行
    # ax2.set_ylim(0, 1.05)

    # --- 标题与图例 ---
    plt.title(f'Fold {k}: Training & Test Metrics', fontsize=16)

    # 合并图例 (将两个轴的图例合并展示)
    lines = l1 + l2 + l3 + l4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', fontsize=10, framealpha=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()
    print(f"Combined metrics plot saved to {save_path}")


def plot_three_classfier_AUC(pred_auc, label_all, k, n_classes, save_path):
    # 确保标签是整数类型
    label_all = label_all.astype(int)

    classes_list = list(range(n_classes))
    # 二值化标签
    y_true = label_binarize(label_all, classes=classes_list)

    # 处理二分类特殊情况 (如果 n_classes=2 但 y_true 只有一列)
    if n_classes == 2 and y_true.shape[1] == 1:
        y_true = np.hstack((1 - y_true, y_true))

    y_score = pred_auc

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # 计算每一类的 ROC
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 计算 micro-average ROC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # 计算 macro-average ROC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # 开始绘图
    lw = 2
    plt.figure(figsize=(10, 8))

    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Test AUC of ' + str(k) + '-fold ', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"AUC plot saved to {save_path}")


### k-fold loop
# 循环读取data1-data5
for k in range(1, 6):
    print("*******************************************")
    print("************** SPLIT (%d/%d) **************" % (k, 5))
    print("*******************************************")

    data = sio.loadmat(r'data/1new/xinxueguan/全部光谱/data' + str(k) + '.mat')
    # data = sio.loadmat(r'data/1new/banmo/data' + str(k) + '.mat')
    # data = sio.loadmat(r'data/1new/Cancer_ALL/data' + str(k) + '.mat')

    ### 创建文件夹存储结果
    if not os.path.exists(os.path.join(opt.results, opt.exp_name, opt.model_name, '%d_fold' % (k))):
        os.makedirs(os.path.join(opt.results, opt.exp_name, opt.model_name, '%d_fold' % (k)))

    ### 1 Trains Model
    model, optimizer, metric_logger = train(opt, data, device, k)

    epochs_list = range(opt.epoch_count, opt.niter + opt.niter_decay + 1)

    ### [NEW PLOTS] ###
    ### --- 绘制合并的 Loss 和 Accuracy 曲线 (Train + Test) --- ###
    save_dir = os.path.join(opt.results, opt.exp_name, opt.model_name, '%d_fold' % k)

    # 提取 Loss 和 Accuracy
    train_loss = metric_logger['train']['loss']
    train_acc = metric_logger['train']['Accuracy']
    test_loss = metric_logger['test']['loss']
    test_acc = metric_logger['test']['Accuracy']

    # 调用合并绘图函数
    plot_combined_curves(
        train_loss, test_loss,
        train_acc, test_acc,
        k,
        os.path.join(save_dir, 'combined_metrics_fold%d.png' % k)
    )
    ### [END PLOTS] ###

    ### 2 Evalutes Train + Test Error
    loss_train, Accuracy_train, Precision_train, sensitivity_train, specificity_train, f1_score_train, AUC_train, pred_train, per_class_results_train = Test(
        opt, model, data, 'train', device, opt.label_dim, list(range(opt.label_dim)))
    loss_test, Accuracy_test, Precision_test, sensitivity_test, specificity_test, f1_score_test, AUC_test, pred_test, per_class_results_test = Test(
        opt, model, data, 'test', device, opt.label_dim, list(range(opt.label_dim)))

    # [NEW] 绘制 AUC
    pred_scores = pred_test[2]
    label_all = pred_test[1]
    plot_three_classfier_AUC(pred_scores, label_all, k, opt.label_dim,
                             os.path.join(save_dir, 'test_AUC_fold%d.png' % k))

    train_display(loss_train, Accuracy_train, Precision_train, sensitivity_train, specificity_train, f1_score_train,
                  AUC_train)
    Test_display(loss_test, Accuracy_test, Precision_test, sensitivity_test, specificity_test, f1_score_test, AUC_test)
    logging.info(
        "[Final] Apply model to training set: Accuracy: %.5f, Precision: %.5f, Sensitivity: %.5f, Specificity: %.5f, F1_score: %.5f, AUC: %.5f" % (
            Accuracy_train, Precision_train, sensitivity_train, specificity_train, f1_score_train, AUC_train))

    # [NEW] 打印测试集的每个类别的详细结果
    print("\n" + "=" * 20 + " Per-Class Test Results " + "=" * 20)
    print(
        f"{'Class':<10} | {'Prec.':<10} | {'Sens.':<10} | {'Spec.':<10} | {'F1':<10} | {'AUC':<10} | {'Acc(OvR)':<10}")
    print("-" * 85)

    for i in range(opt.label_dim):
        print(f"Class {i:<4} | "
              f"{per_class_results_test['Precision'][i]:<10.4f} | "
              f"{per_class_results_test['Sensitivity'][i]:<10.4f} | "
              f"{per_class_results_test['Specificity'][i]:<10.4f} | "
              f"{per_class_results_test['F1'][i]:<10.4f} | "
              f"{per_class_results_test['AUC'][i]:<10.4f} | "
              f"{per_class_results_test['Accuracy'][i]:<10.4f}")
    print("=" * 60 + "\n")

    loss_train_results.append(loss_train)
    loss_test_results.append(loss_test)

    average_acc_results.append(Accuracy_test)
    average_Precision_results.append(Precision_test)
    average_Sensitivity_results.append(sensitivity_test)
    average_Specificity_results.append(specificity_test)
    average_F1_score_results.append(f1_score_test)
    average_auc_results.append(AUC_test)
    ## 3 Saves Model
    if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
        model_state_dict = model.state_dict()
    # else:
    #     model_state_dict = model.state_dict()
    torch.save({
        'split': k,
        'opt': opt,
        'epoch': opt.niter + opt.niter_decay,
        'data': data,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metric_logger},
        os.path.join(opt.model_save, opt.exp_name, opt.model_name, '%s_%d.pt' % (opt.model_name, k)))

    print()
    if k == 5:
        calculate_model_efficiency(model, data, device)

    pickle.dump(pred_train, open(os.path.join(opt.results, opt.exp_name, opt.model_name, '%d_fold' % (k),
                                              '%s_%dpred_train.pkl' % (opt.model_name, k)), 'wb'))
    pickle.dump(pred_test, open(os.path.join(opt.results, opt.exp_name, opt.model_name, '%d_fold' % (k),
                                             '%s_%dpred_test.pkl' % (opt.model_name, k)), 'wb'))

np.savetxt(opt.results + "split_Accuracy_tes_results.csv", np.round(np.array(average_acc_results), 4), delimiter=",")
np.savetxt(opt.results + "split_loss_train_results.csv", np.round(np.array(loss_train_results), 4), delimiter=",")
np.savetxt(opt.results + "split_loss_test_results.csv", np.round(np.array(loss_test_results), 4), delimiter=",")
np.savetxt(opt.results + "split_AUC_test_results.csv", np.round(np.array(average_auc_results), 4), delimiter=",")

# 显示每折的测试集结果
print('\nSplit loss_train Results:', np.round(np.array(loss_train_results), 4))
print('Split loss_test Results:', np.round(np.array(loss_test_results), 4))

print('\nSplit Accuracy_test Results:', np.round(np.array(average_acc_results), 4))
print('Split Precision_test Results:', np.round(np.array(average_Precision_results), 4))
print('Split Sensitivity_test Results:', np.round(np.array(average_Sensitivity_results), 4))
print('Split Specificity_test Results:', np.round(np.array(average_Specificity_results), 4))
print('Split F1_score_test Results:', np.round(np.array(average_F1_score_results), 4))
print('Split AUC_test Results:', np.round(np.array(average_auc_results), 4))

# 五折取平均
print("\nAverage_Accuracy_test_results: ", np.round(np.array(average_acc_results).mean(), 4))
print("Average_Precision_test_results: ", np.round(np.array(average_Precision_results).mean(), 4))
print("Average_Sensitivity_test_results: ", np.round(np.array(average_Sensitivity_results).mean(), 4))
print("Average_Specificity_test_results: ", np.round(np.array(average_Specificity_results).mean(), 4))
print("Average_F1_score_test_results: ", np.round(np.array(average_F1_score_results).mean(), 4))
print("Average_AUC_test_results: ", np.round(np.array(average_auc_results).mean(), 4))