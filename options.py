"""
options.py

This module defines the configuration parser for the ChaosSpec framework. 
It handles the loading and saving of hyperparameters, device configurations, 
and input dimensions for the multimodal spectral learning tasks.

Author: [Your Name]
Date: [Current Date]
"""

import argparse
import os
import torch


def parse_args():
    """
    Parse command-line arguments for training and testing.

    Returns:
        opt (argparse.Namespace): Parsed configuration options.
    """
    parser = argparse.ArgumentParser(description="ChaosSpec Multimodal Framework Options")

    # ------------------------------------------------------------------
    # Basic Project Configurations
    # ------------------------------------------------------------------
    parser.add_argument('--model_save', type=str, default='model_save', help='Directory to save models.')
    parser.add_argument('--results', type=str, default='Mresults', help='Directory to save experimental results.')
    parser.add_argument('--exp_name', type=str, default='1007',
                        help='Name of the project. Determines the storage structure.')
    parser.add_argument('--model_name', type=str, default='', help='Name of the specific model architecture.')
    parser.add_argument('--gpu_ids', type=str, default='0', help='GPU ids: e.g., 0 or 0,1,2. Use -1 for CPU only.')

    # ------------------------------------------------------------------
    # Data & Architecture Dimensions
    # ------------------------------------------------------------------
    parser.add_argument('--input_size', type=int, default=256, help='Input size for generic feature vectors.')
    parser.add_argument('--input_size1', type=int, default=1670,
                        help='Dimension of the first modality (Infrared spectra 1660 + chaotic features 10).')
    parser.add_argument('--input_size2', type=int, default=322,
                        help='Dimension of the second modality (Raman spectra 312 + chaotic features 10).')
    parser.add_argument('--label_dim', type=int, default=4,
                        help='Size of output classes (e.g., 4 for lethal chest pain classification).')

    # ------------------------------------------------------------------
    # Training Hyperparameters
    # ------------------------------------------------------------------
    parser.add_argument('--seed', type=int, default=64, help='Random seed to ensure reproducibility.')
    parser.add_argument('--epoch_count', type=int, default=1, help='Starting epoch number.')
    parser.add_argument('--niter', type=int, default=0, help='Number of epochs at the starting learning rate.')
    parser.add_argument('--niter_decay', type=int, default=120,
                        help='Number of epochs to linearly decay learning rate to zero (120 is optimal for the cardiovascular dataset).')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and testing.')
    parser.add_argument('--dropout_rate', type=float, default=0.0,
                        help='Dropout rate (0.0 - 0.25). Increase to prevent overfitting.')

    # ------------------------------------------------------------------
    # Optimizer & Regularization
    # ------------------------------------------------------------------
    parser.add_argument('--optimizer_type', type=str, default='adam', help='Type of optimizer (e.g., adam, adamw).')
    parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate (Default: 5e-4 for Adam).')
    parser.add_argument('--lr_policy', type=str, default='linear', help='Learning rate scheduling policy.')
    parser.add_argument('--beta1', type=float, default=0.9, help='Momentum term beta1 for Adam optimizer.')
    parser.add_argument('--beta2', type=float, default=0.999, help='Momentum term beta2 for Adam optimizer.')
    parser.add_argument('--weight_decay', type=float, default=3e-8, help='L2 Regularization penalty. Default: 3e-8.')
    parser.add_argument('--lambda_reg', type=float, default=2e-6, help='Regularization lambda.')

    # ------------------------------------------------------------------
    # Evaluation & Logging
    # ------------------------------------------------------------------
    parser.add_argument('--measure', type=int, default=1, help='Flag to enable metrics calculation during training.')
    parser.add_argument('--verbose', type=int, default=1, help='Verbosity level.')

    # Parse args, print them, and configure devices
    opt = parser.parse_known_args()[0]
    print_options(parser, opt)
    opt = parse_gpuids(opt)

    return opt


def print_options(parser, opt):
    """
    Print and save options.

    It will print both current options and default values (if different).
    It will also save options into a text file at [model_save/exp_name/model_name/train_opt.txt].

    Args:
        parser (argparse.ArgumentParser): The defined argument parser.
        opt (argparse.Namespace): The parsed options.
    """
    message = ''
    message += '----------------- Options ---------------\n'

    # vars(opt) returns the __dict__ of the namespace. items() gets key-value pairs.
    # The pairs are sorted alphabetically by key.
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

    # Save to the disk
    expr_dir = os.path.join(opt.model_save, opt.exp_name, opt.model_name)
    mkdirs(expr_dir)
    file_name = os.path.join(expr_dir, '{}_opt.txt'.format('train'))
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')


def parse_gpuids(opt):
    """
    Parse GPU IDs from string to a list of integers and set the active device.

    Args:
        opt (argparse.Namespace): The parsed options.

    Returns:
        opt (argparse.Namespace): Options with updated gpu_ids list.
    """
    str_ids = opt.gpu_ids.split(',')
    opt.gpu_ids = []
    for str_id in str_ids:
        id_int = int(str_id)
        if id_int >= 0:
            opt.gpu_ids.append(id_int)

    if len(opt.gpu_ids) > 0:
        torch.cuda.set_device(opt.gpu_ids[0])

    return opt


def mkdirs(paths):
    """
    Create empty directories if they don't exist.

    Args:
        paths (str or list of str): A single path or a list of directory paths.
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """
    Create a single empty directory if it does not exist.

    Args:
        path (str): A single directory path.
    """
    if not os.path.exists(path):
        os.makedirs(path)