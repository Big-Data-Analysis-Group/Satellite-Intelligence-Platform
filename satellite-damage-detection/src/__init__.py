"""Satellite damage detection package."""

from .data import SentinelDataLoader, apply_cloud_mask, normalize_bands, create_patches
from .features import SpectralIndices, compute_patch_features, compute_change_index
from .model import DamageClassifier, ChangeDetectionModel, create_model
from .train import Trainer
from .utils import compute_metrics, plot_confusion_matrix, plot_training_history, EarlyStopping

__version__ = "0.1.0"
__author__ = "Jonathan"

__all__ = [
    'SentinelDataLoader',
    'apply_cloud_mask',
    'normalize_bands',
    'create_patches',
    'SpectralIndices',
    'compute_patch_features',
    'compute_change_index',
    'DamageClassifier',
    'ChangeDetectionModel',
    'create_model',
    'Trainer',
    'compute_metrics',
    'plot_confusion_matrix',
    'plot_training_history',
    'EarlyStopping',
]