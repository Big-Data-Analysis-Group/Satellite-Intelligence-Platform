"""Utility functions."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import seaborn as sns


def compute_metrics(y_true, y_pred):
    """Compute classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary of metrics
    """
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'report': classification_report(y_true, y_pred)
    }
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None):
    """Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                xticklabels=class_names, yticklabels=class_names)
    
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_training_history(history, save_path=None):
    """Plot training history.
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', etc.
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training History - Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    if 'train_acc' in history and 'val_acc' in history:
        axes[1].plot(history['train_acc'], label='Train Acc')
        axes[1].plot(history['val_acc'], label='Val Acc')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training History - Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def visualize_patches(patches, indices=None, titles=None, figsize=(12, 8)):
    """Visualize image patches.
    
    Args:
        patches: List of patches (each is H, W, C array)
        indices: Indices of patches to visualize
        titles: Titles for each patch
        figsize: Figure size
    """
    if indices is None:
        indices = range(min(4, len(patches)))
    
    fig, axes = plt.subplots(1, len(indices), figsize=figsize)
    if len(indices) == 1:
        axes = [axes]
    
    for ax, idx in zip(axes, indices):
        patch = patches[idx]
        
        # Use RGB channels (B04, B03, B02)
        if patch.shape[2] >= 3:
            rgb = patch[..., [2, 1, 0]]  # Reorder to RGB
            rgb = np.clip(rgb / 10000.0, 0, 1)  # Normalize
            ax.imshow(rgb)
        else:
            ax.imshow(patch[..., 0], cmap='gray')
        
        ax.axis('off')
        if titles:
            ax.set_title(titles[idx])
    
    plt.tight_layout()
    plt.show()


class EarlyStopping:
    """Early stopping callback."""
    
    def __init__(self, patience: int = 3, min_delta: float = 0.0):
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs with no improvement to wait
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
    
    def __call__(self, val_loss: float) -> bool:
        """Check if training should stop.
        
        Args:
            val_loss: Validation loss
            
        Returns:
            True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False
