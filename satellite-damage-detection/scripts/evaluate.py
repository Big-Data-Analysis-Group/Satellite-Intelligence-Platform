"""Evaluation script for damage detection model."""

import argparse
import torch
from pathlib import Path
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import DamageClassifier, compute_metrics, plot_confusion_matrix


def main(args):
    """Main evaluation function."""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    model = DamageClassifier(num_classes=args.num_classes, input_channels=args.input_channels)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded from {args.model_path}")
    
    # TODO: Load test data
    # test_loader = DataLoader(...)
    
    # Evaluate
    print("Evaluating...")
    
    all_preds = []
    all_labels = []
    
    # with torch.no_grad():
    #     for images, labels in test_loader:
    #         images = images.to(device)
    #         outputs = model(images)
    #         _, preds = torch.max(outputs, 1)
    #         all_preds.extend(preds.cpu().numpy())
    #         all_labels.extend(labels.numpy())
    
    # all_preds = np.array(all_preds)
    # all_labels = np.array(all_labels)
    
    # # Compute metrics
    # metrics = compute_metrics(all_labels, all_preds)
    # print("\nMetrics:")
    # print(f"Precision: {metrics['precision']:.4f}")
    # print(f"Recall: {metrics['recall']:.4f}")
    # print(f"F1 Score: {metrics['f1']:.4f}")
    # print("\n" + metrics['report'])
    
    # # Plot confusion matrix
    # plot_confusion_matrix(all_labels, all_preds, save_path=args.save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate damage detection model')
    parser.add_argument('--model-path', type=str, required=True, help='Path to model weights')
    parser.add_argument('--num-classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--input-channels', type=int, default=6, help='Number of input channels')
    parser.add_argument('--save-path', type=str, default='results/plots/confusion_matrix.png', help='Path to save results')
    
    args = parser.parse_args()
    main(args)
