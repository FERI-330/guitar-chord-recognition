"""
Model evaluation module.
Implements robust multiclass evaluation metrics and cross-validation.
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    roc_auc_score
)
from sklearn.preprocessing import LabelBinarizer
import seaborn as sns

logger = logging.getLogger('guitar_chord_recognition')


class ModelEvaluator:
    """Evaluates model performance with comprehensive multiclass metrics."""

    def __init__(self, classes: List[str], save_plots: bool = True, plot_dir: str = "plots"):
        """
        Initialize ModelEvaluator.

        Args:
            classes: List of class labels
            save_plots: Whether to save evaluation plots
            plot_dir: Directory to save plots
        """
        self.classes = classes
        self.save_plots = save_plots
        self.plot_dir = plot_dir
        self.n_classes = len(classes)

    def evaluate_multiclass(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute comprehensive multiclass metrics.

        Args:
            y_true: True labels (encoded)
            y_pred: Predicted labels (encoded)
            y_pred_proba: Predicted probabilities (n_samples, n_classes)

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)

        # Macro-averaged metrics (unweighted mean across classes)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)

        # Weighted metrics
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)

        # ROC-AUC (if probabilities provided)
        if y_pred_proba is not None:
            try:
                # One-vs-rest AUC
                metrics['roc_auc_ovr'] = roc_auc_score(
                    y_true, y_pred_proba, multi_class='ovr', zero_division=0
                )
                # One-vs-one AUC
                metrics['roc_auc_ovo'] = roc_auc_score(
                    y_true, y_pred_proba, multi_class='ovo', zero_division=0
                )
            except Exception as e:
                logger.warning(f"Could not compute ROC-AUC: {e}")

        logger.info(f"Multiclass metrics: Accuracy={metrics['accuracy']:.4f}, "
                   f"F1(macro)={metrics['f1_macro']:.4f}")

        return metrics

    def evaluate_per_class(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute per-class evaluation metrics.

        Args:
            y_true: True labels (encoded)
            y_pred: Predicted labels (encoded)
            y_pred_proba: Predicted probabilities (n_samples, n_classes)

        Returns:
            Dictionary with per-class metrics
        """
        per_class_metrics = {}

        # Per-class precision, recall, F1
        y_pred_binary = np.eye(self.n_classes)[y_pred]
        y_true_binary = np.eye(self.n_classes)[y_true]

        for i, class_name in enumerate(self.classes):
            metrics = {
                'precision': precision_score(y_true_binary[:, i], y_pred_binary[:, i], zero_division=0),
                'recall': recall_score(y_true_binary[:, i], y_pred_binary[:, i], zero_division=0),
                'f1': f1_score(y_true_binary[:, i], y_pred_binary[:, i], zero_division=0),
            }

            # Per-class AUC if probabilities provided
            if y_pred_proba is not None:
                try:
                    metrics['auc'] = roc_auc_score(
                        y_true_binary[:, i], y_pred_proba[:, i], zero_division=0
                    )
                except Exception as e:
                    logger.warning(f"Could not compute AUC for class {class_name}: {e}")
                    metrics['auc'] = 0.0

            per_class_metrics[class_name] = metrics

        return per_class_metrics

    def get_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """
        Generate detailed classification report.

        Args:
            y_true: True labels (encoded)
            y_pred: Predicted labels (encoded)

        Returns:
            Classification report string
        """
        return classification_report(
            y_true, y_pred,
            target_names=self.classes,
            zero_division=0
        )

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        filename: Optional[str] = None
    ) -> None:
        """
        Plot and optionally save confusion matrix.

        Args:
            y_true: True labels (encoded)
            y_pred: Predicted labels (encoded)
            filename: Optional filename to save plot
        """
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.classes,
            yticklabels=self.classes
        )
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')

        if self.save_plots and filename:
            plt.savefig(f"{self.plot_dir}/{filename}", dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {self.plot_dir}/{filename}")

        plt.close()

    def plot_roc_curves(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        filename: Optional[str] = None
    ) -> None:
        """
        Plot ROC curves for all classes and micro-average.

        Args:
            y_true: True labels (encoded)
            y_pred_proba: Predicted probabilities (n_samples, n_classes)
            filename: Optional filename to save plot
        """
        # One-hot encode y_true
        y_true_bin = np.eye(self.n_classes)[y_true]

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        # Per-class ROC curves
        for i in range(self.n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Micro-average
        fpr['micro'], tpr['micro'], _ = roc_curve(y_true_bin.ravel(), y_pred_proba.ravel())
        roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

        # Plot
        plt.figure(figsize=(12, 8))

        for i in range(self.n_classes):
            plt.plot(
                fpr[i], tpr[i],
                label=f'Class {self.classes[i]} (AUC = {roc_auc[i]:.2f})',
                linewidth=2
            )

        plt.plot(
            fpr['micro'], tpr['micro'],
            label=f'Micro-average (AUC = {roc_auc["micro"]:.2f})',
            linestyle='--', linewidth=2.5, color='black'
        )

        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.50)')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) - Multiclass')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)

        if self.save_plots and filename:
            plt.savefig(f"{self.plot_dir}/{filename}", dpi=300, bbox_inches='tight')
            logger.info(f"ROC curves saved to {self.plot_dir}/{filename}")

        plt.close()

    def plot_metrics_summary(
        self,
        metrics_per_class: Dict[str, Dict[str, float]],
        filename: Optional[str] = None
    ) -> None:
        """
        Plot summary of per-class metrics.

        Args:
            metrics_per_class: Dictionary of per-class metrics
            filename: Optional filename to save plot
        """
        metric_names = ['precision', 'recall', 'f1']
        x_pos = np.arange(len(self.classes))
        width = 0.25

        fig, ax = plt.subplots(figsize=(12, 6))

        for idx, metric in enumerate(metric_names):
            values = [metrics_per_class[c].get(metric, 0) for c in self.classes]
            ax.bar(x_pos + idx * width, values, width, label=metric.capitalize())

        ax.set_ylabel('Score')
        ax.set_title('Per-Class Metrics Summary')
        ax.set_xticks(x_pos + width)
        ax.set_xticklabels(self.classes)
        ax.legend()
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')

        if self.save_plots and filename:
            plt.savefig(f"{self.plot_dir}/{filename}", dpi=300, bbox_inches='tight')
            logger.info(f"Metrics summary saved to {self.plot_dir}/{filename}")

        plt.close()
