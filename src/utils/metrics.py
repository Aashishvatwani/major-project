"""
Aerospace Anomaly Detection Performance Metrics
Computes ROC-AUC, PR-AUC, Detection Latency, False Alarm Rate, and Safety Override Statistics.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)


def calculate_telemetry_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    inference_latencies_ms: Optional[List[float]] = None
) -> Dict[str, Any]:
    """Calculates comprehensive benchmark metrics"""
    y_true_clean = np.array(y_true, dtype=int)
    y_pred_clean = np.array(y_pred, dtype=int)
    y_probs_clean = np.array(y_probs, dtype=float)

    has_both_classes = len(np.unique(y_true_clean)) > 1

    roc = float(roc_auc_score(y_true_clean, y_probs_clean)) if has_both_classes else 0.5
    pr = float(average_precision_score(y_true_clean, y_probs_clean)) if has_both_classes else 0.0
    f1 = float(f1_score(y_true_clean, y_pred_clean, zero_division=0))
    prec = float(precision_score(y_true_clean, y_pred_clean, zero_division=0))
    rec = float(recall_score(y_true_clean, y_pred_clean, zero_division=0))

    cm = confusion_matrix(y_true_clean, y_pred_clean)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        far = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        mdr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    else:
        far, mdr, tn, fp, fn, tp = 0.0, 0.0, int(cm[0, 0]), 0, 0, 0

    avg_latency = float(np.mean(inference_latencies_ms)) if inference_latencies_ms else 0.0
    p99_latency = float(np.percentile(inference_latencies_ms, 99)) if inference_latencies_ms else 0.0

    return {
        "roc_auc": roc,
        "pr_auc": pr,
        "f1_score": f1,
        "precision": prec,
        "recall": rec,
        "false_alarm_rate": far,
        "missed_detection_rate": mdr,
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        },
        "latency_ms": {
            "mean": avg_latency,
            "p99": p99_latency
        }
    }
