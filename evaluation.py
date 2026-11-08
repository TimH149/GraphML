"""Multi-label evaluation: score-based metrics plus F1 under several decision rules.

Score-based metrics (AP, AUROC) use the predicted probabilities directly, so they do not
depend on the decision rule. F1 needs binary predictions, which come from one of:

  threshold_0.5        predict every label with p >= 0.5
  per_label_threshold  one threshold per label, tuned on the validation nodes for per-label F1
  topk_pred            the top-k_i labels per node, with k_i from a cardinality head
  topk_oracle          the top-k_i labels per node, with k_i the true number of labels
                       (an upper bound for analysis only; this is what metrics.f1_loss computes)

All functions take numpy arrays of shape (num_nodes, num_labels).
"""
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

DECISION_RULES = ["threshold_0.5", "per_label_threshold", "topk_pred", "topk_oracle"]
THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.05), 2)


def _labels_with_both_classes(y_true):
    num_pos = y_true.sum(axis=0)
    return np.where((num_pos > 0) & (num_pos < y_true.shape[0]))[0]


def macro_ap(y_true, y_score):
    """Macro average precision over the labels that have at least one positive and one negative node."""
    labels = _labels_with_both_classes(y_true)
    return float(np.mean([average_precision_score(y_true[:, c], y_score[:, c]) for c in labels]))


def macro_auroc(y_true, y_score):
    """Macro AUROC over the labels that have at least one positive and one negative node (as metrics.rocauc_)."""
    labels = _labels_with_both_classes(y_true)
    return float(np.mean([roc_auc_score(y_true[:, c], y_score[:, c]) for c in labels]))


def predict_threshold(y_score, thresholds=0.5):
    """Binary predictions from a global threshold or one threshold per label."""
    return (y_score >= np.asarray(thresholds)).astype(int)


def tune_per_label_thresholds(y_true, y_score, grid=THRESHOLD_GRID, default=0.5):
    """For every label, the threshold from the grid with the highest F1 on (y_true, y_score).

    Labels without a positive node in y_true keep the default threshold.
    """
    pred = y_score[None, :, :] >= grid[:, None, None]   # (num_thresholds, num_nodes, num_labels)
    tp = (pred & (y_true[None] == 1)).sum(axis=1)
    fp = (pred & (y_true[None] == 0)).sum(axis=1)
    fn = ((~pred) & (y_true[None] == 1)).sum(axis=1)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    thresholds = grid[f1.argmax(axis=0)]
    thresholds[y_true.sum(axis=0) == 0] = default
    return thresholds


def predict_topk(y_score, k):
    """Predict the k_i highest-scoring labels of every node, with k_i clipped to [1, num_labels]."""
    k = np.clip(np.asarray(k).astype(int), 1, y_score.shape[1])
    ranks = np.argsort(np.argsort(-y_score, axis=1), axis=1)  # 0 = highest score
    return (ranks < k[:, None]).astype(int)


def f1_pair(y_true, y_pred):
    return (float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
            float(f1_score(y_true, y_pred, average="macro", zero_division=0)))


def evaluate(y_true, y_score, thresholds=None, k_pred=None):
    """All metrics for one set of nodes.

    thresholds: per-label thresholds tuned on validation (enables per_label_threshold).
    k_pred:     predicted number of labels per node (enables topk_pred).
    Returns a flat dict such as {"ap_macro": ..., "micro_f1@threshold_0.5": ...}.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)
    results = {"ap_macro": macro_ap(y_true, y_score), "auroc_macro": macro_auroc(y_true, y_score)}

    predictions = {"threshold_0.5": predict_threshold(y_score, 0.5),
                   "topk_oracle": predict_topk(y_score, y_true.sum(axis=1))}
    if thresholds is not None:
        predictions["per_label_threshold"] = predict_threshold(y_score, thresholds)
    if k_pred is not None:
        predictions["topk_pred"] = predict_topk(y_score, k_pred)

    for rule in DECISION_RULES:
        if rule in predictions:
            results[f"micro_f1@{rule}"], results[f"macro_f1@{rule}"] = f1_pair(y_true, predictions[rule])
        else:
            results[f"micro_f1@{rule}"] = results[f"macro_f1@{rule}"] = float("nan")
    return results


SELECTABLE_RULES = ["threshold_0.5", "per_label_threshold", "topk_pred"]


def crossfit_val_f1(y_val, p_val, k_pred_val=None, seed=0):
    """Validation micro/macro-F1 of each real decision rule, for choosing the rule on validation.

    The per-label thresholds are tuned on validation, so scoring them on the same nodes is optimistic. Here
    they are cross-fitted instead: the validation nodes are split in two halves, thresholds tuned on one
    half predict the other, and both halves are scored together. The other rules need no tuning.
    Returns {rule: (micro_f1, macro_f1)}; top-k predicted only if k_pred_val is given.
    """
    y_val = np.asarray(y_val).astype(int)
    halves = np.array_split(np.random.default_rng(seed).permutation(len(y_val)), 2)
    cross_fitted = np.zeros_like(y_val)
    for tune, score in [(0, 1), (1, 0)]:
        thresholds = tune_per_label_thresholds(y_val[halves[tune]], p_val[halves[tune]])
        cross_fitted[halves[score]] = predict_threshold(p_val[halves[score]], thresholds)
    predictions = {"threshold_0.5": predict_threshold(p_val, 0.5), "per_label_threshold": cross_fitted}
    if k_pred_val is not None:
        predictions["topk_pred"] = predict_topk(p_val, k_pred_val)
    return {rule: f1_pair(y_val, pred) for rule, pred in predictions.items()}
