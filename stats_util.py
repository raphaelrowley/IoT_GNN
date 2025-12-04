# stats_util.py
# Author: Raphael Rowley, November 2025
#
# Description:
# Set of helper functions to compute binary classification performance metrics.
# They assume that TN, TP, FN, FP have all been previously computed.
# TN: True negative.
# TP: True positive.
# FN: False negative.
# FP: False positive.

def get_accuracy(tn, tp, fn, fp):
    return float((tp + tn)) / (tp + tn + fp + fn)

def get_balanced_accuracy(tn, tp, fn, fp):
    # Treat both classes equally.
    tpr = float(tp) / (tp + fn + 1e-12)
    tnr = float(tn) / (tn + fp + 1e-12)
    return 0.5 * (tpr + tnr)

# Computes the FAR (False Alarm Rate).
def get_far(tn, tp, fn, fp):
    return float(fp) / (fp + tn)

def get_recall(tn, tp, fn, fp):
    return float(tp) / (tp + fn)

def get_precision(tn, tp, fn, fp):
    return float(tp) / (tp + fp)

# Computes the F1-score (harmonic mean of recall and precision).
def get_f1_score(tn, tp, fn, fp):
    recall = get_recall(tn, tp, fn, fp)
    precision = get_precision(tn, tp, fn, fp)

    return 2 * ((recall*precision) / (recall+precision))

