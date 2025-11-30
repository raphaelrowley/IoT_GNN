def get_accuracy(tn, tp, fn, fp):
    return float((tp + tn)) / (tp + tn + fp + fn)

def get_balanced_accuracy(tn, tp, fn, fp):
    # Treat both classes equally.
    tpr = float(tp) / (tp + fn + 1e-12)
    tnr = float(tn) / (tn + fp + 1e-12)
    return 0.5 * (tpr + tnr)

def get_far(tn, tp, fn, fp):
    return float(fp) / (fp + tn)

def get_recall(tn, tp, fn, fp):
    return float(tp) / (tp + fn)

def get_precision(tn, tp, fn, fp):
    return float(tp) / (tp + fp)

def get_f1_score(tn, tp, fn, fp):
    recall = get_recall(tn, tp, fn, fp)
    precision = get_precision(tn, tp, fn, fp)

    return 2 * ((recall*precision) / (recall+precision))

