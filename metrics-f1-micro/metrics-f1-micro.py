def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    tp, i = 0, 0
    while i < len(y_true):
        if y_pred[i] == y_true[i]:
            tp += 1

        i += 1

    return float(tp/len(y_true))