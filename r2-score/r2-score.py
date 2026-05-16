import numpy as np

def r2_score(y_true, y_pred) -> float:
    """
    Compute R² (coefficient of determination) for 1D regression.
    Handle the constant-target edge case:
      - return 1.0 if predictions match exactly,
      - else 0.0.
    """
    
    y_true_np = np.array(y_true, dtype=float)
    y_pred_np = np.array(y_pred, dtype=float)

    # Sum of squared residuals
    ss_res = np.sum((y_true_np - y_pred_np) ** 2)

    # Sum of squares total
    mean_true = np.mean(y_true_np)
    ss_tot = np.sum((y_true_np - mean_true) ** 2)

    # Handle constant target case
    if ss_tot == 0:
        if np.allclose(y_true_np, y_pred_np):
            return 1.0
        return 0.0

    return 1 - (ss_res / ss_tot)