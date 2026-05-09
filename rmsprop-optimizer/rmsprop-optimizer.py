import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    w_np = np.array(w, dtype=np.float64)
    g_np = np.array(g, dtype=np.float64)
    s_np = np.array(s, dtype=np.float64)

    # Update running average of squared gradients
    new_s = beta * s_np + (1 - beta) * (g_np ** 2)

    # Update weights
    new_w = w_np - lr * g_np / (np.sqrt(new_s) + eps)

    return new_w, new_s