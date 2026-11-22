import numpy as np
from scipy.linalg import fractional_matrix_power

def euclidean_alignment(X):
    """
    Fixes domain shift between healthy and stroke EEG datasets.
    Input X shape: (n_trials, n_channels, n_timepoints)
    """
    n_trials, n_channels, n_times = X.shape
    covs = np.zeros((n_trials, n_channels, n_channels))

    for i in range(n_trials):
        covs[i] = np.dot(X[i], X[i].T) / n_times

    mean_cov = np.mean(covs, axis=0) + 1e-6 * np.eye(n_channels)
    r_inv_sqrt = np.real(fractional_matrix_power(mean_cov, -0.5))

    return np.array([np.dot(r_inv_sqrt, X[i]) for i in range(n_trials)])

def adaptive_fatigue_threshold(theta_power, alpha_power, base_threshold=0.75):
    """
    Monitors frontal lobe fatigue (Theta/Alpha ratio).
    Scales AI threshold down when patient is exhausted.
    """
    ratio = theta_power / max(alpha_power, 1e-5)
    
    if ratio < 1.0:
        return base_threshold, ratio, "Focused (Normal Threshold)"
    elif ratio < 1.8:
        return base_threshold - 0.12, ratio, "Moderate Fatigue (Auto-Scaled -12%)"
    else:
        return max(0.50, base_threshold - 0.22), ratio, "High Burnout (Auto-Scaled -22%)"
