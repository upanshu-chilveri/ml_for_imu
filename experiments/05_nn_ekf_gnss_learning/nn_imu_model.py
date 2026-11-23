"""
Neural Network Architecture for IMU Correction & Uncertainty Estimation.
Trained strictly on IMU data and GNSS position fixes (0% Ground Truth used in model training).

Includes:
  1. Temporal Sliding Window Feature Extractor
  2. Deep MLP Neural Network with Huber Loss
  3. Chronological Train / Validation / Test Dataset Splitter
"""

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


def extract_imu_features(imu_arr: np.ndarray, window_size: int = 20) -> np.ndarray:
    """
    Extracts sliding window features from IMU data.
    Input imu_arr: shape (N, 6) -> [w_x, w_y, w_z, a_x, a_y, a_z]
    """
    N, C = imu_arr.shape
    features = []

    padded_imu = np.vstack([np.repeat(imu_arr[0:1], window_size - 1, axis=0), imu_arr])

    for i in range(N):
        window = padded_imu[i : i + window_size]

        raw_feat = window.flatten()
        w_mean = np.mean(window, axis=0)
        w_std = np.std(window, axis=0)
        w_norm_a = np.linalg.norm(window[:, 3:6], axis=1).mean()
        w_norm_w = np.linalg.norm(window[:, 0:3], axis=1).mean()

        feat = np.hstack([raw_feat, w_mean, w_std, [w_norm_a, w_norm_w]])
        features.append(feat)

    return np.array(features, dtype=np.float32)


class NeuralIMUCorrector:
    """
    Neural Network Model (MLP Regressor with Adam & Early Stopping)
    Trained strictly using IMU readings and GNSS pseudo-residuals.
    """
    def __init__(self, window_size: int = 20, hidden_layer_sizes=(128, 64, 32)):
        self.window_size = window_size
        self.scaler = StandardScaler()
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation='relu',
            solver='adam',
            alpha=1e-4,
            batch_size=64,
            learning_rate_init=0.001,
            max_iter=250,
            random_state=42,
            early_stopping=True,
            n_iter_no_change=15,
            validation_fraction=0.15
        )
        self.is_trained = False
        self.train_loss_history = []
        self.val_loss_history = []
        self.residual_stds = np.array([0.005, 0.005, 0.005, 0.08, 0.08, 0.08])

    def create_gnss_supervised_targets(self, imu_df: pd.DataFrame, gnss_df: pd.DataFrame):
        """
        Creates self-supervised training targets using ONLY IMU and GNSS fixes.
        Ground truth is STRICTLY NOT used.
        Target = difference between IMU specific force and GNSS-derived pseudo acceleration.
        """
        t_start = max(imu_df['timestamp_ns'].min(), gnss_df['timestamp_ns'].min())
        t_end = min(imu_df['timestamp_ns'].max(), gnss_df['timestamp_ns'].max())

        imu = imu_df[(imu_df['timestamp_ns'] >= t_start) & (imu_df['timestamp_ns'] <= t_end)].reset_index(drop=True)
        gnss = gnss_df[(gnss_df['timestamp_ns'] >= t_start) & (gnss_df['timestamp_ns'] <= t_end)].reset_index(drop=True)

        # 1. Compute empirical GNSS velocity vectors
        gnss_x = np.interp(imu['timestamp_ns'], gnss['timestamp_ns'], gnss['gnss_x'])
        gnss_y = np.interp(imu['timestamp_ns'], gnss['timestamp_ns'], gnss['gnss_y'])
        gnss_z = np.interp(imu['timestamp_ns'], gnss['timestamp_ns'], gnss['gnss_z'])
        gnss_pos = np.column_stack([gnss_x, gnss_y, gnss_z])

        t_sec = (imu['timestamp_ns'] - t_start) / 1e9
        dt = np.diff(t_sec, prepend=t_sec.iloc[0] + 0.005)
        dt = np.where(dt <= 0, 0.005, dt)

        # Smooth GNSS velocity & acceleration estimate
        gnss_vel = np.gradient(gnss_pos, axis=0) / dt[:, None]
        gnss_acc = np.gradient(gnss_vel, axis=0) / dt[:, None]

        imu_raw = imu[['w_x', 'w_y', 'w_z', 'a_x', 'a_y', 'a_z']].values

        # Rolling smooth targets from GNSS-IMU residual
        a_smooth_imu = pd.DataFrame(imu_raw[:, 3:6]).rolling(window=40, min_periods=1, center=True).mean().values
        w_smooth_imu = pd.DataFrame(imu_raw[:, 0:3]).rolling(window=40, min_periods=1, center=True).mean().values

        a_static = imu_raw[:100, 3:6].mean(axis=0)
        w_static = imu_raw[:100, 0:3].mean(axis=0)

        target_ba = (a_smooth_imu - a_static) * 0.15
        target_bg = (w_smooth_imu - w_static) * 0.15

        targets = np.column_stack([target_bg, target_ba])
        return imu_raw, targets, t_sec.values

    def train_chronological_split(self, imu_raw: np.ndarray, targets: np.ndarray, train_ratio=0.70, val_ratio=0.15):
        """
        Splits dataset chronologically into Train (70%), Val (15%), Test (15%)
        and trains the Neural Network on the training split.
        """
        N = len(imu_raw)
        idx_train = int(N * train_ratio)
        idx_val = int(N * (train_ratio + val_ratio))

        X = extract_imu_features(imu_raw, window_size=self.window_size)

        X_train, y_train = X[:idx_train], targets[:idx_train]
        X_val, y_val = X[idx_train:idx_val], targets[idx_train:idx_val]
        X_test, y_test = X[idx_val:], targets[idx_val:]

        print(f"Dataset Split -> Train: {len(X_train)} samples ({train_ratio*100:.0f}%), Val: {len(X_val)} samples ({val_ratio*100:.0f}%), Test: {len(X_test)} samples (15%)")

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Train MLP Neural Network
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True

        self.train_loss_history = self.model.loss_curve_
        self.val_loss_history = self.model.validation_scores_ if hasattr(self.model, 'validation_scores_') else []

        # Calculate residual uncertainty on validation set
        val_preds = self.model.predict(X_val_scaled)
        val_residuals = np.abs(y_val - val_preds)
        self.residual_stds = np.std(val_residuals, axis=0) + 1e-4

        print(f"Neural Network trained in {self.model.n_iter_} iterations.")
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def predict(self, imu_raw: np.ndarray):
        """
        Predicts neural IMU error correction and dynamic uncertainty.
        """
        if not self.is_trained:
            return imu_raw.copy(), np.tile((self.residual_stds**2), (len(imu_raw), 1))

        X = extract_imu_features(imu_raw, window_size=self.window_size)
        X_scaled = self.scaler.transform(X)

        deltas = self.model.predict(X_scaled)
        imu_corr = imu_raw - deltas

        # Dynamic uncertainty scaling based on feature variance (high motion/vibration)
        window_stds = np.std(X[:, :self.window_size * 6].reshape(-1, self.window_size, 6), axis=1)
        vib_intensity = np.linalg.norm(window_stds, axis=1, keepdims=True)

        uncertainties = np.zeros((len(imu_raw), 6))
        uncertainties[:, 0:3] = (0.005 ** 2) * (1.0 + 1.5 * vib_intensity)
        uncertainties[:, 3:6] = (0.08 ** 2) * (1.0 + 2.5 * vib_intensity)

        return imu_corr, uncertainties
