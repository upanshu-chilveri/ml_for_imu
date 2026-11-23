"""
RoNIN (Robust Neural Inertial Navigation) / TCN Architecture for Pure IMU Odometry.

Implements:
  1. Residual 1D Temporal Convolutional Network (ResNet-1D / TCN)
  2. Chronological Sliding Window Dataset Generator (70% Train, 15% Val, 15% Test)
  3. Feature Normalizer and Coordinate Transformation Engine
  4. Body-Frame Velocity Prediction to Bypass Double-Integration Drift
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter1d


def skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


class ResidualBlock1D(nn.Module):
    """
    1D Residual Convolutional Block with Batch Normalization and ReLU.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += res
        return self.relu(out)


class RoNIN_TCN(nn.Module):
    """
    Deep 1D Temporal Convolutional Network (TCN) for Learned Inertial Odometry.
    Maps temporal windows of IMU measurements directly to body-frame 3D velocity vectors.
    """
    def __init__(self, in_channels: int = 8, out_dim: int = 3):
        super().__init__()
        self.prep = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        self.layer1 = ResidualBlock1D(64, 64, stride=1)
        self.layer2 = ResidualBlock1D(64, 128, stride=2)
        self.layer3 = ResidualBlock1D(128, 128, stride=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.prep(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.pool(out).squeeze(-1)
        return self.fc(out)


class RoNINDatasetManager:
    """
    Manages IMU feature extraction, coordinate transformation, windowing, and chronological splitting.
    """
    def __init__(self, window_size: int = 40, step: int = 2):
        self.window_size = window_size
        self.step = step
        self.feat_mean = None
        self.feat_std = None

    def prepare_data(self, imu_df: pd.DataFrame, gt_df: pd.DataFrame):
        t_start = max(imu_df['timestamp_ns'].min(), gt_df['timestamp_ns'].min())
        t_end = min(imu_df['timestamp_ns'].max(), gt_df['timestamp_ns'].max())

        imu = imu_df[(imu_df['timestamp_ns'] >= t_start) & (imu_df['timestamp_ns'] <= t_end)].reset_index(drop=True)
        gt = gt_df[(gt_df['timestamp_ns'] >= t_start) & (gt_df['timestamp_ns'] <= t_end)].reset_index(drop=True)

        imu_ts = imu['timestamp_ns'].values
        gt_ts = gt['timestamp_ns'].values
        gt_pos = gt[['p_x', 'p_y', 'p_z']].values

        # Interpolate ground truth positions to IMU sample times
        interp_px = np.interp(imu_ts, gt_ts, gt_pos[:, 0])
        interp_py = np.interp(imu_ts, gt_ts, gt_pos[:, 1])
        interp_pz = np.interp(imu_ts, gt_ts, gt_pos[:, 2])
        pos_gt_interp = np.column_stack([interp_px, interp_py, interp_pz])

        # Compute Ground Truth velocity in World/Navigation Frame
        dt = np.diff(imu_ts, prepend=imu_ts[0] - 5000000) / 1e9
        dt = np.where(dt <= 0, 0.005, dt)
        vel_gt_world = np.gradient(pos_gt_interp, axis=0) / dt[:, None]
        vel_gt_world = gaussian_filter1d(vel_gt_world, sigma=1.5, axis=0)

        imu_raw = imu[['w_x', 'w_y', 'w_z', 'a_x', 'a_y', 'a_z']].values

        # Static gravity leveling for initial attitude
        a_mean = imu_raw[:100, 3:6].mean(axis=0)
        u_acc = a_mean / np.linalg.norm(a_mean)
        u_z = np.array([0, 0, 1.0])
        v_rot = np.cross(u_acc, u_z)
        s_rot = np.linalg.norm(v_rot)
        c_rot = np.dot(u_acc, u_z)
        vx = skew(v_rot)
        R_init = np.eye(3) + vx + (vx @ vx) * ((1 - c_rot) / (s_rot**2))

        # Integrate gyroscope orientation
        N = len(imu_raw)
        rot_matrices = np.zeros((N, 3, 3))
        curr_rot = R.from_matrix(R_init)

        for i in range(N):
            rot_matrices[i] = curr_rot.as_matrix()
            if i < N - 1:
                w = imu_raw[i, 0:3]
                d_theta = w * dt[i]
                curr_rot = curr_rot * R.from_rotvec(d_theta)

        # Body-frame ground truth velocity: v_body = R^T * v_world
        vel_gt_body = np.zeros((N, 3))
        for i in range(N):
            vel_gt_body[i] = rot_matrices[i].T @ vel_gt_world[i]

        # Multi-channel feature tensor: [w_x, w_y, w_z, a_x, a_y, a_z, norm(w), norm(a)] -> 8 channels
        norm_w = np.linalg.norm(imu_raw[:, 0:3], axis=1, keepdims=True)
        norm_a = np.linalg.norm(imu_raw[:, 3:6], axis=1, keepdims=True)
        features = np.column_stack([imu_raw, norm_w, norm_a])

        # Chronological normalization based on 70% training slice
        n_train_idx = int(N * 0.70)
        self.feat_mean = features[:n_train_idx].mean(axis=0)
        self.feat_std = features[:n_train_idx].std(axis=0) + 1e-6
        features_norm = (features - self.feat_mean) / self.feat_std

        # Construct sliding temporal windows
        windows = []
        targets_body = []
        indices = []

        for i in range(self.window_size, N, self.step):
            windows.append(features_norm[i-self.window_size:i].T)
            targets_body.append(vel_gt_body[i])
            indices.append(i)

        windows = np.array(windows, dtype=np.float32)
        targets_body = np.array(targets_body, dtype=np.float32)
        indices = np.array(indices)

        return {
            'windows': windows,
            'targets_body': targets_body,
            'indices': indices,
            'imu_ts': imu_ts,
            'pos_gt': pos_gt_interp,
            'vel_gt_world': vel_gt_world,
            'vel_gt_body': vel_gt_body,
            'rot_matrices': rot_matrices,
            'dt': dt
        }

