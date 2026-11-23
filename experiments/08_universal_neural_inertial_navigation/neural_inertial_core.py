"""
Experiment 8: Universal Neural-Inertial Navigation System (NINS).
Core PyTorch Model: 12-Channel Torque-Augmented ResNet-1D + Bi-GRU with Uncertainty.

Features:
  1. 12-Channel Rotational Torque Dynamics:
     - Angular velocities: [w_x, w_y, w_z]
     - Accelerations: [a_x, a_y, a_z]
     - Angular accelerations: [dw_x/dt, dw_y/dt, dw_z/dt]
     - Gyroscopic coupling torques: [tau_x, tau_y, tau_z]
  2. ResNet-1D feature encoder for multi-frequency vibration isolation
  3. Bi-directional GRU for continuous drone kinetic momentum tracking
  4. Dual Heads: Body-frame velocity vector (v^b) + Log-variance uncertainty (s)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter1d


def skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


class ResBlock1D(nn.Module):
    """
    1D Residual Block with Batch Normalization and ReLU.
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


class UniversalNeuralInertialCore(nn.Module):
    """
    Hybrid ResNet-1D + Bi-GRU Neural Core for Drone Inertial Dynamics.
    """
    def __init__(self, in_channels: int = 12):
        super().__init__()
        self.prep = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        self.res1 = ResBlock1D(64, 64, stride=1)
        self.res2 = ResBlock1D(64, 128, stride=2)

        # Bi-directional GRU to maintain flight kinetic momentum
        self.gru = nn.GRU(128, 64, batch_first=True, bidirectional=True)

        # Dual Heads
        self.fc_vel = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, 3)
        )
        self.fc_var = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3)
        )

    def forward(self, x: torch.Tensor):
        # x: (B, 12, W)
        out = self.res2(self.res1(self.prep(x))) # (B, 128, W//2)
        out = out.permute(0, 2, 1) # (B, W//2, 128)
        gru_out, _ = self.gru(out)
        feat = gru_out[:, -1, :] # (B, 128)

        v_body = self.fc_vel(feat)
        log_var = torch.clamp(self.fc_var(feat), min=-6.0, max=2.0)
        return v_body, log_var


class NINSDatasetManager:
    """
    Constructs 12-channel torque features, body-frame ground truth velocities,
    and sliding window datasets.
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

        interp_px = np.interp(imu_ts, gt_ts, gt_pos[:, 0])
        interp_py = np.interp(imu_ts, gt_ts, gt_pos[:, 1])
        interp_pz = np.interp(imu_ts, gt_ts, gt_pos[:, 2])
        pos_gt_interp = np.column_stack([interp_px, interp_py, interp_pz])

        dt = np.diff(imu_ts, prepend=imu_ts[0] - 5000000) / 1e9
        dt = np.where(dt <= 0, 0.005, dt)
        vel_gt_world = np.gradient(pos_gt_interp, axis=0) / dt[:, None]
        vel_gt_world = gaussian_filter1d(vel_gt_world, sigma=1.5, axis=0)

        imu_raw = imu[['w_x', 'w_y', 'w_z', 'a_x', 'a_y', 'a_z']].values

        # 12D Torque-Augmented Feature Extraction
        w = imu_raw[:, 0:3]
        a = imu_raw[:, 3:6]
        dw_dt = gaussian_filter1d(np.gradient(w, axis=0) / dt[:, None], sigma=1.0, axis=0)
        tau_gyro = np.column_stack([
            w[:, 1] * w[:, 2] * 0.5,
            w[:, 2] * w[:, 0] * -0.5,
            w[:, 0] * w[:, 1] * 0.2
        ])
        features_12d = np.column_stack([w, a, dw_dt, tau_gyro])

        # Initial attitude via gravity leveling
        a_mean = imu_raw[:100, 3:6].mean(axis=0)
        u_acc = a_mean / np.linalg.norm(a_mean)
        u_z = np.array([0, 0, 1.0])
        v_rot = np.cross(u_acc, u_z)
        s_rot = np.linalg.norm(v_rot)
        c_rot = np.dot(u_acc, u_z)
        vx = skew(v_rot)
        R_init = np.eye(3) + vx + (vx @ vx) * ((1 - c_rot) / (s_rot**2))

        N = len(imu_raw)
        rot_matrices = np.zeros((N, 3, 3))
        curr_rot = R.from_matrix(R_init)

        for i in range(N):
            rot_matrices[i] = curr_rot.as_matrix()
            if i < N - 1:
                curr_rot = curr_rot * R.from_rotvec(w[i] * dt[i])

        # Body-frame ground truth velocities: v_body = R^T * v_world
        vel_gt_body = np.zeros((N, 3))
        for i in range(N):
            vel_gt_body[i] = rot_matrices[i].T @ vel_gt_world[i]

        # Chronological normalization
        n_train_idx = int(N * 0.70)
        self.feat_mean = features_12d[:n_train_idx].mean(axis=0)
        self.feat_std = features_12d[:n_train_idx].std(axis=0) + 1e-6
        features_norm = (features_12d - self.feat_mean) / self.feat_std

        # Construct sliding windows
        windows = []
        targets = []
        indices = []

        for i in range(self.window_size, N, self.step):
            windows.append(features_norm[i-self.window_size:i].T)
            targets.append(vel_gt_body[i])
            indices.append(i)

        windows = np.array(windows, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)
        indices = np.array(indices)

        return {
            'windows': windows,
            'targets_body': targets,
            'indices': indices,
            'imu_ts': imu_ts,
            'pos_gt': pos_gt_interp,
            'vel_gt_world': vel_gt_world,
            'vel_gt_body': vel_gt_body,
            'rot_matrices': rot_matrices,
            'dt': dt,
            'features_12d': features_12d,
            'R_init': R_init,
            'imu_raw': imu_raw
        }
