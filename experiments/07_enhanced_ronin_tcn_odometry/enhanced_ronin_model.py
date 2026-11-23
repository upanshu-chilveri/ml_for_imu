"""
Experiment 7: Multi-Scale Dilated Temporal Convolutional Network (Dilated TCN)
with End-to-End Window Integration & Displacement Loss.

Implements:
  1. Dilated Residual 1D Convolutions with exponentially growing receptive fields (d=1, 2, 4)
  2. End-to-End Window Integration & Relative Displacement Loss
  3. Systematic Velocity Bias Regularizer to eliminate unidirectional drift
  4. Expanded temporal context window (W=80, 0.40s)
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


class DilatedResidualBlock1D(nn.Module):
    """
    Dilated 1D Residual Convolutional Block.
    Captures multi-scale temporal dependencies without downsampling spatial resolution.
    """
    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1, stride: int = 1):
        super().__init__()
        padding = dilation
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, stride=stride,
            padding=padding, dilation=dilation, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, stride=1,
            padding=padding, dilation=dilation, bias=False
        )
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


class MultiScaleDilatedTCN(nn.Module):
    """
    Multi-Scale Dilated Temporal Convolutional Network (TCN).
    Receptive field covers:
      - d=1: High-frequency MEMS vibrations & angular accelerations
      - d=2: Short flight maneuvers & velocity changes (~0.20s)
      - d=4: Macro flight transitions & curved trajectory dynamics (~0.40s)
    """
    def __init__(self, in_channels: int = 8, out_dim: int = 3):
        super().__init__()
        self.prep = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        self.b1 = DilatedResidualBlock1D(64, 64, dilation=1)
        self.b2 = DilatedResidualBlock1D(64, 128, dilation=2, stride=2)
        self.b3 = DilatedResidualBlock1D(128, 128, dilation=4)
        self.b4 = DilatedResidualBlock1D(128, 128, dilation=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.prep(x)
        out = self.b1(out)
        out = self.b2(out)
        out = self.b3(out)
        out = self.b4(out)
        out = self.pool(out).squeeze(-1)
        return self.fc(out)


class IntegrationLoss(nn.Module):
    """
    Compound Loss Function:
      1. Step-Wise Velocity Huber Loss (Smooth L1)
      2. Multi-Step Relative Displacement Loss: || sum(v * dt) - Delta p ||
      3. Zero-Mean Velocity Error Regularizer: || mean(v - v*) ||^2 (squashes systematic drift)
    """
    def __init__(self, lambda_disp: float = 2.0, lambda_bias: float = 1.0):
        super().__init__()
        self.lambda_disp = lambda_disp
        self.lambda_bias = lambda_bias
        self.vel_loss = nn.SmoothL1Loss()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor, dts: torch.Tensor):
        # preds: (B, K, 3), targets: (B, K, 3), dts: (B, K)
        # 1. Step-wise velocity loss
        l_vel = self.vel_loss(preds, targets)

        # 2. Window integration displacement loss
        pred_disp = torch.sum(preds * dts.unsqueeze(-1), dim=1) # (B, 3)
        true_disp = torch.sum(targets * dts.unsqueeze(-1), dim=1) # (B, 3)
        l_disp = nn.functional.smooth_l1_loss(pred_disp, true_disp)

        # 3. Systematic velocity bias regularizer
        mean_bias = torch.mean(preds - targets, dim=1) # (B, 3)
        l_bias = torch.mean(mean_bias ** 2)

        total_loss = l_vel + self.lambda_disp * l_disp + self.lambda_bias * l_bias
        return total_loss, l_vel, l_disp, l_bias


class ChunkDataset(Dataset):
    """
    Sequence chunk dataset providing contiguous K-step temporal segments for integration loss.
    """
    def __init__(self, X: torch.Tensor, y: torch.Tensor, dts: torch.Tensor):
        self.X = X
        self.y = y
        self.dts = dts

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.dts[idx]


class EnhancedRoNINDatasetManager:
    """
    Prepares multi-channel IMU sliding windows (W=80, 0.40s context) and sequence chunks.
    """
    def __init__(self, window_size: int = 80, step: int = 2, chunk_len: int = 16):
        self.window_size = window_size
        self.step = step
        self.chunk_len = chunk_len
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

        # Static gravity alignment
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
                w = imu_raw[i, 0:3]
                d_theta = w * dt[i]
                curr_rot = curr_rot * R.from_rotvec(d_theta)

        # Ground truth body velocity
        vel_gt_body = np.zeros((N, 3))
        for i in range(N):
            vel_gt_body[i] = rot_matrices[i].T @ vel_gt_world[i]

        norm_w = np.linalg.norm(imu_raw[:, 0:3], axis=1, keepdims=True)
        norm_a = np.linalg.norm(imu_raw[:, 3:6], axis=1, keepdims=True)
        features = np.column_stack([imu_raw, norm_w, norm_a])

        n_train_idx = int(N * 0.70)
        self.feat_mean = features[:n_train_idx].mean(axis=0)
        self.feat_std = features[:n_train_idx].std(axis=0) + 1e-6
        features_norm = (features - self.feat_mean) / self.feat_std

        # Build sliding windows
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

        # Build sequence chunks of length K for integration loss
        K = self.chunk_len
        chunk_windows = []
        chunk_targets = []
        chunk_dts = []

        for i in range(0, len(windows) - K, K // 2):
            chunk_windows.append(windows[i:i+K])
            chunk_targets.append(targets_body[i:i+K])
            idx_chunk = indices[i:i+K]
            dt_chunk = np.diff(imu_ts[idx_chunk], prepend=imu_ts[idx_chunk[0]] - int(0.01*1e9)) / 1e9
            chunk_dts.append(dt_chunk.astype(np.float32))

        chunk_windows = np.array(chunk_windows, dtype=np.float32)
        chunk_targets = np.array(chunk_targets, dtype=np.float32)
        chunk_dts = np.array(chunk_dts, dtype=np.float32)

        return {
            'windows': windows,
            'targets_body': targets_body,
            'indices': indices,
            'chunk_windows': chunk_windows,
            'chunk_targets': chunk_targets,
            'chunk_dts': chunk_dts,
            'imu_ts': imu_ts,
            'pos_gt': pos_gt_interp,
            'vel_gt_world': vel_gt_world,
            'vel_gt_body': vel_gt_body,
            'rot_matrices': rot_matrices,
            'dt': dt
        }

