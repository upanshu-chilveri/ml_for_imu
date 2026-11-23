"""
EuRoC MAV Multi-Sequence Data Loader for Experiment 09.

Loads raw IMU + ground truth CSVs from any EuRoC ASL-format sequence,
synchronises them by timestamp, computes 12-channel torque features,
and builds sliding-window datasets with optional rotation augmentation.

EuRoC IMU CSV columns:
  #timestamp [ns], w_RS_S_x, w_RS_S_y, w_RS_S_z, a_RS_S_x, a_RS_S_y, a_RS_S_z

EuRoC GT CSV columns:
  #timestamp, p_RS_R_x, p_RS_R_y, p_RS_R_z, q_RS_w, q_RS_x, q_RS_y, q_RS_z,
  v_RS_R_x, v_RS_R_y, v_RS_R_z, b_w_RS_S_x..., b_a_RS_S_x...
"""

import os
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


# ── Cross-sequence dataset split ──────────────────────────────────────────────
SEQUENCE_SPLIT = {
    "train": [
        "MH_01_easy", "MH_02_easy", "MH_03_medium",
        "V1_01_easy", "V1_02_medium",
        "V2_01_easy", "V2_02_medium",
    ],
    "val": [
        "MH_04_difficult", "V1_03_difficult",
    ],
    "test": [
        "MH_05_difficult", "V2_03_difficult",   # completely unseen
    ],
}

SEQUENCE_PATHS = {
    "MH_01_easy":     r"D:\ML For IMU\machine_hall\MH_01_easy\mav0",
    "MH_02_easy":     r"D:\ML For IMU\machine_hall\MH_02_easy\mav0",
    "MH_03_medium":   r"D:\ML For IMU\machine_hall\MH_03_medium\mav0",
    "MH_04_difficult":r"D:\ML For IMU\machine_hall\MH_04_difficult\mav0",
    "MH_05_difficult":r"D:\ML For IMU\machine_hall\MH_05_difficult\mav0",
    "V1_01_easy":     r"D:\ML For IMU\vicon_room1\V1_01_easy\mav0",
    "V1_02_medium":   r"D:\ML For IMU\vicon_room1\V1_02_medium\mav0",
    "V1_03_difficult":r"D:\ML For IMU\vicon_room1\V1_03_difficult\mav0",
    "V2_01_easy":     r"D:\ML For IMU\vicon_room2\V2_01_easy\mav0",
    "V2_02_medium":   r"D:\ML For IMU\vicon_room2\V2_02_medium\mav0",
    "V2_03_difficult":r"D:\ML For IMU\vicon_room2\V2_03_difficult\mav0",
}


def load_euroc_sequence(mav0_path: str, min_gt_overlap: float = 0.5):
    """
    Load and synchronise one EuRoC sequence.

    Returns a dict with aligned arrays:
      imu_ts       : (N,)   timestamp [ns]
      gyro         : (N, 3) [rad/s]
      accel        : (N, 3) [m/s²]
      pos_gt       : (N, 3) [m]  world-frame position (interpolated to IMU rate)
      vel_gt       : (N, 3) [m/s] world-frame velocity (interpolated)
      rot_matrices : (N, 3, 3) rotation matrices R_body_to_world
      features_12d : (N, 12) [ω, a, dω/dt, ω×ω]
    """
    imu_path = os.path.join(mav0_path, "imu0", "data.csv")
    gt_path  = os.path.join(mav0_path, "state_groundtruth_estimate0", "data.csv")

    # ── Load IMU ──────────────────────────────────────────────────────────────
    imu_df = pd.read_csv(imu_path)
    imu_df.columns = ["ts", "wx", "wy", "wz", "ax", "ay", "az"]
    imu_df = imu_df.sort_values("ts").reset_index(drop=True)

    # ── Load ground truth ─────────────────────────────────────────────────────
    gt_df = pd.read_csv(gt_path)
    gt_df.columns = ["ts", "px", "py", "pz",
                     "qw", "qx", "qy", "qz",
                     "vx", "vy", "vz",
                     "bwx", "bwy", "bwz", "bax", "bay", "baz"]
    gt_df = gt_df.sort_values("ts").reset_index(drop=True)

    # ── Clip to overlapping timestamps ────────────────────────────────────────
    t_start = max(imu_df["ts"].iloc[0],  gt_df["ts"].iloc[0])
    t_end   = min(imu_df["ts"].iloc[-1], gt_df["ts"].iloc[-1])

    imu_df = imu_df[(imu_df["ts"] >= t_start) & (imu_df["ts"] <= t_end)].reset_index(drop=True)
    gt_df  = gt_df[ (gt_df["ts"]  >= t_start) & (gt_df["ts"]  <= t_end)].reset_index(drop=True)

    if len(imu_df) < 100 or len(gt_df) < 10:
        raise ValueError(f"Insufficient overlap in {mav0_path}")

    imu_ts = imu_df["ts"].values.astype(np.float64)
    gt_ts  = gt_df["ts"].values.astype(np.float64)

    # ── Interpolate GT to IMU timestamps ──────────────────────────────────────
    def interp_col(col):
        return np.interp(imu_ts, gt_ts, gt_df[col].values)

    pos_gt = np.column_stack([interp_col("px"), interp_col("py"), interp_col("pz")])
    vel_gt = np.column_stack([interp_col("vx"), interp_col("vy"), interp_col("vz")])

    # Interpolate quaternion (slerp via scipy)
    quats_gt = gt_df[["qw", "qx", "qy", "qz"]].values  # (M, 4) wxyz
    # scipy expects xyzw
    quats_xyzw = np.column_stack([quats_gt[:, 1], quats_gt[:, 2],
                                   quats_gt[:, 3], quats_gt[:, 0]])
    rot_gt = Rotation.from_quat(quats_xyzw)
    times_norm  = (imu_ts - gt_ts[0]) / (gt_ts[-1] - gt_ts[0] + 1e-9)
    times_norm_gt = (gt_ts - gt_ts[0]) / (gt_ts[-1] - gt_ts[0] + 1e-9)

    # Use nearest-index interpolation for rotation (fast, accurate enough at 200 Hz)
    nearest_idx = np.searchsorted(times_norm_gt, times_norm).clip(0, len(gt_ts) - 1)
    rot_matrices = rot_gt[nearest_idx].as_matrix()   # (N, 3, 3)

    # ── Raw IMU arrays ─────────────────────────────────────────────────────────
    gyro  = imu_df[["wx", "wy", "wz"]].values.astype(np.float32)
    accel = imu_df[["ax", "ay", "az"]].values.astype(np.float32)

    # ── 12-Channel Torque Feature Computation ─────────────────────────────────
    dt = np.diff(imu_ts) / 1e9  # seconds (N-1,)
    dt = np.append(dt, dt[-1])  # pad to (N,)

    dw = np.gradient(gyro, axis=0) / (dt[:, None] + 1e-9)      # dω/dt   (N,3)
    tau_gyro = np.cross(gyro, gyro)                              # ω×ω ≡ 0 for self-cross
    # gyroscopic coupling: ω × (J·ω) — approximate with ω × ω_scaled
    tau_gyro = np.column_stack([
        gyro[:, 1] * gyro[:, 2],
        gyro[:, 2] * gyro[:, 0],
        gyro[:, 0] * gyro[:, 1],
    ])  # (N, 3) cross-axis coupling terms

    features_12d = np.hstack([gyro, accel, dw, tau_gyro]).astype(np.float32)

    return {
        "imu_ts":       imu_ts,
        "gyro":         gyro,
        "accel":        accel,
        "pos_gt":       pos_gt.astype(np.float32),
        "vel_gt":       vel_gt.astype(np.float32),
        "rot_matrices": rot_matrices.astype(np.float32),
        "features_12d": features_12d,
    }


def build_windows(seq_data: dict, window_size: int = 40, step: int = 2,
                  augment_yaw: bool = False, n_augments: int = 3):
    """
    Build sliding-window (X, y) pairs from one sequence.

    X : (n_windows, 12, window_size) — 12-channel feature windows (channels first for Conv1d)
    y : (n_windows, 3)               — body-frame velocity target at window end
    """
    feat   = seq_data["features_12d"]   # (N, 12)
    vel_gt = seq_data["vel_gt"]          # (N, 3) world-frame
    R_mats = seq_data["rot_matrices"]    # (N, 3, 3)
    N = len(feat)

    X_list, y_list = [], []

    # Canonical windows (no augmentation)
    for i in range(0, N - window_size, step):
        win    = feat[i:i + window_size]           # (W, 12)
        R_end  = R_mats[i + window_size - 1]       # (3, 3)
        v_world = vel_gt[i + window_size - 1]      # (3,)
        v_body  = R_end.T @ v_world                # rotate to body frame
        X_list.append(win.T)                        # (12, W) channels-first
        y_list.append(v_body)

    if augment_yaw and n_augments > 0:
        # Random yaw rotations — only rotates the horizontal plane
        rng = np.random.default_rng(42)
        yaw_angles = rng.uniform(0, 2 * np.pi, n_augments)
        for yaw in yaw_angles:
            cy, sy = np.cos(yaw), np.sin(yaw)
            Rz = np.array([[cy, -sy, 0],
                           [sy,  cy, 0],
                           [0,   0,  1]], dtype=np.float32)
            for i in range(0, N - window_size, step):
                win     = feat[i:i + window_size].copy()
                # Rotate ω and a channels (0:3 and 3:6) by Rz
                win[:, 0:3] = (Rz @ win[:, 0:3].T).T
                win[:, 3:6] = (Rz @ win[:, 3:6].T).T
                # Recompute dω and tau from rotated gyro (approximate)
                win[:, 6:9]  = (Rz @ win[:, 6:9].T).T
                win[:, 9:12] = (Rz @ win[:, 9:12].T).T

                R_end  = Rz @ R_mats[i + window_size - 1]
                v_world = vel_gt[i + window_size - 1]
                v_body  = R_end.T @ v_world
                X_list.append(win.T)
                y_list.append(v_body)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y


def load_split(split: str, window_size: int = 40, step: int = 2,
               augment_yaw: bool = False, n_augments: int = 3,
               verbose: bool = True):
    """
    Load all sequences for a given split ('train', 'val', 'test').
    Returns concatenated (X, y) arrays and a list of per-sequence metadata.
    """
    seqs = SEQUENCE_SPLIT[split]
    X_all, y_all, meta = [], [], []

    for name in seqs:
        mav0 = SEQUENCE_PATHS[name]
        try:
            seq = load_euroc_sequence(mav0)
            aug = augment_yaw and (split == "train")
            X, y = build_windows(seq, window_size=window_size, step=step,
                                  augment_yaw=aug, n_augments=n_augments)
            X_all.append(X)
            y_all.append(y)
            meta.append({
                "name": name, "n_imu": len(seq["imu_ts"]),
                "n_windows": len(X), "split": split
            })
            if verbose:
                print(f"  [{split:5s}] {name:20s} -> {len(seq['imu_ts']):6d} IMU samples, "
                      f"{len(X):6d} windows (aug={aug})")
        except Exception as e:
            print(f"  [WARN] Failed to load {name}: {e}")

    X_cat = np.concatenate(X_all, axis=0)
    y_cat = np.concatenate(y_all, axis=0)
    return X_cat, y_cat, meta
