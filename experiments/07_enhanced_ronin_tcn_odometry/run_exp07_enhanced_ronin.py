"""
Experiment 7: Multi-Scale Dilated TCN with End-to-End Window Integration Loss.
Pure IMU Dead-Reckoning (Bypassing Both Double-Integration and Systematic Velocity Bias).

Key Innovations:
  1. Multi-Scale Dilated Convolutions (dilation rates d=1, 2, 4 with W=80 temporal context = 0.40s)
  2. End-to-End Window Integration Loss: Penalizes relative displacement error across multi-step horizons
  3. Systematic Velocity Bias Regularizer: Drives mean velocity residual to zero, preventing unidirectional drift
  4. Pure IMU Dead-Reckoning (0% GNSS, 0% Camera, 0% Sensor Fusion during evaluation)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from enhanced_ronin_model import (
    MultiScaleDilatedTCN,
    IntegrationLoss,
    EnhancedRoNINDatasetManager,
    ChunkDataset
)

# Matplotlib Publication Styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


def calc_metrics(est_pos: np.ndarray, gt_pos: np.ndarray):
    err = est_pos - gt_pos
    euc_err = np.sqrt(np.sum(err**2, axis=1))
    ate_rmse = float(np.sqrt(np.mean(euc_err**2)))
    rmse_xyz = np.sqrt(np.mean(err**2, axis=0))

    return {
        'ate_rmse_m': ate_rmse,
        'mean_error_m': float(np.mean(euc_err)),
        'median_error_m': float(np.median(euc_err)),
        'max_error_m': float(np.max(euc_err)),
        'rmse_x_m': float(rmse_xyz[0]),
        'rmse_y_m': float(rmse_xyz[1]),
        'rmse_z_m': float(rmse_xyz[2]),
        'euc_errors': euc_err
    }


def main():
    exp_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(exp_dir, '..', '..', 'MH_01_easy.bag', 'extracted_ros1_data'))

    print("==========================================================================")
    print("  EXPERIMENT 7: MULTI-SCALE DILATED TCN + END-TO-END INTEGRATION LOSS   ")
    print("==========================================================================")
    print("Objective: Eliminate systematic velocity drift via multi-step displacement loss.")
    print("Architecture: Dilated Convolutions (d=1, 2, 4) with W=80 (0.40s context).")
    print("--------------------------------------------------------------------------")

    imu_df = pd.read_csv(os.path.join(data_dir, 'imu.csv'))
    gt_df = pd.read_csv(os.path.join(data_dir, 'ground_truth.csv'))

    # 1. Dataset Preparation
    ds_manager = EnhancedRoNINDatasetManager(window_size=80, step=2, chunk_len=16)
    prep = ds_manager.prepare_data(imu_df, gt_df)

    windows = prep['windows']
    targets_body = prep['targets_body']
    indices = prep['indices']
    chunk_windows = prep['chunk_windows']
    chunk_targets = prep['chunk_targets']
    chunk_dts = prep['chunk_dts']
    imu_ts = prep['imu_ts']
    pos_gt = prep['pos_gt']
    vel_gt_world = prep['vel_gt_world']
    rot_matrices = prep['rot_matrices']

    n_chunks = len(chunk_windows)
    n_train_chunks = int(n_chunks * 0.70)
    n_val_chunks = int(n_chunks * 0.15)
    n_test_chunks = n_chunks - n_train_chunks - n_val_chunks

    print(f"Dataset Split (Chronological Sequences):")
    print(f"  Training Chunks:   {n_train_chunks} sequence chunks (70.0%)")
    print(f"  Validation Chunks: {n_val_chunks} sequence chunks (15.0%)")
    print(f"  Test Chunks:       {n_test_chunks} sequence chunks (15.0%) [Unseen]")

    X_tr = torch.tensor(chunk_windows[:n_train_chunks])
    y_tr = torch.tensor(chunk_targets[:n_train_chunks])
    dt_tr = torch.tensor(chunk_dts[:n_train_chunks])

    X_val = torch.tensor(chunk_windows[n_train_chunks:n_train_chunks+n_val_chunks])
    y_val = torch.tensor(chunk_targets[n_train_chunks:n_train_chunks+n_val_chunks])
    dt_val = torch.tensor(chunk_dts[n_train_chunks:n_train_chunks+n_val_chunks])

    train_loader = DataLoader(ChunkDataset(X_tr, y_tr, dt_tr), batch_size=32, shuffle=True)
    val_loader = DataLoader(ChunkDataset(X_val, y_val, dt_val), batch_size=64, shuffle=False)

    # 2. Model, Optimizer, and Loss
    model = MultiScaleDilatedTCN(in_channels=8, out_dim=3)
    loss_fn = IntegrationLoss(lambda_disp=2.0, lambda_bias=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    print("\nTraining Multi-Scale Dilated TCN with Integration & Bias Loss...")
    t0_train = time.time()
    best_val_loss = float('inf')
    best_weights = None
    train_loss_hist = []
    val_loss_hist = []

    for epoch in range(10):
        model.train()
        running_train_loss = 0.0
        for bx, by, bdt in train_loader:
            B, K, C, W_len = bx.shape
            bx_flat = bx.view(B * K, C, W_len)
            preds_flat = model(bx_flat)
            preds = preds_flat.view(B, K, 3)

            loss, l_vel, l_disp, l_bias = loss_fn(preds, by, bdt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * len(bx)
        scheduler.step()

        train_epoch_loss = running_train_loss / len(X_tr)
        train_loss_hist.append(train_epoch_loss)

        # Validation
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for bx, by, bdt in val_loader:
                B, K, C, W_len = bx.shape
                bx_flat = bx.view(B * K, C, W_len)
                preds = model(bx_flat).view(B, K, 3)
                loss, _, l_disp, _ = loss_fn(preds, by, bdt)
                running_val_loss += loss.item() * len(bx)
        val_epoch_loss = running_val_loss / len(X_val)
        val_loss_hist.append(val_epoch_loss)

        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"  Epoch [{epoch+1:02d}/10] - Train Loss: {train_epoch_loss:.5f} | Val Loss: {val_epoch_loss:.5f}", flush=True)

    t_train_duration = time.time() - t0_train
    print(f"Training converged in {t_train_duration:.2f}s (Best Val Loss: {best_val_loss:.5f})")

    # Save model checkpoint
    model.load_state_dict(best_weights)
    torch.save(best_weights, os.path.join(exp_dir, 'best_enhanced_ronin_model.pth'))
    model.eval()

    # 3. Pure IMU Dead-Reckoning Inference (0% Sensor Fusion / 0% GNSS)
    print("\nRunning Pure IMU Dead-Reckoning Inference...")
    with torch.no_grad():
        all_w = torch.tensor(windows)
        test_loader_all = DataLoader(TensorDataset(all_w), batch_size=256, shuffle=False)
        all_preds = []
        for (bw,) in test_loader_all:
            all_preds.append(model(bw).numpy())
        preds_body = np.vstack(all_preds)

    # Rotate body velocity to world navigation frame
    preds_world = np.zeros_like(preds_body)
    for k, idx in enumerate(indices):
        preds_world[k] = rot_matrices[idx] @ preds_body[k]

    # Single integration
    gt_matched_pos = pos_gt[indices]
    pred_pos = np.zeros((len(preds_world), 3))
    pred_pos[0] = gt_matched_pos[0]

    for k in range(1, len(preds_world)):
        dt_step = (imu_ts[indices[k]] - imu_ts[indices[k-1]]) / 1e9
        pred_pos[k] = pred_pos[k-1] + preds_world[k] * dt_step

    t_sec_series = (imu_ts[indices] - imu_ts[0]) / 1e9

    # 4. Metrics Evaluation
    full_metrics = calc_metrics(pred_pos, gt_matched_pos)

    # Test set portion (last 15% unseen sequence)
    n_total_windows = len(windows)
    idx_test_start = int(n_total_windows * 0.85)
    test_metrics = calc_metrics(pred_pos[idx_test_start:], gt_matched_pos[idx_test_start:])

    # Baseline comparisons
    b1_drift = 73768.63 # Classical Pure IMU
    exp06_full_drift = 4.4171 # Exp 06 Standard RoNIN
    exp06_test_drift = 9.4099

    improvement_over_exp06_full = (1.0 - (full_metrics['ate_rmse_m'] / exp06_full_drift)) * 100.0
    improvement_over_exp06_test = (1.0 - (test_metrics['ate_rmse_m'] / exp06_test_drift)) * 100.0
    drift_reduction_factor_b1 = b1_drift / full_metrics['ate_rmse_m']

    print("\n==========================================================================")
    print("                     EXPERIMENT 7 QUANTITATIVE RESULTS                    ")
    print("==========================================================================")
    print(f"Classical Pure IMU (Baseline 1):                     {b1_drift:,.2f} m (73.7 km)")
    print(f"Standard RoNIN TCN (Experiment 6 Full):              {exp06_full_drift:.4f} m (4.42 m)")
    print(f"Enhanced Dilated RoNIN (Experiment 7 Full):          {full_metrics['ate_rmse_m']:.4f} m")
    print(f"Drift Reduction vs Exp 06 (Full Trajectory):         {improvement_over_exp06_full:+.2f}%")
    print(f"Drift Reduction vs Baseline 1:                       {drift_reduction_factor_b1:,.0f}x Error Reduction! 🚀")
    print("--------------------------------------------------------------------------")
    print(f"Held-Out Unseen Test Set (Last 15% Sequence):")
    print(f"  Exp 06 Standard RoNIN Test ATE:                    {exp06_test_drift:.4f} m")
    print(f"  Exp 07 Enhanced Dilated RoNIN Test ATE:            {test_metrics['ate_rmse_m']:.4f} m")
    print(f"  Test Set Improvement vs Exp 06:                   {improvement_over_exp06_test:+.2f}%")
    print(f"  X-Axis RMSE:                                       {test_metrics['rmse_x_m']:.4f} m")
    print(f"  Y-Axis RMSE:                                       {test_metrics['rmse_y_m']:.4f} m")
    print(f"  Z-Axis RMSE:                                       {test_metrics['rmse_z_m']:.4f} m")
    print("==========================================================================")

    # 5. Save Results JSON
    results = {
        'experiment': '07_enhanced_ronin_tcn_odometry',
        'model_architecture': 'Multi-Scale Dilated TCN (d=1, 2, 4) with End-to-End Integration Loss',
        'external_sensor_updates_during_eval': 0,
        'training_time_sec': t_train_duration,
        'dataset_split': {
            'train_percent': 70,
            'val_percent': 15,
            'test_percent': 15,
            'total_windows': n_total_windows,
            'window_size_samples': 80,
            'window_duration_sec': 0.40
        },
        'comparison_with_baselines': {
            'baseline_1_classical_pure_imu_ate_m': b1_drift,
            'exp_06_standard_ronin_full_ate_m': exp06_full_drift,
            'exp_06_standard_ronin_test_ate_m': exp06_test_drift,
            'exp_07_enhanced_ronin_full_ate_m': full_metrics['ate_rmse_m'],
            'exp_07_enhanced_ronin_test_ate_m': test_metrics['ate_rmse_m'],
            'improvement_over_exp06_full_percent': improvement_over_exp06_full,
            'improvement_over_exp06_test_percent': improvement_over_exp06_test,
            'drift_reduction_factor_over_b1': drift_reduction_factor_b1
        },
        'full_trajectory_metrics': {
            'ate_rmse_m': full_metrics['ate_rmse_m'],
            'mean_error_m': full_metrics['mean_error_m'],
            'median_error_m': full_metrics['median_error_m'],
            'max_error_m': full_metrics['max_error_m'],
            'rmse_x_m': full_metrics['rmse_x_m'],
            'rmse_y_m': full_metrics['rmse_y_m'],
            'rmse_z_m': full_metrics['rmse_z_m']
        },
        'held_out_test_set_metrics': {
            'ate_rmse_m': test_metrics['ate_rmse_m'],
            'mean_error_m': test_metrics['mean_error_m'],
            'median_error_m': test_metrics['median_error_m'],
            'max_error_m': test_metrics['max_error_m'],
            'rmse_x_m': test_metrics['rmse_x_m'],
            'rmse_y_m': test_metrics['rmse_y_m'],
            'rmse_z_m': test_metrics['rmse_z_m']
        }
    }

    with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Save trajectory CSV
    traj_df = pd.DataFrame({
        'timestamp_ns': imu_ts[indices],
        'time_sec': t_sec_series,
        'p_x_est': pred_pos[:, 0],
        'p_y_est': pred_pos[:, 1],
        'p_z_est': pred_pos[:, 2],
        'p_x_gt': gt_matched_pos[:, 0],
        'p_y_gt': gt_matched_pos[:, 1],
        'p_z_gt': gt_matched_pos[:, 2],
        'v_x_pred': preds_world[:, 0],
        'v_y_pred': preds_world[:, 1],
        'v_z_pred': preds_world[:, 2],
        'v_x_gt': vel_gt_world[indices, 0],
        'v_y_gt': vel_gt_world[indices, 1],
        'v_z_gt': vel_gt_world[indices, 2],
        'error_euclidean_m': full_metrics['euc_errors']
    })
    traj_df.to_csv(os.path.join(exp_dir, 'enhanced_trajectory_ronin.csv'), index=False)

    # 6. Generate 6 Publication-Quality Figures
    plots_dir = os.path.join(exp_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Load Exp 06 trajectory for direct visual comparison
    exp06_dir = os.path.join(exp_dir, '..', '06_learned_inertial_odometry_ronin')
    exp06_traj = pd.read_csv(os.path.join(exp06_dir, 'trajectory_ronin.csv'))

    # Figure 1: Training & Validation Loss
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
    epochs_range = range(1, len(train_loss_hist) + 1)
    ax.plot(epochs_range, train_loss_hist, 'b-o', linewidth=2.0, label='Training Compound Loss')
    ax.plot(epochs_range, val_loss_hist, 'r--s', linewidth=2.0, label='Validation Compound Loss')
    ax.set_title("Exp 7: Multi-Scale Dilated TCN Training & Validation Loss", fontsize=13, fontweight='bold')
    ax.set_xlabel("Epoch", fontweight='bold')
    ax.set_ylabel("Compound Loss (Velocity + Disp + Bias)", fontweight='bold')
    ax.set_xticks(epochs_range)
    ax.legend(frameon=True, loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.6)
    fig.tight_layout()
    plt.savefig(os.path.join(plots_dir, '01_enhanced_ronin_loss_curves.png'))
    plt.close()

    # Figure 2: 3D Trajectory Overlay (Exp 6 vs Exp 7 vs GT)
    fig = plt.figure(figsize=(10, 7.5), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(gt_matched_pos[:, 0], gt_matched_pos[:, 1], gt_matched_pos[:, 2], 'k-', label='Ground Truth Reference', linewidth=2.2, alpha=0.9)
    ax.plot(exp06_traj['p_x_est'], exp06_traj['p_y_est'], exp06_traj['p_z_est'], color='#999999', linestyle=':', label=f'Exp 06 Standard RoNIN (ATE: {exp06_full_drift:.2f} m)', linewidth=1.5)
    ax.plot(pred_pos[:, 0], pred_pos[:, 1], pred_pos[:, 2], color='#1f77b4', linestyle='-', label=f'Exp 07 Enhanced Dilated RoNIN (ATE: {full_metrics["ate_rmse_m"]:.2f} m)', linewidth=2.0)
    ax.scatter(pred_pos[0, 0], pred_pos[0, 1], pred_pos[0, 2], color='green', s=70, label='Start Point', zorder=5)
    ax.scatter(pred_pos[-1, 0], pred_pos[-1, 1], pred_pos[-1, 2], color='red', s=70, label='End Point', zorder=5)
    ax.set_title("Exp 7: 3D Trajectory Tracking (Enhanced Dilated RoNIN vs Baseline RoNIN)", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("X Position [m]", fontweight='bold')
    ax.set_ylabel("Y Position [m]", fontweight='bold')
    ax.set_zlabel("Z Position [m]", fontweight='bold')
    ax.legend(frameon=True, loc='upper right')
    fig.tight_layout()
    plt.savefig(os.path.join(plots_dir, '02_3d_trajectory_enhanced_vs_baseline_ronin_and_gt.png'))
    plt.close()

    # Figure 3: Per-Axis Position Tracking Over Time (Exp 6 vs Exp 7 vs GT)
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, dpi=300)
    axes[0].plot(t_sec_series, gt_matched_pos[:, 0], 'k-', label='Ground Truth X', linewidth=1.8, alpha=0.8)
    axes[0].plot(exp06_traj['time_sec'], exp06_traj['p_x_est'], 'gray', linestyle=':', label='Exp 06 RoNIN X', linewidth=1.2)
    axes[0].plot(t_sec_series, pred_pos[:, 0], color='#1f77b4', label=f'Exp 07 Enhanced X ({full_metrics["rmse_x_m"]:.2f}m)', linewidth=1.5)
    axes[0].set_ylabel("X [m]", fontweight='bold')
    axes[0].legend(loc='upper right', frameon=True)
    axes[0].grid(True, linestyle=':', alpha=0.6)

    axes[1].plot(t_sec_series, gt_matched_pos[:, 1], 'k-', label='Ground Truth Y', linewidth=1.8, alpha=0.8)
    axes[1].plot(exp06_traj['time_sec'], exp06_traj['p_y_est'], 'gray', linestyle=':', label='Exp 06 RoNIN Y', linewidth=1.2)
    axes[1].plot(t_sec_series, pred_pos[:, 1], color='#1f77b4', label=f'Exp 07 Enhanced Y ({full_metrics["rmse_y_m"]:.2f}m)', linewidth=1.5)
    axes[1].set_ylabel("Y [m]", fontweight='bold')
    axes[1].legend(loc='upper right', frameon=True)
    axes[1].grid(True, linestyle=':', alpha=0.6)

    axes[2].plot(t_sec_series, gt_matched_pos[:, 2], 'k-', label='Ground Truth Z', linewidth=1.8, alpha=0.8)
    axes[2].plot(exp06_traj['time_sec'], exp06_traj['p_z_est'], 'gray', linestyle=':', label='Exp 06 RoNIN Z', linewidth=1.2)
    axes[2].plot(t_sec_series, pred_pos[:, 2], color='#1f77b4', label=f'Exp 07 Enhanced Z ({full_metrics["rmse_z_m"]:.2f}m)', linewidth=1.5)
    axes[2].set_ylabel("Z [m]", fontweight='bold')
    axes[2].set_xlabel("Time [s]", fontweight='bold')
    axes[2].legend(loc='upper right', frameon=True)
    axes[2].grid(True, linestyle=':', alpha=0.6)

    fig.suptitle("Exp 7: Per-Axis Position Tracking Over Time (Exp 6 vs Exp 7)", fontsize=14, fontweight='bold')
    fig.tight_layout()
    plt.savefig(os.path.join(plots_dir, '03_per_axis_position_tracking_comparison.png'))
    plt.close()

    # Figure 4: Velocity Profile & Residual Velocity Error Distribution
    fig, (ax_v, ax_hist) = plt.subplots(1, 2, figsize=(14, 5.0), dpi=300)
    # Subplot A: Velocity profile overlay
    ax_v.plot(t_sec_series, vel_gt_world[indices, 0], 'k-', label='True v_x', linewidth=1.3, alpha=0.7)
    ax_v.plot(t_sec_series, preds_world[:, 0], color='#1f77b4', label='Enhanced v_x', linewidth=1.2)
    ax_v.set_title("3D Velocity Tracking (v_x Component)", fontsize=12, fontweight='bold')
    ax_v.set_xlabel("Time [s]", fontweight='bold')
    ax_v.set_ylabel("Velocity [m/s]", fontweight='bold')
    ax_v.legend(frameon=True, loc='upper right')
    ax_v.grid(True, linestyle=':', alpha=0.6)

    # Subplot B: Residual Error Distribution (Exp 6 vs Exp 7)
    res_exp06 = exp06_traj['error_euclidean_m'].values
    res_exp07 = full_metrics['euc_errors']
    ax_hist.hist(res_exp06, bins=40, density=True, alpha=0.5, color='gray', label=f'Exp 06 Error (Mean: {np.mean(res_exp06):.2f}m)')
    ax_hist.hist(res_exp07, bins=40, density=True, alpha=0.6, color='#1f77b4', label=f'Exp 07 Error (Mean: {np.mean(res_exp07):.2f}m)')
    ax_hist.set_title("Position Error Probability Density Distribution", fontsize=12, fontweight='bold')
    ax_hist.set_xlabel("Euclidean Position Error [m]", fontweight='bold')
    ax_hist.set_ylabel("Density", fontweight='bold')
    ax_hist.legend(frameon=True, loc='upper right')
    ax_hist.grid(True, linestyle=':', alpha=0.6)

    fig.suptitle("Exp 7: Velocity Tracking & Residual Error Distribution", fontsize=14, fontweight='bold')
    fig.tight_layout()
    plt.savefig(os.path.join(plots_dir, '04_velocity_profile_and_residual_error.png'))
    plt.close()

    # Figure 5: Position Drift Over Time (Exp 6 vs Exp 7)
    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=300)
    ax.plot(exp06_traj['time_sec'], exp06_traj['error_euclidean_m'], color='gray', linestyle='--', linewidth=1.5, label=f'Exp 06 Standard RoNIN Error (RMSE: {exp06_full_drift:.2f}m)')
    ax.plot(t_sec_series, full_metrics['euc_errors'], color='#1f77b4', linewidth=1.8, label=f'Exp 07 Enhanced Dilated RoNIN Error (RMSE: {full_metrics["ate_rmse_m"]:.2f}m)')
    t_test_split_sec = t_sec_series[idx_test_start]
    ax.axvline(t_test_split_sec, color='purple', linestyle='--', linewidth=1.8, label=f'Held-Out Test Split Start ({t_test_split_sec:.1f}s)')
    ax.set_title("Exp 7: Direct Drift Over Time Comparison (Exp 6 vs Exp 7)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Time [s]", fontweight='bold')
    ax.set_ylabel("Position Error [m]", fontweight='bold')
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper left', frameon=True)
    ax.grid(True, linestyle=':', alpha=0.6)
    fig.tight_layout()
    plt.savefig(os.path.join(plots_dir, '05_position_drift_over_time_exp6_vs_exp7.png'))
    plt.close()

    # Figure 6: All Experiments Evolutionary Benchmark
    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=300)
    configs = [
        'Exp 01: GT EKF\n(Upper Bound)',
        'Exp 02: 1Hz GNSS\nTrad EKF',
        'Exp 05: NN-EKF\n(Test Set)',
        'Exp 06: RoNIN\n(Pure IMU)',
        'Exp 07: Enhanced\nDilated RoNIN'
    ]
    vals = [0.0343, 1.257, 0.8998, exp06_full_drift, full_metrics['ate_rmse_m']]
    colors = ['#2ca02c', '#ff7f0e', '#9467bd', '#e377c2', '#1f77b4']

    bars = ax.bar(configs, vals, color=colors, width=0.45, edgecolor='black', linewidth=1.0)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f"{yval:.3f} m", ha='center', va='bottom', fontweight='bold')

    ax.set_ylabel("ATE RMSE [m]", fontweight='bold')
    ax.set_title("Complete Evolutionary Suite: From Upper Bound to Pure IMU Deep Learning", fontsize=13, fontweight='bold')
    ax.grid(True, axis='y', linestyle=':', alpha=0.6)
    fig.tight_layout()
    plt.savefig(os.path.join(plots_dir, '06_all_experiments_evolutionary_benchmark.png'))
    plt.close()

    print(f"\nGenerated all 6 publication figures in: {plots_dir}")


if __name__ == '__main__':
    main()

