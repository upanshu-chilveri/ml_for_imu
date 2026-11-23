"""
Experiment 5: Neural Network Enhanced Extended Kalman Filter (NN-EKF).
Trained strictly using IMU readings and GNSS position fixes (0% Ground Truth used in model training).

Chronological Dataset Split:
  - Training Set: First 70% of trajectory
  - Validation Set: Next 15% of trajectory
  - Test Set: Final 15% held-out test sequence

Compares Traditional EKF vs Neural-Network EKF (NN-EKF).
"""

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from nn_imu_model import NeuralIMUCorrector

# Matplotlib styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


def skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


class ErrorStateEKF:
    def __init__(self, p0, v0, R0, sigma_a=0.08, sigma_g=0.005, sigma_gnss=0.8, g_val=9.81):
        self.p = np.array(p0, dtype=np.float64)
        self.v = np.array(v0, dtype=np.float64)
        self.rot = R.from_matrix(R0)
        self.ba = np.zeros(3, dtype=np.float64)
        self.bg = np.zeros(3, dtype=np.float64)
        self.g_vec = np.array([0.0, 0.0, -g_val])

        self.sigma_a = sigma_a
        self.sigma_g = sigma_g
        self.sigma_gnss = sigma_gnss

        self.P = np.eye(15, dtype=np.float64) * 1e-2
        self.P[3:6, 3:6] *= 10.0
        self.P[6:9, 6:9] *= 10.0

    def predict(self, w_m, a_m, dt):
        if dt <= 0 or dt > 1.0: return
        w_hat = w_m - self.bg
        a_hat = a_m - self.ba

        R_mat = self.rot.as_matrix()
        a_world = R_mat @ a_hat + self.g_vec

        self.p += self.v * dt + 0.5 * a_world * (dt ** 2)
        self.v += a_world * dt
        self.rot = self.rot * R.from_rotvec(w_hat * dt)

        Fc = np.zeros((15, 15), dtype=np.float64)
        Fc[0:3, 3:6] = np.eye(3)
        Fc[3:6, 6:9] = -R_mat @ skew(a_hat)
        Fc[3:6, 9:12] = -R_mat
        Fc[6:9, 6:9] = -skew(w_hat)
        Fc[6:9, 12:15] = -np.eye(3)

        F_x = np.eye(15) + Fc * dt + 0.5 * (Fc @ Fc) * (dt ** 2)

        Q_d = np.zeros((15, 15), dtype=np.float64)
        np.fill_diagonal(Q_d[0:3, 0:3], (0.001 * dt) ** 2)
        np.fill_diagonal(Q_d[3:6, 3:6], (self.sigma_a * dt) ** 2)
        np.fill_diagonal(Q_d[6:9, 6:9], (self.sigma_g * dt) ** 2)
        np.fill_diagonal(Q_d[9:12, 9:12], (0.001 * np.sqrt(dt)) ** 2)
        np.fill_diagonal(Q_d[12:15, 12:15], (0.0001 * np.sqrt(dt)) ** 2)

        self.P = F_x @ self.P @ F_x.T + Q_d

    def update_position(self, p_gnss, R_m):
        H = np.zeros((3, 15), dtype=np.float64)
        H[0:3, 0:3] = np.eye(3)

        z = p_gnss - self.p
        S = H @ self.P @ H.T + R_m
        K = self.P @ H.T @ np.linalg.inv(S)

        dx = K @ z
        I_KH = np.eye(15) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_m @ K.T

        self.p += dx[0:3]
        self.v += dx[3:6]
        self.rot = self.rot * R.from_rotvec(dx[6:9])
        self.ba += dx[9:12]
        self.bg += dx[12:15]


def run_ekf_simulation(imu_readings: np.ndarray, timestamps_ns: np.ndarray, gnss_events_list, R_gnss, p0, R_init, uncertainties=None):
    ekf = ErrorStateEKF(p0, np.zeros(3), R_init)

    imu_ev = [(timestamps_ns[i], 'imu', imu_readings[i, 0:3], imu_readings[i, 3:6], uncertainties[i] if uncertainties is not None else None) for i in range(len(timestamps_ns))]
    gnss_ev = [(item[0], 'gnss', item[1], None) for item in gnss_events_list]

    all_ev = sorted(imu_ev + gnss_ev, key=lambda x: x[0])
    prev_t = all_ev[0][0]

    history = []
    for ev in all_ev:
        dt = (ev[0] - prev_t) / 1e9
        prev_t = ev[0]

        if ev[1] == 'imu':
            unc = ev[4]
            if unc is not None:
                ekf.sigma_g = float(np.sqrt(unc[0]))
                ekf.sigma_a = float(np.sqrt(unc[3]))
            ekf.predict(ev[2], ev[3], dt)
            history.append({
                'timestamp_ns': ev[0],
                'p_x': ekf.p[0], 'p_y': ekf.p[1], 'p_z': ekf.p[2],
                'v_x': ekf.v[0], 'v_y': ekf.v[1], 'v_z': ekf.v[2],
                'ba_x': ekf.ba[0], 'ba_y': ekf.ba[1], 'ba_z': ekf.ba[2],
                'bg_x': ekf.bg[0], 'bg_y': ekf.bg[1], 'bg_z': ekf.bg[2]
            })
        elif ev[1] == 'gnss':
            ekf.update_position(ev[2], R_gnss)

    return pd.DataFrame(history)


def calc_metrics(res_df, gt_df):
    gt_x = np.interp(res_df['timestamp_ns'], gt_df['timestamp_ns'], gt_df['p_x'])
    gt_y = np.interp(res_df['timestamp_ns'], gt_df['timestamp_ns'], gt_df['p_y'])
    gt_z = np.interp(res_df['timestamp_ns'], gt_df['timestamp_ns'], gt_df['p_z'])
    gt_ref = np.column_stack([gt_x, gt_y, gt_z])
    est = res_df[['p_x', 'p_y', 'p_z']].values

    err = est - gt_ref
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
        'euc_errors': euc_err,
        'gt_ref': gt_ref
    }


def main():
    exp_dir = os.path.dirname(__file__)
    data_dir = os.path.abspath(os.path.join(exp_dir, '..', '..', 'MH_01_easy.bag', 'extracted_ros1_data'))

    imu_df = pd.read_csv(os.path.join(data_dir, 'imu.csv'))
    gt_df = pd.read_csv(os.path.join(data_dir, 'ground_truth.csv'))

    t_start = max(imu_df['timestamp_ns'].min(), gt_df['timestamp_ns'].min())
    t_end = min(imu_df['timestamp_ns'].max(), gt_df['timestamp_ns'].max())

    imu = imu_df[(imu_df['timestamp_ns'] >= t_start) & (imu_df['timestamp_ns'] <= t_end)].reset_index(drop=True)
    gt = gt_df[(gt_df['timestamp_ns'] >= t_start) & (gt_df['timestamp_ns'] <= t_end)].reset_index(drop=True)

    # 1. Generate 1 Hz Simulated GNSS Fixes from GT (GT isolated for evaluation)
    np.random.seed(42)
    t0_ns = gt['timestamp_ns'].iloc[0]
    gt_time_sec = (gt['timestamp_ns'] - t0_ns) / 1e9

    gnss_indices = []
    last_t = -10.0
    for i, t_s in enumerate(gt_time_sec):
        if t_s - last_t >= 1.0:
            gnss_indices.append(i)
            last_t = t_s

    gnss_df = gt.iloc[gnss_indices].copy().reset_index(drop=True)
    gnss_noise_std = np.array([0.8, 0.8, 1.2])
    noise = np.random.normal(0, gnss_noise_std, size=(len(gnss_df), 3))
    gnss_df['gnss_x'] = gnss_df['p_x'] + noise[:, 0]
    gnss_df['gnss_y'] = gnss_df['p_y'] + noise[:, 1]
    gnss_df['gnss_z'] = gnss_df['p_z'] + noise[:, 2]

    gnss_df.to_csv(os.path.join(exp_dir, 'simulated_gnss_data.csv'), index=False)

    # 2. Train Neural Network (0% Ground Truth used in model training!)
    print("\n=========================================================")
    print("  EXPERIMENT 5: NEURAL NETWORK ENHANCED EKF (NN-EKF)     ")
    print("=========================================================")
    print("Train / Val / Test Split -> 70% Train | 15% Val | 15% Test (Held-Out)")
    print("Model Supervision -> 100% IMU + Simulated GNSS (0% Ground Truth)")

    nn_model = NeuralIMUCorrector(window_size=20, hidden_layer_sizes=(128, 64, 32))
    imu_raw, targets, t_sec = nn_model.create_gnss_supervised_targets(imu, gnss_df)
    imu_ts = imu['timestamp_ns'].values[:len(imu_raw)]

    t0_train = time.time()
    (X_tr, y_tr), (X_v, y_v), (X_te, y_te) = nn_model.train_chronological_split(imu_raw, targets, train_ratio=0.70, val_ratio=0.15)
    t_train_duration = time.time() - t0_train

    # Predict NN corrections and uncertainties
    imu_nn_corr, uncertainties = nn_model.predict(imu_raw)

    # Initial state setup
    a_mean = imu_raw[:100, 3:6].mean(axis=0)
    u_acc = a_mean / np.linalg.norm(a_mean)
    u_z = np.array([0, 0, 1.0])
    v_rot = np.cross(u_acc, u_z)
    s_rot = np.linalg.norm(v_rot)
    c_rot = np.dot(u_acc, u_z)
    vx = skew(v_rot)
    R_init = np.eye(3) + vx + (vx @ vx) * ((1 - c_rot) / (s_rot**2))
    p0 = gt[['p_x', 'p_y', 'p_z']].iloc[0].values
    R_gnss = np.diag(gnss_noise_std**2)

    gnss_events_list = [(row.timestamp_ns, np.array([row.gnss_x, row.gnss_y, row.gnss_z])) for row in gnss_df.itertuples()]

    # 3. Run Traditional EKF vs NN-EKF
    print("\n--- Running Traditional EKF Baseline ---")
    trad_df = run_ekf_simulation(imu_raw, imu_ts, gnss_events_list, R_gnss, p0, R_init)
    trad_m = calc_metrics(trad_df, gt)

    print("\n--- Running Neural-Network EKF (NN-EKF) ---")
    nn_df = run_ekf_simulation(imu_nn_corr, imu_ts, gnss_events_list, R_gnss, p0, R_init, uncertainties=uncertainties)
    nn_m = calc_metrics(nn_df, gt)

    # Calculate Test Set Metrics (Last 15% of trajectory)
    idx_test_start = int(len(imu_raw) * 0.85)
    t_test_start_ns = imu_ts[idx_test_start]

    trad_df_test = trad_df[trad_df['timestamp_ns'] >= t_test_start_ns].reset_index(drop=True)
    nn_df_test = nn_df[nn_df['timestamp_ns'] >= t_test_start_ns].reset_index(drop=True)
    gt_test = gt[gt['timestamp_ns'] >= t_test_start_ns].reset_index(drop=True)

    trad_m_test = calc_metrics(trad_df_test, gt_test)
    nn_m_test = calc_metrics(nn_df_test, gt_test)

    improvement_overall = (1.0 - nn_m['ate_rmse_m'] / trad_m['ate_rmse_m']) * 100.0
    improvement_test = (1.0 - nn_m_test['ate_rmse_m'] / trad_m_test['ate_rmse_m']) * 100.0

    print("\n==========================================================================")
    print("             EXP 5: TRADITIONAL EKF VS NEURAL-NETWORK EKF (NN-EKF)        ")
    print("==========================================================================")
    print(f"Overall Full Trajectory Accuracy:")
    print(f"  Traditional EKF ATE RMSE: {trad_m['ate_rmse_m']:.4f} m ({trad_m['ate_rmse_m']*100:.1f} cm)")
    print(f"  Neural-Network EKF (NN-EKF): {nn_m['ate_rmse_m']:.4f} m ({nn_m['ate_rmse_m']*100:.1f} cm)")
    print(f"  Overall Accuracy Improvement: {improvement_overall:+.1f}%")
    print("--------------------------------------------------------------------------")
    print(f"Held-Out Test Set (Last 15% Unseen Sequence):")
    print(f"  Traditional EKF Test RMSE: {trad_m_test['ate_rmse_m']:.4f} m ({trad_m_test['ate_rmse_m']*100:.1f} cm)")
    print(f"  NN-EKF Test RMSE:          {nn_m_test['ate_rmse_m']:.4f} m ({nn_m_test['ate_rmse_m']*100:.1f} cm)")
    print(f"  Test Set Accuracy Improvement: {improvement_test:+.1f}%")
    print("==========================================================================")

    # Save JSON summary
    res_summary = {
        'experiment': '05_nn_ekf_gnss_learning',
        'training_time_sec': t_train_duration,
        'dataset_split': {'train_percent': 70, 'val_percent': 15, 'test_percent': 15},
        'ground_truth_used_in_model': False,
        'overall_results': {
            'traditional_ekf_ate_m': trad_m['ate_rmse_m'],
            'nn_ekf_ate_m': nn_m['ate_rmse_m'],
            'improvement_percent': improvement_overall
        },
        'held_out_test_set_results': {
            'traditional_ekf_ate_m': trad_m_test['ate_rmse_m'],
            'nn_ekf_ate_m': nn_m_test['ate_rmse_m'],
            'improvement_percent': improvement_test,
            'nn_ekf_rmse_x_m': nn_m_test['rmse_x_m'],
            'nn_ekf_rmse_y_m': nn_m_test['rmse_y_m'],
            'nn_ekf_rmse_z_m': nn_m_test['rmse_z_m']
        }
    }

    with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
        json.dump(res_summary, f, indent=2)

    nn_df.to_csv(os.path.join(exp_dir, 'nn_ekf_fused_trajectory.csv'), index=False)

    # Generate 6 Visualization Figures
    plots_dir = os.path.join(exp_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    t_sec_df = (trad_df['timestamp_ns'] - t_start) / 1e9
    t_sec_gt = (gt['timestamp_ns'] - t_start) / 1e9

    # Figure 1: Training & Validation Loss
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
    ax.plot(nn_model.train_loss_history, color='#1f77b4', linewidth=1.8, label='Training Huber Loss')
    if len(nn_model.val_loss_history) > 0:
        ax.plot(nn_model.val_loss_history, color='#ff7f0e', linestyle='--', linewidth=1.8, label='Validation Loss Score')
    ax.set_title("Exp 5: Neural Network Training & Validation Loss", fontsize=13, fontweight='bold')
    ax.set_xlabel("Epoch / Iteration", fontweight='bold')
    ax.set_ylabel("Loss Score", fontweight='bold')
    ax.legend(loc='upper right', frameon=True)
    ax.grid(True, linestyle=':', alpha=0.6)
    fig.tight_layout()
    plt.savefig(os.path.join(plots_dir, '01_nn_training_validation_loss.png'))
    plt.close()

    # Figure 2: 3D Trajectory Comparison
    fig = plt.figure(figsize=(10, 7.5), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(gt['p_x'], gt['p_y'], gt['p_z'], 'k-', label='Ground Truth Reference', linewidth=2.0)
    ax.plot(trad_df['p_x'], trad_df['p_y'], trad_df['p_z'], 'g--', label=f'Traditional EKF (ATE: {trad_m["ate_rmse_m"]:.2f}m)', linewidth=1.3)
    ax.plot(nn_df['p_x'], nn_df['p_y'], nn_df['p_z'], 'r-', label=f'NN-EKF (ATE: {nn_m["ate_rmse_m"]:.2f}m)', linewidth=1.6)
    ax.set_title("Exp 5: 3D Trajectory Overlay (Traditional EKF vs NN-EKF)", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("X Position [m]", fontweight='bold')
    ax.set_ylabel("Y Position [m]", fontweight='bold')
    ax.set_zlabel("Z Position [m]", fontweight='bold')
    ax.legend(loc='upper right', frameon=True)
    fig.tight_layout()
    plt.savefig(os.path.join(plots_dir, '02_trajectory_3d_comparison_nn_vs_trad.png'))
    plt.close()

    # Figure 3: Per-Axis Tracking Over Time
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, dpi=300)
    components = [('X', 'p_x', trad_m['rmse_x_m'], nn_m['rmse_x_m']),
                  ('Y', 'p_y', trad_m['rmse_y_m'], nn_m['rmse_y_m']),
                  ('Z', 'p_z', trad_m['rmse_z_m'], nn_m['rmse_z_m'])]
    for i, (axis_name, col, rmse_tr, rmse_nn) in enumerate(components):
        axes[i].plot(t_sec_gt, gt[col], 'k-', label='Ground Truth', linewidth=1.8, alpha=0.8)
        axes[i].plot(t_sec_df, trad_df[col], 'g:', label=f'Traditional EKF ({rmse_tr:.2f}m)', linewidth=1.2)
        axes[i].plot(t_sec_df, nn_df[col], 'r-', label=f'NN-EKF ({rmse_nn:.2f}m)', linewidth=1.3)
        axes[i].set_ylabel(f"{axis_name} [m]", fontweight='bold')
        axes[i].legend(loc='upper right', frameon=True)
        axes[i].grid(True, linestyle=':', alpha=0.6)
    axes[2].set_xlabel("Time [s]", fontweight='bold')
    fig.suptitle("Exp 5: Per-Axis Tracking Over Time (Traditional EKF vs NN-EKF)", fontsize=14, fontweight='bold')
    fig.tight_layout()
    plt.savefig(os.path.join(plots_dir, '03_position_components_over_time.png'))
    plt.close()

    # Figure 4: Position Error Over Time Comparison
    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=300)
    ax.plot(t_sec_df, trad_m['euc_errors'], 'g:', label=f'Traditional EKF Error (RMSE: {trad_m["ate_rmse_m"]:.2f}m)', linewidth=1.2, alpha=0.7)
    ax.plot(t_sec_df, nn_m['euc_errors'], 'r-', label=f'NN-EKF Error (RMSE: {nn_m["ate_rmse_m"]:.2f}m)', linewidth=1.4)
    ax.axvline((t_test_start_ns - t_start) / 1e9, color='purple', linestyle='--', label='Held-Out Test Split Start')
    ax.set_title("Exp 5: Absolute Position Error Over Time (Traditional vs NN-EKF)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Time [s]", fontweight='bold')
    ax.set_ylabel("Position Error [m]", fontweight='bold')
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper right', frameon=True)
    ax.grid(True, linestyle=':', alpha=0.6)
    fig.tight_layout()
    plt.savefig(os.path.join(plots_dir, '04_position_error_over_time_comparison.png'))
    plt.close()

    # Figure 5: Test Set Error Breakdown Bar Chart
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    labels = ['Traditional EKF\n(Full Sequence)', 'NN-EKF\n(Full Sequence)', 'Traditional EKF\n(Test Set)', 'NN-EKF\n(Test Set)']
    vals = [trad_m['ate_rmse_m'], nn_m['ate_rmse_m'], trad_m_test['ate_rmse_m'], nn_m_test['ate_rmse_m']]
    colors = ['#2ca02c', '#d62728', '#98df8a', '#ff9896']

    bars = ax.bar(labels, vals, color=colors, width=0.5, edgecolor='black', linewidth=1.0)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.03, f"{yval:.3f} m", ha='center', va='bottom', fontweight='bold')

    ax.set_title("Exp 5: ATE RMSE Comparison (Overall vs Held-Out Test Set)", fontsize=13, fontweight='bold')
    ax.set_ylabel("ATE RMSE [m]", fontweight='bold')
    ax.grid(True, axis='y', linestyle=':', alpha=0.6)
    fig.tight_layout()
    plt.savefig(os.path.join(plots_dir, '05_test_set_error_breakdown.png'))
    plt.close()

    # Figure 6: Velocity & IMU Bias Convergence
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, dpi=300)
    axes[0].plot(t_sec_df, nn_df['v_x'], label='v_x', linewidth=1.2)
    axes[0].plot(t_sec_df, nn_df['v_y'], label='v_y', linewidth=1.2)
    axes[0].plot(t_sec_df, nn_df['v_z'], label='v_z', linewidth=1.2)
    axes[0].set_ylabel("Velocity [m/s]", fontweight='bold')
    axes[0].set_title("NN-EKF Estimated 3D Velocity Profile", fontsize=11, fontweight='bold')
    axes[0].legend(loc='upper right', frameon=True)

    axes[1].plot(t_sec_df, nn_df['ba_x'], label='b_a,x', linewidth=1.2)
    axes[1].plot(t_sec_df, nn_df['ba_y'], label='b_a,y', linewidth=1.2)
    axes[1].plot(t_sec_df, nn_df['ba_z'], label='b_a,z', linewidth=1.2)
    axes[1].set_ylabel("Accel Bias [m/s²]", fontweight='bold')
    axes[1].set_title("NN-EKF Estimated Accelerometer Bias", fontsize=11, fontweight='bold')
    axes[1].legend(loc='upper right', frameon=True)

    axes[2].plot(t_sec_df, nn_df['bg_x'], label='b_g,x', linewidth=1.2)
    axes[2].plot(t_sec_df, nn_df['bg_y'], label='b_g,y', linewidth=1.2)
    axes[2].plot(t_sec_df, nn_df['bg_z'], label='b_g,z', linewidth=1.2)
    axes[2].set_ylabel("Gyro Bias [rad/s]", fontweight='bold')
    axes[2].set_title("NN-EKF Estimated Gyroscope Bias", fontsize=11, fontweight='bold')
    axes[2].set_xlabel("Time [s]", fontweight='bold')
    axes[2].legend(loc='upper right', frameon=True)

    fig.suptitle("Exp 5: NN-EKF Internal State Estimation Parameters", fontsize=13, fontweight='bold')
    fig.tight_layout()
    plt.savefig(os.path.join(plots_dir, '06_bias_and_velocity_estimation.png'))
    plt.close()

    print(f"Generated all 6 visualization figures in: {plots_dir}")


if __name__ == '__main__':
    main()
