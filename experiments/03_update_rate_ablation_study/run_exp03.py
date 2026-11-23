"""
Experiment 3: Update Rate & Sensor Loss Sensitivity Ablation Study.

Evaluates EKF performance across different exteroceptive update rates:
16.5 Hz, 3.3 Hz, 1.65 Hz, 0.82 Hz, 0.33 Hz, 0.16 Hz.
Generates all 4 comprehensive evaluation plots:
  1. 3D Trajectories across update rates
  2. Per-Axis Position Components Over Time
  3. Absolute Euclidean Error Over Time across update rates
  4. Update Rate Sensitivity Curve (ATE RMSE vs Fix Frequency)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


def skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


class ErrorStateEKF:
    def __init__(self, p0, v0, R0, sigma_a=0.08, sigma_g=0.005, sigma_gt=0.01, g_val=9.81):
        self.p = np.array(p0, dtype=np.float64)
        self.v = np.array(v0, dtype=np.float64)
        self.rot = R.from_matrix(R0)
        self.ba = np.zeros(3, dtype=np.float64)
        self.bg = np.zeros(3, dtype=np.float64)
        self.g_vec = np.array([0.0, 0.0, -g_val])

        self.sigma_a = sigma_a
        self.sigma_g = sigma_g
        self.sigma_gt = sigma_gt

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

    def update_position(self, p_gt):
        H = np.zeros((3, 15), dtype=np.float64)
        H[0:3, 0:3] = np.eye(3)

        R_m = (self.sigma_gt ** 2) * np.eye(3)
        z = p_gt - self.p
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


def run_single_ablation(imu, gt, step_size):
    gt_sub = gt.iloc[::step_size].reset_index(drop=True)

    a_mean = imu[['a_x', 'a_y', 'a_z']].iloc[:100].mean().values
    u_acc = a_mean / np.linalg.norm(a_mean)
    u_z = np.array([0, 0, 1.0])
    v_rot = np.cross(u_acc, u_z)
    s_rot = np.linalg.norm(v_rot)
    c_rot = np.dot(u_acc, u_z)
    vx = skew(v_rot)
    R_init = np.eye(3) + vx + (vx @ vx) * ((1 - c_rot) / (s_rot**2))

    p0 = gt[['p_x', 'p_y', 'p_z']].iloc[0].values
    ekf = ErrorStateEKF(p0=p0, v0=np.zeros(3), R0=R_init)

    imu_events = [(row.timestamp_ns, 'imu', np.array([row.w_x, row.w_y, row.w_z]), np.array([row.a_x, row.a_y, row.a_z])) for row in imu.itertuples()]
    gt_events = [(row.timestamp_ns, 'gt', np.array([row.p_x, row.p_y, row.p_z])) for row in gt_sub.itertuples()]

    all_events = sorted(imu_events + gt_events, key=lambda x: x[0])
    prev_t = all_events[0][0]

    history = []
    for ev in all_events:
        t_ns = ev[0]
        dt = (t_ns - prev_t) / 1e9
        prev_t = t_ns

        if ev[1] == 'imu':
            ekf.predict(ev[2], ev[3], dt)
            history.append((t_ns, ekf.p.copy()))
        elif ev[1] == 'gt':
            ekf.update_position(ev[2])

    est_ts = np.array([x[0] for x in history])
    est_p = np.array([x[1] for x in history])

    gt_x = np.interp(est_ts, gt['timestamp_ns'], gt['p_x'])
    gt_y = np.interp(est_ts, gt['timestamp_ns'], gt['p_y'])
    gt_z = np.interp(est_ts, gt['timestamp_ns'], gt['p_z'])
    gt_interp = np.column_stack([gt_x, gt_y, gt_z])

    err = est_p - gt_interp
    euc_err = np.sqrt(np.sum(err**2, axis=1))
    ate_rmse = float(np.sqrt(np.mean(euc_err**2)))
    return est_ts, est_p, euc_err, ate_rmse


def main():
    exp_dir = os.path.dirname(__file__)
    data_dir = os.path.abspath(os.path.join(exp_dir, '..', '..', 'MH_01_easy.bag', 'extracted_ros1_data'))

    imu_df = pd.read_csv(os.path.join(data_dir, 'imu.csv'))
    gt_df = pd.read_csv(os.path.join(data_dir, 'ground_truth.csv'))

    t_start = max(imu_df['timestamp_ns'].min(), gt_df['timestamp_ns'].min())
    t_end = min(imu_df['timestamp_ns'].max(), gt_df['timestamp_ns'].max())

    imu = imu_df[(imu_df['timestamp_ns'] >= t_start) & (imu_df['timestamp_ns'] <= t_end)].reset_index(drop=True)
    gt = gt_df[(gt_df['timestamp_ns'] >= t_start) & (gt_df['timestamp_ns'] <= t_end)].reset_index(drop=True)

    print("=========================================================")
    print("  EXPERIMENT 3: UPDATE RATE ABLATION STUDY               ")
    print("=========================================================")

    ablation_results = {}
    trajectories = {}
    errors = {}

    test_steps = [(1, 16.5), (5, 3.3), (10, 1.65), (20, 0.82), (50, 0.33), (100, 0.16)]

    for step, rate in test_steps:
        ts, est_p, euc_err, ate = run_single_ablation(imu, gt, step)
        key_str = f"{rate:.2f}_Hz"
        ablation_results[key_str] = {
            'update_rate_hz': rate,
            'subsample_step': step,
            'ate_rmse_m': ate
        }
        trajectories[key_str] = (ts, est_p)
        errors[key_str] = (ts, euc_err)
        print(f"Update Rate: {rate:5.2f} Hz (Step {step:3d}) -> ATE RMSE: {ate:.4f} m ({ate*100:.1f} cm)")

    with open(os.path.join(exp_dir, 'ablation_results.json'), 'w') as f:
        json.dump(ablation_results, f, indent=2)

    plots_dir = os.path.join(exp_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    t_sec_gt = (gt['timestamp_ns'] - t_start) / 1e9

    # Plot 1: 3D Trajectory Comparison Across Update Rates
    fig = plt.figure(figsize=(10, 7.5), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(gt['p_x'], gt['p_y'], gt['p_z'], 'k-', label='Ground Truth Reference', linewidth=2.0)
    colors = ['#d62728', '#2ca02c', '#1f77b4', '#ff7f0e', '#9467bd', '#8c564b']
    for idx, (step, rate) in enumerate([(1, 16.5), (5, 3.3), (10, 1.65), (20, 0.82)]):
        key = f"{rate:.2f}_Hz"
        ts, est_p = trajectories[key]
        ate_cm = ablation_results[key]['ate_rmse_m'] * 100
        ax.plot(est_p[:, 0], est_p[:, 1], est_p[:, 2], '--', color=colors[idx], label=f'{rate:.2f} Hz Update ({ate_cm:.1f} cm)', linewidth=1.3)

    ax.set_title("Exp 3: 3D Trajectories Across External Update Rates", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("X Position [m]", fontweight='bold')
    ax.set_ylabel("Y Position [m]", fontweight='bold')
    ax.set_zlabel("Z Position [m]", fontweight='bold')
    ax.legend(loc='upper right', frameon=True)
    fig.tight_layout()
    plt.savefig(os.path.join(plots_dir, '01_trajectory_3d_exp3.png'))
    plt.close()

    # Plot 2: Per-Axis Tracking Over Time (16.5 Hz vs 1.65 Hz vs 0.82 Hz)
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, dpi=300)
    for i, col in enumerate(['p_x', 'p_y', 'p_z']):
        axes[i].plot(t_sec_gt, gt[col], 'k-', label='Ground Truth Ref', linewidth=1.8, alpha=0.8)
        for idx, rate in enumerate([16.5, 1.65, 0.82]):
            key = f"{rate:.2f}_Hz"
            ts, est_p = trajectories[key]
            t_s = (ts - t_start) / 1e9
            axes[i].plot(t_s, est_p[:, i], '--', color=colors[idx], label=f'{rate:.2f} Hz Rate', linewidth=1.2)
        axes[i].set_ylabel(f"{col[-1].upper()} [m]", fontweight='bold')
        axes[i].legend(loc='upper right', frameon=True)
        axes[i].grid(True, linestyle=':', alpha=0.6)
    axes[2].set_xlabel("Time [s]", fontweight='bold')
    fig.suptitle("Exp 3: Per-Axis Position Tracking Across Update Rates", fontsize=14, fontweight='bold')
    fig.tight_layout()
    plt.savefig(os.path.join(plots_dir, '02_position_components_exp3.png'))
    plt.close()

    # Plot 3: Euclidean Position Error Over Time Across Rates
    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=300)
    for idx, rate in enumerate([16.5, 3.3, 1.65, 0.82]):
        key = f"{rate:.2f}_Hz"
        ts, euc_err = errors[key]
        t_s = (ts - t_start) / 1e9
        ax.plot(t_s, euc_err * 100, color=colors[idx], linewidth=1.2, label=f'{rate:.2f} Hz Rate (ATE: {ablation_results[key]["ate_rmse_m"]*100:.1f} cm)')
    ax.set_title("Exp 3: Euclidean Error Progression Across Update Frequencies", fontsize=13, fontweight='bold')
    ax.set_xlabel("Time [s]", fontweight='bold')
    ax.set_ylabel("Error [cm]", fontweight='bold')
    ax.set_ylim(bottom=0, top=150)
    ax.legend(loc='upper right', frameon=True)
    ax.grid(True, linestyle=':', alpha=0.6)
    fig.tight_layout()
    plt.savefig(os.path.join(plots_dir, '03_position_error_exp3.png'))
    plt.close()

    # Plot 4: Update Rate Ablation Sensitivity Curve
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    rates_all = [v['update_rate_hz'] for v in ablation_results.values()]
    ates_all = [v['ate_rmse_m'] * 100 for v in ablation_results.values()]

    ax.plot(rates_all, ates_all, 'o-', color='#d62728', linewidth=2.0, markersize=8)
    for r, a in zip(rates_all, ates_all):
        ax.annotate(f"{a:.1f} cm", (r, a), textcoords="offset points", xytext=(0, 8), ha='center', fontweight='bold')

    ax.set_xscale('log')
    ax.set_title("Exp 3: Update Frequency Sensitivity Curve", fontsize=13, fontweight='bold')
    ax.set_xlabel("Update Frequency [Hz] (Log Scale)", fontweight='bold')
    ax.set_ylabel("ATE RMSE [cm]", fontweight='bold')
    ax.grid(True, which="both", linestyle=':', alpha=0.6)
    fig.tight_layout()
    plt.savefig(os.path.join(plots_dir, '04_update_rate_sensitivity_exp3.png'))
    plt.close()

    print(f"Generated 4 complete visualization figures in: {plots_dir}")


if __name__ == '__main__':
    main()
