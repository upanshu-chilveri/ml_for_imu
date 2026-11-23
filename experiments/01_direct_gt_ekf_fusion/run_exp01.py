"""
Experiment 1: Direct Local Ground Truth EKF Fusion (Theoretical Upper Bound Benchmark).

Fuses 200 Hz IMU linear acceleration and angular velocity measurements
with 16.5 Hz high-precision local laser tracker position ground truth updates.
Generates all 4 comprehensive evaluation plots:
  1. 3D Trajectory Comparison
  2. Per-Axis Position Tracking Over Time (X, Y, Z)
  3. Absolute Euclidean Error Over Time
  4. Internal State Estimation (Velocity & IMU Biases Convergence)
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
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ])


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


def main():
    exp_dir = os.path.dirname(__file__)
    data_dir = os.path.abspath(os.path.join(exp_dir, '..', '..', 'MH_01_easy.bag', 'extracted_ros1_data'))

    imu_df = pd.read_csv(os.path.join(data_dir, 'imu.csv'))
    gt_df = pd.read_csv(os.path.join(data_dir, 'ground_truth.csv'))

    t_start = max(imu_df['timestamp_ns'].min(), gt_df['timestamp_ns'].min())
    t_end = min(imu_df['timestamp_ns'].max(), gt_df['timestamp_ns'].max())

    imu = imu_df[(imu_df['timestamp_ns'] >= t_start) & (imu_df['timestamp_ns'] <= t_end)].reset_index(drop=True)
    gt = gt_df[(gt_df['timestamp_ns'] >= t_start) & (gt_df['timestamp_ns'] <= t_end)].reset_index(drop=True)

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
    gt_events = [(row.timestamp_ns, 'gt', np.array([row.p_x, row.p_y, row.p_z])) for row in gt.itertuples()]

    all_events = sorted(imu_events + gt_events, key=lambda x: x[0])
    prev_t = all_events[0][0]

    history = []
    for ev in all_events:
        t_ns = ev[0]
        dt = (t_ns - prev_t) / 1e9
        prev_t = t_ns

        if ev[1] == 'imu':
            ekf.predict(ev[2], ev[3], dt)
            history.append({
                'timestamp_ns': t_ns,
                't_sec': (t_ns - t_start) / 1e9,
                'p_x': ekf.p[0], 'p_y': ekf.p[1], 'p_z': ekf.p[2],
                'v_x': ekf.v[0], 'v_y': ekf.v[1], 'v_z': ekf.v[2],
                'ba_x': ekf.ba[0], 'ba_y': ekf.ba[1], 'ba_z': ekf.ba[2],
                'bg_x': ekf.bg[0], 'bg_y': ekf.bg[1], 'bg_z': ekf.bg[2]
            })
        elif ev[1] == 'gt':
            ekf.update_position(ev[2])

    res_df = pd.DataFrame(history)

    gt_x = np.interp(res_df['timestamp_ns'], gt['timestamp_ns'], gt['p_x'])
    gt_y = np.interp(res_df['timestamp_ns'], gt['timestamp_ns'], gt['p_y'])
    gt_z = np.interp(res_df['timestamp_ns'], gt['timestamp_ns'], gt['p_z'])
    gt_interp = np.column_stack([gt_x, gt_y, gt_z])
    est = res_df[['p_x', 'p_y', 'p_z']].values

    err = est - gt_interp
    euc_err = np.sqrt(np.sum(err**2, axis=1))
    ate_rmse = float(np.sqrt(np.mean(euc_err**2)))
    rmse_xyz = np.sqrt(np.mean(err**2, axis=0))

    metrics = {
        'experiment': '01_direct_gt_ekf_fusion',
        'ate_rmse_m': ate_rmse,
        'mean_error_m': float(np.mean(euc_err)),
        'median_error_m': float(np.median(euc_err)),
        'max_error_m': float(np.max(euc_err)),
        'rmse_x_m': float(rmse_xyz[0]),
        'rmse_y_m': float(rmse_xyz[1]),
        'rmse_z_m': float(rmse_xyz[2])
    }

    print("=========================================================")
    print("  EXPERIMENT 1: DIRECT GT EKF FUSION RESULTS (UPPER BOUND)")
    print("=========================================================")
    print(f"ATE RMSE: {ate_rmse:.4f} m ({ate_rmse*100:.2f} cm)")
    print(f"Per-axis RMSE: X={rmse_xyz[0]:.4f}m, Y={rmse_xyz[1]:.4f}m, Z={rmse_xyz[2]:.4f}m")

    with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    res_df.to_csv(os.path.join(exp_dir, 'fused_trajectory.csv'), index=False)

    plots_dir = os.path.join(exp_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    t_sec = res_df['t_sec']
    gt_t_sec = (gt['timestamp_ns'] - t_start) / 1e9

    # Plot 1: 3D Trajectory Comparison
    fig = plt.figure(figsize=(10, 7.5), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(gt['p_x'], gt['p_y'], gt['p_z'], 'k-', label='Ground Truth Reference', linewidth=2.0)
    ax.plot(res_df['p_x'], res_df['p_y'], res_df['p_z'], 'r--', label=f'EKF Direct GT Updates (ATE: {ate_rmse*100:.2f} cm)', linewidth=1.5)
    ax.set_title("Exp 1: 3D Trajectory Comparison (Direct GT Fusion)", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("X Position [m]", fontweight='bold')
    ax.set_ylabel("Y Position [m]", fontweight='bold')
    ax.set_zlabel("Z Position [m]", fontweight='bold')
    ax.legend(loc='upper right', frameon=True)
    fig.tight_layout()
    plt.savefig(os.path.join(plots_dir, '01_trajectory_3d_exp1.png'))
    plt.close()

    # Plot 2: Per-Axis Position Components Over Time
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, dpi=300)
    components = [('X', 'p_x', rmse_xyz[0]), ('Y', 'p_y', rmse_xyz[1]), ('Z', 'p_z', rmse_xyz[2])]
    for i, (axis_name, col, rmse_val) in enumerate(components):
        axes[i].plot(gt_t_sec, gt[col], 'k-', label='Ground Truth', linewidth=1.8, alpha=0.8)
        axes[i].plot(t_sec, res_df[col], 'r--', label=f'EKF Estimate (RMSE: {rmse_val*100:.2f} cm)', linewidth=1.2)
        axes[i].set_ylabel(f"{axis_name} [m]", fontweight='bold')
        axes[i].legend(loc='upper right', frameon=True)
        axes[i].grid(True, linestyle=':', alpha=0.6)
    axes[2].set_xlabel("Time [s]", fontweight='bold')
    fig.suptitle("Exp 1: Per-Axis Position Tracking Over Time", fontsize=14, fontweight='bold')
    fig.tight_layout()
    plt.savefig(os.path.join(plots_dir, '02_position_components_exp1.png'))
    plt.close()

    # Plot 3: Absolute Euclidean Error Over Time
    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=300)
    ax.plot(t_sec, euc_err * 100, color='#1f77b4', linewidth=1.2, label='Euclidean Error [cm]')
    ax.axhline(ate_rmse * 100, color='red', linestyle='--', linewidth=1.5, label=f'RMSE ({ate_rmse*100:.2f} cm)')
    ax.axhline(np.mean(euc_err) * 100, color='orange', linestyle=':', linewidth=1.5, label=f'Mean ({np.mean(euc_err)*100:.2f} cm)')
    ax.set_title("Exp 1: Absolute Euclidean Position Error Over Time", fontsize=13, fontweight='bold')
    ax.set_xlabel("Time [s]", fontweight='bold')
    ax.set_ylabel("Error [cm]", fontweight='bold')
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper right', frameon=True)
    ax.grid(True, linestyle=':', alpha=0.6)
    fig.tight_layout()
    plt.savefig(os.path.join(plots_dir, '03_position_error_exp1.png'))
    plt.close()

    # Plot 4: Internal State Estimation (Velocity & IMU Biases)
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, dpi=300)
    axes[0].plot(t_sec, res_df['v_x'], label='v_x', linewidth=1.2)
    axes[0].plot(t_sec, res_df['v_y'], label='v_y', linewidth=1.2)
    axes[0].plot(t_sec, res_df['v_z'], label='v_z', linewidth=1.2)
    axes[0].set_ylabel("Velocity [m/s]", fontweight='bold')
    axes[0].set_title("Estimated Velocity Profile", fontsize=11, fontweight='bold')
    axes[0].legend(loc='upper right', frameon=True)

    axes[1].plot(t_sec, res_df['ba_x'], label='b_a,x', linewidth=1.2)
    axes[1].plot(t_sec, res_df['ba_y'], label='b_a,y', linewidth=1.2)
    axes[1].plot(t_sec, res_df['ba_z'], label='b_a,z', linewidth=1.2)
    axes[1].set_ylabel("Accel Bias [m/s²]", fontweight='bold')
    axes[1].set_title("Estimated Accelerometer Bias Convergence", fontsize=11, fontweight='bold')
    axes[1].legend(loc='upper right', frameon=True)

    axes[2].plot(t_sec, res_df['bg_x'], label='b_g,x', linewidth=1.2)
    axes[2].plot(t_sec, res_df['bg_y'], label='b_g,y', linewidth=1.2)
    axes[2].plot(t_sec, res_df['bg_z'], label='b_g,z', linewidth=1.2)
    axes[2].set_ylabel("Gyro Bias [rad/s]", fontweight='bold')
    axes[2].set_title("Estimated Gyroscope Bias Convergence", fontsize=11, fontweight='bold')
    axes[2].set_xlabel("Time [s]", fontweight='bold')
    axes[2].legend(loc='upper right', frameon=True)

    fig.suptitle("Exp 1: EKF Internal State Estimation (Velocity & Biases)", fontsize=13, fontweight='bold')
    fig.tight_layout()
    plt.savefig(os.path.join(plots_dir, '04_bias_and_velocity_estimation_exp1.png'))
    plt.close()

    print(f"Generated 4 complete visualization figures in: {plots_dir}")


if __name__ == '__main__':
    main()
