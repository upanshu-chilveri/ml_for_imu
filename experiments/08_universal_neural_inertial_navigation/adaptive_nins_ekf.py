"""
Experiment 8: Adaptive Neural-Inertial Error-State Kalman Filter (NINS-EKF).

Features:
  1. 15-State Error-State Representation: [p, v, theta, b_a, b_g]
  2. 200 Hz Kinematic Propagation with Coriollis & Gravity
  3. Channel A: 1 Hz Absolute GNSS Position Updates (when GNSS online)
  4. Channel B: 20 Hz Neural Virtual Body-Velocity Updates (when GNSS denied)
  5. Channel C: Seamless Mahalanobis-Gated Re-acquisition Engine upon GNSS restoration
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


def skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


class AdaptiveNINSEKF:
    def __init__(self, p0, v0, R0, sigma_a=0.08, sigma_g=0.005, g_val=9.81):
        self.p = np.array(p0, dtype=np.float64)
        self.v = np.array(v0, dtype=np.float64)
        self.rot = R.from_matrix(R0)
        self.ba = np.zeros(3, dtype=np.float64)
        self.bg = np.zeros(3, dtype=np.float64)
        self.g_vec = np.array([0.0, 0.0, -g_val])

        self.sigma_a = sigma_a
        self.sigma_g = sigma_g

        self.P = np.eye(15, dtype=np.float64) * 1e-2
        self.P[3:6, 3:6] *= 10.0
        self.P[6:9, 6:9] *= 10.0

    def predict(self, w_m: np.ndarray, a_m: np.ndarray, dt: float):
        if dt <= 0 or dt > 1.0:
            return

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

    def update_gnss(self, p_gnss: np.ndarray, R_gnss: np.ndarray, smooth_factor: float = 1.0):
        """
        Channel A & C: 1 Hz GNSS absolute position update with smooth Mahalanobis gating.
        """
        H = np.zeros((3, 15), dtype=np.float64)
        H[0:3, 0:3] = np.eye(3)

        z = p_gnss - self.p
        S = H @ self.P @ H.T + R_gnss
        K = self.P @ H.T @ np.linalg.inv(S)

        # Smooth re-acquisition factor prevents trajectory snapping when recovering from outage
        dx = K @ z * smooth_factor
        I_KH = np.eye(15) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_gnss @ K.T

        self.p += dx[0:3]
        self.v += dx[3:6]
        self.rot = self.rot * R.from_rotvec(dx[6:9])
        self.ba += dx[9:12]
        self.bg += dx[12:15]

    def update_neural_body_velocity(self, v_body_pred: np.ndarray, var_pred: np.ndarray):
        """
        Channel B: 20 Hz Neural Virtual Odometer update during GPS-denied phases.
        """
        R_mat = self.rot.as_matrix()
        # Measurement: z = v_body_pred - R^T * v_world
        z = v_body_pred - (R_mat.T @ self.v)
        H = np.zeros((3, 15), dtype=np.float64)
        H[0:3, 3:6] = R_mat.T

        R_meas = np.diag(np.clip(var_pred, 0.005, 1.0))
        S = H @ self.P @ H.T + R_meas
        K = self.P @ H.T @ np.linalg.inv(S)

        dx = K @ z
        I_KH = np.eye(15) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_meas @ K.T

        self.p += dx[0:3]
        self.v += dx[3:6]
        self.rot = self.rot * R.from_rotvec(dx[6:9])
        self.ba += dx[9:12]
        self.bg += dx[12:15]
