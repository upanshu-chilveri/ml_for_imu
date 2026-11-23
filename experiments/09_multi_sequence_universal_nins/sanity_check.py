import sys
sys.path.insert(0, ".")
from euroc_loader import load_euroc_sequence, build_windows, SEQUENCE_PATHS
import numpy as np

seq = load_euroc_sequence(SEQUENCE_PATHS["MH_01_easy"])
print("MH_01_easy loaded OK")
print("  IMU samples  :", len(seq["imu_ts"]))
print("  pos_gt shape :", seq["pos_gt"].shape)
print("  vel_gt shape :", seq["vel_gt"].shape)
print("  rot shape    :", seq["rot_matrices"].shape)
print("  12d feat     :", seq["features_12d"].shape)

X, y = build_windows(seq, window_size=40, step=2, augment_yaw=True, n_augments=3)
print("  Windows X    :", X.shape)
print("  Windows y    :", y.shape)
print("  y range      : [{:.3f}, {:.3f}]".format(float(y.min()), float(y.max())))
print("Sanity check PASSED")
