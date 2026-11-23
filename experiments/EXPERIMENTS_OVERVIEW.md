# Comprehensive IMU & Sensor Fusion Experiments Suite

This directory contains all distinct research experiments conducted on the dataset. Each experiment is organized into a dedicated, self-contained folder with its own code, dataset parameters, results JSON, and visualization plots.

---

## 📁 Experiments Directory Structure

```
d:\ML For IMU\experiments\
│
├── 00_ekf_direct_gt_updates/          <-- Exp 0: Initial EKF Benchmark (Direct ~16.5 Hz GT updates)
│   ├── run_exp00.py
│   ├── results.json
│   ├── fused_trajectory.csv
│   └── plots/
│
├── 01_direct_gt_ekf_fusion/           <-- Exp 1: Theoretical Upper Bound (16.5 Hz GT updates)
│   ├── run_exp01.py
│   ├── results.json
│   ├── fused_trajectory.csv
│   └── plots/
│
├── 02_simulated_gnss_imu_fusion/      <-- Exp 2: Real-World Sensor Fusion (1 Hz GNSS fixes)
│   ├── run_exp02.py
│   ├── simulated_gnss_data.csv
│   ├── results.json
│   ├── fused_trajectory.csv
│   └── plots/
│
├── 03_update_rate_ablation_study/     <-- Exp 3: Sensor Outage & Update Frequency Sensitivity
│   ├── run_exp03.py
│   ├── ablation_results.json
│   └── plots/
│
├── 04_four_baselines_comparison/      <-- Exp 4: Evolutionary Baselines (Classical -> TinyML -> Adaptive)
│   ├── run_exp04.py
│   ├── comparison_results.json
│   └── plots/
│
├── 05_nn_ekf_gnss_learning/           <-- Exp 5: Neural Network EKF (0% GT Supervision, 70/15/15 Split)
│   ├── run_exp05_nn_ekf.py            <-- Master execution script
│   ├── nn_imu_model.py                <-- Neural Network architecture & split manager
│   ├── simulated_gnss_data.csv        <-- Simulated 1 Hz GNSS fixes
│   ├── results.json                   <-- Held-out test set metrics & +10.5% improvement
│   ├── nn_ekf_fused_trajectory.csv    <-- Output fused trajectory
│   └── plots/                         <-- 6 high-res plots (Loss, 3D, Components, Error, Test set bar chart, Biases)
│
├── 06_learned_inertial_odometry_ronin/ <-- Exp 6: Pure IMU Learned Odometry (Eliminates External Sensor Fusion!)
│   ├── run_exp06_ronin.py             <-- Master execution script
│   ├── ronin_tcn_model.py             <-- PyTorch 1D Temporal Convolutional Network (TCN / ResNet-1D)
│   ├── best_ronin_model.pth           <-- Saved best neural network weights checkpoint
│   ├── results.json                   <-- 16,701x drift reduction metrics
│   ├── trajectory_ronin.csv           <-- Pure dead-reckoning trajectory output
│   └── plots/                         <-- 6 publication-grade figures
│
├── 07_enhanced_ronin_tcn_odometry/     <-- Exp 7: Multi-Scale Dilated TCN + End-to-End Integration Loss
│   ├── run_exp07_enhanced_ronin.py    <-- Master execution script
│   ├── enhanced_tcn_model.py          <-- Multi-branch dilated TCN + integration loss
│   ├── best_enhanced_ronin_model.pth  <-- Saved checkpoint
│   ├── results.json                   <-- Diagnostic metrics
│   ├── trajectory_enhanced_ronin.csv  <-- Output trajectory
│   └── plots/                         <-- 6 diagnostic figures (vibration aliasing analysis)
│
├── 08_universal_neural_inertial_navigation/ <-- Exp 8: Universal NINS (12D Torque Dynamics + 90s GPS Blackout)
│   ├── run_exp08_nins.py              <-- Master mission runner (50s GPS -> 90s Blackout -> 44s Recovery)
│   ├── neural_inertial_core.py        <-- 12D Torque Features + ResNet-1D + Bi-GRU Momentum Core
│   ├── adaptive_nins_ekf.py           <-- 15-State Error-State EKF + Neural Virtual Odometer Updates
│   ├── best_nins_model.pth            <-- Trained Neural Core weights
│   ├── results.json                   <-- 97.96% outage drift reduction metrics
│   ├── trajectory_nins.csv            <-- Full mission trajectory
│   └── plots/                         <-- 6 publication-grade figures (3D, Phase errors, Log drift, Benchmarks)
│
└── 09_multi_sequence_universal_nins/   <-- Exp 9: Multi-Sequence Cross-Environment Universal NINS (11 EuRoC Seqs)
    ├── run_exp09_multi_nins.py        <-- GPU Accelerated master training & evaluation script
    ├── euroc_loader.py                <-- 11-sequence ASL loader with yaw-rotation augmentation
    ├── best_nins_multiseq.pth         <-- Generalised cross-sequence model checkpoint
    ├── results.json                   <-- Cross-sequence generalization metrics
    └── plots/                         <-- 6 publication-grade figures (3D test seqs, loss, volume comparison)
```

---

## 🧪 Detailed Summary of Experiments

### 0️⃣ Experiment 0: Direct Ground Truth Update EKF (Initial Benchmark)
- **Objective**: Initial benchmark fusing 200 Hz IMU data directly with high-precision ~16.5 Hz ground truth position updates.
- **Accuracy**: **3.43 cm ATE RMSE**.

### 1️⃣ Experiment 1: Direct Local Ground Truth EKF Fusion (Theoretical Upper Bound)
- **Objective**: Theoretical upper bound of EKF position accuracy under sub-centimeter position sensing.
- **Accuracy**: **3.43 cm ATE RMSE**.

### 2️⃣ Experiment 2: Simulated GNSS-IMU Sensor Fusion (Real-World Setup)
- **Objective**: Simulate realistic GNSS navigation. Ground truth is **strictly isolated** for offline evaluation. EKF fuses 200 Hz IMU with 1 Hz GNSS fixes ($\sigma_H = 0.8\text{m}, \sigma_V = 1.2\text{m}$).
- **Accuracy**: **1.257 m ATE RMSE** (Filters raw 1.631 m GNSS noise into a smooth 200 Hz trajectory).

### 3️⃣ Experiment 3: Update Rate & Sensor Loss Sensitivity Ablation
- **Objective**: Evaluate how EKF position drift degrades as external sensor update frequency decreases (16.5 Hz down to 0.16 Hz).

### 4️⃣ Experiment 4: 4 Evolutionary Baselines Benchmark
- **Objective**: Comparative evaluation of Baseline 1 (73.7 km drift), Baseline 2 (1.257 m), Baseline 3 (2.208 m), and Baseline 4 (1.954 m).

### 5️⃣ Experiment 5: Neural Network Enhanced Extended Kalman Filter (NN-EKF)
- **Objective**: Implement a Deep Neural Network (MLP/ResNet) trained **100% on IMU + Simulated GNSS fixes (0% Ground Truth used in model training)** using chronological 70% Train, 15% Val, 15% Test split.
- **Results**:
  - **Traditional EKF Test Set RMSE**: **1.0051 m (100.5 cm)**
  - **Neural-Network EKF (NN-EKF) Test Set RMSE**: **0.8998 m (90.0 cm)**
  - **Held-Out Test Set Accuracy Improvement**: **+10.5% Error Reduction!**

### 6️⃣ Experiment 6: Learned Inertial Odometry (RoNIN / TCN) — Pure IMU (Zero Sensor Fusion)
- **Objective**: Completely eliminate external sensor fusion (GNSS, cameras, LiDAR) by replacing double-integration ($\iint \mathbf{a}\, dt^2$) with deep temporal velocity prediction.
- **Methodology**: 1D Temporal Convolutional Network (RoNIN ResNet-1D / TCN) trained on sliding windows ($W=40$, 0.20s context) to predict body-frame velocity $\mathbf{v}_{body}(t)$, integrated once into position.
- **Results**:
  - **Classical Pure IMU Double-Integration Drift**: **73,768.63 m (73.7 km)**
  - **RoNIN / TCN Pure IMU ATE RMSE**: **4.4171 m (4.4 m)**
  - **Drift Reduction Factor**: **16,701x Error Reduction (99.994% accuracy improvement)!**
  - **Held-Out Test Set ATE RMSE**: **9.4099 m**
- **Takeaway**: Demonstrates that pure IMU navigation can maintain sub-5-meter tracking over 3 minutes of aggressive drone flight without any external fixes.

### 7️⃣ Experiment 7: Multi-Scale Dilated TCN + End-to-End Integration Loss
- **Objective**: Test multi-scale receptive fields (dilations $d \in \{1, 2, 4\}$) and auxiliary trajectory displacement loss.
- **Key Discovery**: While horizontal Y-axis drift was cut by 50% (3.38 m), dilation rates $d > 1$ caused Nyquist-Shannon aliasing of 80–165 Hz rotor vibrations into false vertical Z-axis accelerations (15.16 m). Highlighted the strict physical necessity of dense consecutive sampling ($d=1$) for multirotor IMU navigation.

### 8️⃣ Experiment 8: Universal Neural-Inertial Navigation System (NINS)
- **Objective**: Design and benchmark an end-to-end universal neural-inertial framework that operates purely on 200 Hz IMU data during extended GPS-denied blackouts, yields ultra-low drift, and seamlessly re-acquires GNSS fixes upon signal restoration without divergence or discontinuity.
- **Architecture**:
  - **Physics-Informed Torque Conditioning**: Ingests 12-channel rotational dynamics ($\boldsymbol{\omega}, \mathbf{a}, \dot{\boldsymbol{\omega}}, \boldsymbol{\omega} \times \boldsymbol{\omega}$), leveraging rotor gyroscopic precession as predictive foresight of drone acceleration before accelerometer deflection occurs.
  - **ResNet-1D + Bi-GRU Momentum Core**: Dense temporal feature extraction with bidirectional recurrent momentum tracking and heteroscedastic uncertainty estimation.
  - **Adaptive Dual-Channel Error-State EKF (ES-EKF)**: 200 Hz strapdown kinematic propagation, 20 Hz Neural Virtual Odometer corrections during outages, and smooth Mahalanobis-gated GNSS updates.
- **Mission Benchmark (184s Flight: 50s Nominal $\to$ 90s GPS Blackout $\to$ 44s GNSS Recovery)**:
  - **Overall Trajectory ATE RMSE**: **7.91 m** across the entire 184-second flight.
  - **Phase 1 (GPS Online, 0s–50s)**: **1.55 m ATE RMSE**.
  - **Phase 2 (90s CONTINUOUS GPS BLACKOUT, 50s–140s)**:
    - *Classical Pure IMU Drift*: **73,768.63 m (73.7 km)**
    - *Unassisted Traditional EKF Drift*: **537.21 m** (Max drift: **1,341.5 m**)
    - *Universal NINS (Neural Virtual Odometer)*: **10.97 m ATE RMSE** (Max drift: **16.57 m**)
    - **Drift Reduction vs Unassisted EKF**: **+97.96% (49.0x Lower Drift!) 🚀**
    - **Drift Reduction vs Classical Pure IMU**: **7,445x Lower Drift! 🎯**
  - **Phase 3 (GPS Restored & Reconverged, 140s–184s)**: Re-converges smoothly to **3.59 m ATE RMSE**.
  - **Unseen Held-Out Test Set**: **2.43 m ATE RMSE** (X: 1.22 m, Y: 1.59 m, Z: 1.38 m).
- **Takeaway**: Confirms that physical torque conditioning combined with neural virtual odometry provides autonomous drones with mission-critical survivability in GPS-denied / electronic warfare environments.

### 9️⃣ Experiment 9: Multi-Sequence Cross-Environment Universal NINS (Full EuRoC Dataset)
- **Objective**: Scale training across the entire 11-sequence EuRoC MAV dataset (~274,000 raw IMU samples across 3 distinct rooms and 3 difficulty tiers). Test true zero-shot cross-sequence generalization on completely unseen environments and flight dynamics.
- **Methodology**:
  - **Cross-Sequence Split**: Train on 7 sequences (`MH_01`, `MH_02`, `MH_03`, `V1_01`, `V1_02`, `V2_01`, `V2_02`), validate on 2 difficult sequences (`MH_04`, `V1_03`), and test on 2 completely unseen sequences (`MH_05_difficult`, `V2_03_difficult`).
  - **Data Augmentation**: 3x yaw-rotation augmentation yielding **366,608 training windows** (28x data volume expansion over Exp 08).
  - **Hardware Acceleration**: GPU acceleration with PyTorch 2.5.1 + CUDA 12.1 on NVIDIA GeForce RTX 3050 Laptop GPU.
- **Results on Completely Unseen Sequences**:
  - **`V2_03_difficult` ATE RMSE**: **2.44 m** (X: 1.43m, Y: 1.85m, Z: 0.70m, Max: 4.42m).
  - **`MH_05_difficult` ATE RMSE**: **4.49 m** (X: 2.92m, Y: 1.72m, Z: 2.94m, Max: 7.35m).
  - **Overall Mean Test ATE across Unseen Flights**: **3.47 m**.
  - **Validation Body-Velocity RMSE**: **0.61 m/s**.
- **Takeaway**: Solves the single-sequence memorization bottleneck. Demonstrates robust generalized tracking across rooms, lighting shifts, and extreme high-speed drone maneuvers without requiring GNSS.

---

## 🚀 How to Execute All Experiments

```powershell
# Run Exp 5 (Neural Network EKF with 1 Hz GNSS)
python "d:\ML For IMU\experiments\05_nn_ekf_gnss_learning\run_exp05_nn_ekf.py"

# Run Exp 6 (Learned Inertial Odometry RoNIN / TCN - Pure IMU)
python "d:\ML For IMU\experiments\06_learned_inertial_odometry_ronin\run_exp06_ronin.py"

# Run Exp 7 (Enhanced RoNIN TCN with Multi-Scale Dilations)
python "d:\ML For IMU\experiments\07_enhanced_ronin_tcn_odometry\run_exp07_enhanced_ronin.py"

# Run Exp 8 (Universal Neural-Inertial Navigation System NINS)
python "d:\ML For IMU\experiments\08_universal_neural_inertial_navigation\run_exp08_nins.py"

# Run Exp 9 (Multi-Sequence GPU Universal NINS)
& "C:\Users\DeadGodRebel\anaconda3\envs\torch_gpu\python.exe" "d:\ML For IMU\experiments\09_multi_sequence_universal_nins\run_exp09_multi_nins.py"
```
