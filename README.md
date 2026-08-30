# 🌊🧭 Forward Looking Sonar Odometry using Deep Learning

This repository contains the source code for a Master's Engineering Thesis. The project focuses on the implementation of a robust odometry system designed specifically for **Forward Looking Sonar (FLS)** data, utilizing **Deep Neural Networks**.

## ✨ Visualisation of results

https://github.com/user-attachments/assets/1df754c3-b942-44a8-90f0-d4547678d91d

Odometry system estimation visualisation: trajectory prediction and key points detection.
🟢 - predicted trajectory 
⚪ - reference trajectory (ground truth)

---

## 🔎 Key Features of LoFTR-based Sonar Odometry Module

* **Deep Feature Matching:** Utilizes the state-of-the-art **[LoFTR](https://github.com/zju3dv/LoFTR)** (Local Feature Matching with Transformers) architecture for reliable keypoint extraction and matching in noisy acoustic imagery.
* **Advanced Preprocessing & Filtering:** Implements adaptive spatial bucketing, CLAHE histogram equalization, range masking, and median/bilateral filtering to mitigate acoustic noise and enhance feature detection.
* **Robust Motion Estimation:** Combines RANSAC with the Weighted Kabsch algorithm for precise rigid-body transformation estimation, effectively handling outliers.
* **Sensor Fusion:** Mathematical model supports fusion with DVL depth measurement for depth changes compensation. It also includes the possibility to adopt a magnetometer for supporting yaw angle estimation to improve robustness and reliability.
* **Multi-Frame Tracking (Sliding Window):** Employs a sliding-window approach with dynamic keyframe management to maintain consistency, reduce drift, and improve overall trajectory robustness.
* **Versatile Data Support:** Natively supports both Polar and Cartesian FLS data formats with built-in transformations and depth compensation.

---

## 🔬 Experimental Research: DPSO (Deep Patch-Based Sonar Odometry)

Beyond the fully functional LoFTR-based pipeline, this repository also contains experimental source code for **DPSO**—an acoustic adaptation of the state-of-the-art **[DPVO (Deep Patch-based Visual Odometry)](https://github.com/princeton-vl/DPVO)** model. 

* **Project Scope:** The repository includes the complete infrastructure for the implementation, training, and evaluation of the DPSO model.
* **Current Status (WIP):** Please note that this specific project track is currently unfinished. The adaptation encountered fundamental mathematical barriers related to iterative **Bundle Adjustment** when applied to the highly ambiguous geometry and elevation uncertainties of Forward Looking Sonar data. 

---
## ⚙️ System Architecture & Pipeline of LoFTR-based Sonar Odometry Module

The system is built upon two main pillars: a **Front-end** module responsible for acoustic data preprocessing, robust feature matching, and local motion estimation, and a **Back-end** optimization layer that manages keyframes and aggregates trajectory predictions across a sliding time window.

<img width="3144" height="3265" alt="LoFTR_architecture_eng" src="https://github.com/user-attachments/assets/5ba7d1e5-cf0f-4e4b-a780-29477f8bab14" />

---

The complete processing pipeline operates through the following sequential stages:

### 🌊 1. Acoustic Preprocessing
Due to the challenging nature of acoustic imaging—particularly high-frequency speckle noise and poor contrast—raw FLS data undergoes a specialized filtering pipeline:
* **Median & Bilateral Filtering:** Effectively eliminates isolated noise peaks and smooths the image while strictly preserving sharp topological boundaries.
* **Adaptive Histogram Equalization (CLAHE):** Locally enhances image contrast, pulling critical seabed textures and structures out of acoustic shadows to provide a stable base for feature extraction.

### 🧠 2. Deep Feature Matching (LoFTR)
Instead of relying on classical keypoint detectors, which struggle with the monotonous textures of the seabed, the system utilizes the **[LoFTR](https://github.com/zju3dv/LoFTR)** model. 
> By leveraging both self-attention and cross-attention mechanisms, LoFTR analyzes the global context of the sonar image, allowing for the reliable extraction of dense point correspondences on both coarse and fine levels. 

### 🎯 3. Advanced Filtering & Depth Compensation
Extracted point pairs undergo rigorous geometric verification before motion estimation:
* **Confidence Thresholding:** Matches with low network confidence or those located at the extreme edges of the measurement range (prone to interpolation distortions) are masked and discarded.
* **Depth Change Compensation:** A critical challenge in sonar odometry is that altitude changes can be misinterpreted as planar translation. By integrating measurements from a depth sensor, the system recalculates the slant range into the true horizontal plane distance, rendering the system invariant to vehicle depth changes.

### 📐 4. Robust Motion Estimation
The verified matches are processed using the **RANSAC** algorithm to isolate geometrically consistent inliers based on a 2D Euclidean transformation model. 
* To avoid the numerical instabilities often associated with iterative non-linear optimization (e.g., Bundle Adjustment) in ambiguous sonar geometries, the final rotation and translation matrices are calculated **analytically** using **Singular Value Decomposition (SVD)**. 
* The estimated movement is then adjusted by the sensor's calibration matrix to reflect the true motion of the robot's center of mass.

### 🔄 5. Local Optimization & Keyframe Management
Because global Loop Closure is computationally expensive and highly prone to catastrophic failures in repetitive underwater environments, a custom local optimization mechanism is implemented:
* **Dynamic Keyframing:** The system dynamically spawns keyframes based on distance traveled or rotation angle, storing them in a sliding time window (empirically optimized to 3 frames).
* **Multi-perspective Aggregation:** Each incoming frame is independently matched against multiple historical keyframes from the buffer. Translation vectors are filtered using a **median** function, while rotation angles are aggregated using a **circular mean**. This approach acts as a robust filter against temporal anomalies and significantly reduces cumulative drift.

---

## Dataset

The system was trained and evaluated using a custom dataset generated within the **[Stonefish] (https://github.com/patrykcieslak/stonefish)** marine robotics simulator. 

> **Note:** The raw dataset contains clean, noiseless sonar data. Realistic acoustic noise (speckle, artifacts, etc.) is injected dynamically during training and evaluating using custom augmentations located in `src/data_loader/transforms.py`.

🔗 **[Download the Dataset Here](https://drive.google.com/drive/folders/1WPsnUuISalV1vTJvZHb2KsWKuzgSGwIQ?usp=sharing)** 

---

## 📁 Repository structure

.
├── config/                                    # YAML configuration files
│   ├── model_dpso.yaml                        # Parameters for DPSO model
│   ├── model_loftr_aracati.yaml               # Parameters for LoFTR-based model (for Aracati2017 dataset)
│   ├── model_loftr_sim.yaml                   # Parameters for LoFTR-based model (own dataset)
│   ├── sonar_aracati.yaml                     # Parameters and calibration for the physical sonar (Aracati dataset)
│   └── sonar_sim.yaml                         # Parameters and calibration for the simulated sonar
├── imgs/                                      # Multimedia assets for documentation
│   └── seq_15_visu.mp4                        # Feature matching visualization for the README file
├── notebooks/                                 # Jupyter notebooks for analysis, experiments, and visualizations
│   ├── evaluation/                            # Evaluation of trained models on test datasets
│   │   ├── evaluation_dpso_test.ipynb         # Tests and metrics for the DPSO model
│   │   ├── odometry_loftr_eval_aracati.ipynb  # Evaluation of LoFTR-based odometry (real data - Aracati2017)
│   │   └── odometry_loftr_eval_sim.ipynb      # Evaluation of LoFTR-based odometry (simulated data - own dataset)
│   ├── test/                                  # Preliminary tests, prototyping, and concept verification
│   │   ├── BA_test.ipynb                      # Testing the Bundle Adjustment optimizer on data
│   │   ├── LoFTR_test._aracati.ipynb          # Prototyping LoFTR matches for real-world data (Aracati2017)
│   │   ├── LoFTR_test_sim.ipynb               # Prototyping LoFTR matches for simulator data
│   │   ├── MatchAnythingVsLoFtr.ipynb         # Comparison of matching quality between MatchAnything and LoFTR  models
│   │   ├── dpso_graph_training_test.ipynb     # Verification of graph-based training (DPSO model)
│   │   ├── dpso_key_points.ipynb              # Keypoint extraction tests (DPSO model)
│   │   ├── odometry2d_test.ipynb              # General mathematical tests for planar 2D motion
│   │   ├── sonar_noise_reduction.ipynb        # Experiments with filters (median, bilateral, CLAHE)
│   │   ├── sonar_noise_simulation.ipynb       # Testing methods for injecting artificial acoustic noise
│   │   └── sonar_preprocessing.ipynb          # Tests of the full FLS image processing and transformation pipeline
│   ├── training/                              # Scripts and logs from the model training process
│   │   ├── output/                            # Saved checkpoints, TensorBoard logs, and training reports
│   │   ├── test_lightning.ipynb               # Running model validation in the PyTorch Lightning environment
│   │   └── train_lightning.ipynb              # Main notebooks for running network training
│   ├── utils/                                 # Helper tools and utility scripts
│   │   ├── create_gif.py                      # Script for generating animations (GIFs) from output frames
│   │   ├── dataset_test.ipynb                 # Verification of dataset integrity and loading correctness
│   │   ├── fix_odometry_sequence.py           # Script to fix and align reference frames between prediction and ground truth (for tests)
│   │   └── reduce_yaw_estim.py                # Script replacing yaw prediction with magnetometer measurements (for tests)
│   └── visu/                                  # Tools for generating plots and visualizing results
│       ├── dataset_sequence_visu.ipynb        # Viewer for sequences of raw frames from the dataset
│       ├── fls_img_visu.ipynb                 # Visualizer for single acoustic scans (single ping)
│       └── trajectory_visu.ipynb              # Plotting estimated trajectories against the ground truth reference
├── obsolete/                                  # Deprecated code, previous iterations, and backups
│   ├── logger.py                              # Deprecated error logging system
│   ├── odometry2d_MFT.py                      # Deprecated Multi-Frame Tracking implementation
│   ├── odometry2d_ma.py                       # Deprecated odometry version based on MatchAnything
│   ├── odometry2d_mft_eval_sim.ipynb          # Deprecated evaluation file for MFT
│   ├── odometry2d_new2.py                     # Scratchpad for old, experimental odometry code
│   └── odometry2d_new_backup.py               # Backup copy of an older odometry algorithm version
├── requirements/                              # System dependency files for installation (pip)
│   ├── ubuntu_gpu.txt                         # Packages for Linux environment with GPU acceleration (CUDA) support
│   └── windows_npgpu.txt                      # Packages for Windows environment without GPU support (CPU only)
├── src/                                       # Core source code of the system
│   ├── data_loader/                           # Scripts responsible for data loading and augmentation
│   │   ├── data_module_lightning.py           # PyTorch Lightning DataModule class organizing data loaders
│   │   ├── dataset.py                         # Dataset class loading image pairs and ground truth
│   │   ├── evaluation_data_generator.py       # Generator creating data pools specifically for testing/evaluation
│   │   ├── lightning_module.py                # Class binding the model, loss functions, and optimizers
│   │   ├── metrics.py                         # Implementation of error metrics with evo package(e.g., ATE, RTE, rotation drift)
│   │   ├── test.py                            # Quick testing script for data loading modules
│   │   ├── transforms.py                      # Augmentations implementation (speckle noise, artifacts)
│   │   └── utils.py                           # Minor helper functions used during data processing
│   └── models/                                # Implementations of neural networks, optimizers, and logic
│       ├── bundle_adjustment_v2.py            # First-order Bundle Adjustment optimizer (Adam algorithm)
│       ├── bundle_adjustment_v3.py            # Second-order Bundle Adjustment (Levenberg-Marquardt)
│       ├── dpso_inference.py                  # Inference version for the DPSO model for long trajectories (in progress)
│       ├── dpso_train.py                      # Training version for the DPSO model 
│       ├── encoders.py                        # Convolutional layer architectures for feature extraction (DPSO)
│       ├── graph_inference.py                 # Pose graph for DPSO (inference version for long trajectory)
│       ├── graph_train.py                     # Pose graph for DPSO (training version, optimized for fast learning)
│       ├── odometry_loftr.py                  # LoFTR-based odometry module implementation
│       ├── patchifier.py                      # Patch extraction and feature map generation (DPSO model)
│       ├── update.py                          # Recurrent layers for DPSO model 
│       └── utils.py                           # Mathematical functions, quaternion operations and other minor helpers
├── .gitignore                                 # 
├── README.md                                  # 
└── download_dataset.py                        # Automated script for downloading the dataset archives

---

## 🚀 How to Run
### 1. Clone the repository
Open your terminal and run the following commands to download the repository and navigate into it:

Bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
(Remember to replace YOUR_USERNAME/YOUR_REPO_NAME with your actual GitHub repository URL).

## 2. Install dependencies
It is highly recommended to use a virtual environment. Install the required packages based on your operating system and hardware:

Bash
pip install -r requirements/ubuntu_gpu.txt
For Windows without GPU (CPU only):

## 3. Running the Code (Jupyter Notebooks)
The core training and evaluation processes are handled via Jupyter Notebooks. 
To execute a specific task, launch Jupyter and open the corresponding notebook:

DPSO Model Training: Run notebooks/training/train_lightning.ipynb

DPSO Model Evaluation: Run notebooks/evaluation/evaluation_dpso_test.ipynb

LoFTR-based Odometry Evaluation (Simulated Data): Run notebooks/evaluation/odometry_loftr_eval_sim.ipynb

LoFTR-based Odometry Evaluation (Real Aracati Data): Run notebooks/evaluation/odometry_loftr_eval_aracati.ipynb

⚠️ Jupyter Notebooks are prepared to work either on Google Colab or on local computer. 
⚠️ It is necessery to specifie paths to dataset folders etc. 
