#\Forward Looking Sonar Odometry using Deep Learning

___
## Project Overview

This repository contains the source code for Master's Engineering Thesis. 

The project focuses on the implementation of a robust odometry system designed specifically for **Forward Looking Sonar (FLS)** data, utilizing **Deep Neural Networks**. 

![Results_GiF](C:/Users/janis/Projekty/Magisterka/SonarOdometry/imgs/seq_15_visu.gif)

___
### Key Features:
- **Deep Feature Matching:** Utilizes the state-of-the-art **[LoFTR](https://github.com/zju3dv/LoFTR)** (Local Feature Matching with Transformers) architecture for reliable keypoint extraction and matching in noisy acoustic imagery.

- **Advanced Preprocessing & Filtering:** Implements adaptive spatial bucketing, CLAHE histogram equalization, range masking, and median/bilateral filtering to mitigate acoustic noise and enhance feature detection.

- **Robust Motion Estimation:** Combines RANSAC with Weighted Kabsch algorithm for precise rigid-body transformation estimation, handling outliers effectively.

- **Sensor fusion:** Mathematical model supports fusion with DVL depth measurement for depth changes compensation.  Possibility to adopt magnetometr for supporting yaw angle estimation for improve robustness and relabilty. 

- **Multi-Frame Tracking (Sliding Window):** Employs a sliding-window approach with dynamic keyframe management to maintain consistency, reduce drift, and improve overall trajectory robustness.

- **Versatile Data Support:** Natively supports both Polar and Cartesian FLS data formats with built-in transformations and depth compensation.

___
## System Architecture & Pipeline

*(This section will be expanded later, but here is a brief overview of the process based on `odometry_loftr.py`)*

1. **Preprocessing:** Raw sonar frames are filtered to reduce noise and enhance contrast. If the input is in polar coordinates, it is mathematically transformed into a Cartesian grid.
2. **Feature Matching:** The LoFTR model processes consecutive frames (or keyframes) to find dense point correspondences.
3. **Outlier Rejection:** Spatial bucketing ensures uniform distribution of matches, while range masking and confidence thresholding filter out weak points.
4. **Motion Estimation:** RANSAC isolates inliers, and a weighted Kabsch algorithm calculates the translation and rotation between frames.
5. **Keyframe Management:** The system intelligently decides when to spawn a new keyframe based on distance, rotation, or skipped frame timeouts to maintain tracking stability.

___
## Repository Structure

```text
│
├── config/                     # Configuration files (model params, sonar specs)
│   ├── model.yaml              
│   ├── sonar.yaml              
│
├── notebooks/                  # Jupyter notebooks for testing, visualization, and analysis
│   └── test/     
│         ├── BA_test.ipynb     
│         ├── graph_training_test.ipynb 
│         ├── key_points.ipynb   
│         └── sonar_noise.ipynb 
│                     
├── src/                        # Core source code
│   ├── data_loader/            # Dataset management, augmentations, and PyTorch Lightning modules
│   │       ├── data_module_lightning.py 
│   │       ├── dataset.py 
│   │       ├── transforms.py   # Sonar noise injection and augmentations
│   │       └── ...
│   │
│   └── models/                 # Neural network models and odometry logic
│           ├── odometry_loftr.py               # Core odometry pipeline (LoFTR based)
│           ├── bundle_adjustment_v2.py         # 1st order optimizer (Adam)
│           ├── bundle_adjustment_v3.py         # 2nd order optimizer (LM)
│           ├── patchifier.py                   # Keypoint selection / patch extraction
│           └── ...
│
├── training/                   # Scripts for training and evaluation
│       ├── checkpoints/   
│       ├── output/  
│       └── train_lightning.py                  # Main training script
│
├── requirements/               # Installation requirements
│       ├── requirements_ubuntu_gpu.txt         
│       └── requirements_windows_cpu.txt        
└── README.md

## Dataset
The system was trained and evaluated using a custom dataset generated within the Stonefish marine robotics simulator.

Note: The raw dataset contains clean, noiseless sonar data. Realistic acoustic noise (speckle, ambient, etc.) is injected dynamically during training using custom augmentations located in ./src/data_loader/transforms.py.

🔗 Download the Dataset Here

## Installation & Setup
Clone the repository:

Bash ```
git clone <YOUR_REPO_URL>
cd <REPO_NAME>
Create a virtual environment (recommended):
```

Bash```
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
Install dependencies based on your system:
```

For Ubuntu (GPU):

Bash
pip install -r requirements/requirements_ubuntu_gpu.txt
For Windows (CPU-only):

Bash
pip install -r requirements/requirements_windows_cpu.txt
How to Run
(Instructions on how to run the inference, training, and evaluations will be added here).