# Forward Looking Sonar Odometry using Deep Learning

This repository contains the source code for a Master's Engineering Thesis. The project focuses on the implementation of a robust odometry system designed specifically for **Forward Looking Sonar (FLS)** data, utilizing **Deep Neural Networks**.

## Visualization

<div align="center">
  <video src="imgs/seq_15_visu.mp4" controls autoplay loop muted playsinline width="100%">
    Your browser does not support the video tag.
  </video>
  <p><i>Odometry pipeline in action: feature matching and trajectory estimation.</i></p>
</div>

---

## Key Features

* **Deep Feature Matching:** Utilizes the state-of-the-art **[LoFTR](https://github.com/zju3dv/LoFTR)** (Local Feature Matching with Transformers) architecture for reliable keypoint extraction and matching in noisy acoustic imagery.
* **Advanced Preprocessing & Filtering:** Implements adaptive spatial bucketing, CLAHE histogram equalization, range masking, and median/bilateral filtering to mitigate acoustic noise and enhance feature detection.
* **Robust Motion Estimation:** Combines RANSAC with the Weighted Kabsch algorithm for precise rigid-body transformation estimation, effectively handling outliers.
* **Sensor Fusion:** Mathematical model supports fusion with DVL depth measurement for depth changes compensation. It also includes the possibility to adopt a magnetometer for supporting yaw angle estimation to improve robustness and reliability.
* **Multi-Frame Tracking (Sliding Window):** Employs a sliding-window approach with dynamic keyframe management to maintain consistency, reduce drift, and improve overall trajectory robustness.
* **Versatile Data Support:** Natively supports both Polar and Cartesian FLS data formats with built-in transformations and depth compensation.

---

## System Architecture & Pipeline

1.  **Preprocessing:** Raw sonar frames are filtered to reduce noise and enhance contrast. If the input is in polar coordinates, it is mathematically transformed into a Cartesian grid.
2.  **Feature Matching:** The LoFTR model processes consecutive frames (or keyframes) to find dense point correspondences.
3.  **Outlier Rejection:** Spatial bucketing ensures uniform distribution of matches, while range masking and confidence thresholding filter out weak points.
4.  **Motion Estimation:** RANSAC isolates inliers, and a weighted Kabsch algorithm calculates the translation and rotation between frames.
5.  **Keyframe Management:** The system intelligently decides when to spawn a new keyframe based on distance, rotation, or skipped frame timeouts to maintain tracking stability.

---

## Dataset

The system was trained and evaluated using a custom dataset generated within the **Stonefish** marine robotics simulator. 

> **Note:** The raw dataset contains clean, noiseless sonar data. Realistic acoustic noise (speckle, ambient, etc.) is injected dynamically during training using custom augmentations located in `src/data_loader/transforms.py`.

🔗 **[Download the Dataset Here](#)** *(Add link)*

---

## Installation & Setup

**1. Clone the repository:**
```bash
git clone <YOUR_REPO_URL>
cd <REPO_NAME>
