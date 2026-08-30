# Forward Looking Sonar Odometry using Deep Learning

This repository contains the source code for a Master's Engineering Thesis. The project focuses on the implementation of a robust odometry system designed specifically for **Forward Looking Sonar (FLS)** data, utilizing **Deep Neural Networks**.

## Visualization

https://github.com/user-attachments/assets/1df754c3-b942-44a8-90f0-d4547678d91d

Odometry system estimation visualisation: trajectory prediction and key points detection.
🟢 - predicted trajectory 
⚪ - reference trajectory (ground truth)

---

## Key Features

* **Deep Feature Matching:** Utilizes the state-of-the-art **[LoFTR](https://github.com/zju3dv/LoFTR)** (Local Feature Matching with Transformers) architecture for reliable keypoint extraction and matching in noisy acoustic imagery.
* **Advanced Preprocessing & Filtering:** Implements adaptive spatial bucketing, CLAHE histogram equalization, range masking, and median/bilateral filtering to mitigate acoustic noise and enhance feature detection.
* **Robust Motion Estimation:** Combines RANSAC with the Weighted Kabsch algorithm for precise rigid-body transformation estimation, effectively handling outliers.
* **Sensor Fusion:** Mathematical model supports fusion with DVL depth measurement for depth changes compensation. It also includes the possibility to adopt a magnetometer for supporting yaw angle estimation to improve robustness and reliability.
* **Multi-Frame Tracking (Sliding Window):** Employs a sliding-window approach with dynamic keyframe management to maintain consistency, reduce drift, and improve overall trajectory robustness.
* **Versatile Data Support:** Natively supports both Polar and Cartesian FLS data formats with built-in transformations and depth compensation.

---
## ⚙️ System Architecture & Pipeline

The system is built upon two main pillars: a **Front-end** module responsible for acoustic data preprocessing, robust feature matching, and local motion estimation, and a **Back-end** optimization layer that manages keyframes and aggregates trajectory predictions across a sliding time window.

<img width="1572" height="1633" alt="schemat_architektury2" src="https://github.com/user-attachments/assets/79bcaf69-5963-4d4c-994c-50ec4e118771" />

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


