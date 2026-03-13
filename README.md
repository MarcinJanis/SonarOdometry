# Sonar Odometry 

___
## Project overview 

**This is repositorium contains source code for master engineering thesis.** 

This project focus on implementation of odometry system based on **Forward Looking Sonar**, using **Deep Neural Networks**.

Architecture core is inspired by [**DPVO** (*Deep Patch-based Visual Odometry*)](https://github.com/princeton-vl/DPVO), adapted to work with forward looking sonar data. 

___
## Results and Evaluation

Model was evaluated on both simulated data used in training process and datasets with real forward looking sonar images to examine model accuracy and it's ability to exceed sim-to-real gap.

Results on each dataset ws presented below. All results with metrics are compared in table on the end of this section. 


### 1. [Simulated Dataset](https://drive.google.com/drive/folders/1BgosYlaRkQkSa43Jpgb6hoGK88n3bXLL?usp=sharing)


### 2. [Aracati2017](https://github.com/matheusbg8/aracati2017)

### 3. [Caves-dataset](https://cirs.udg.edu/caves-dataset/)

### 4. [Aurora-dataset](https://ieee-dataport.org/open-access/aurora-multi-sensor-dataset-robotic-ocean-exploration)

### 5. [Seaward-dataset](https://seaward.science/data/pos/)


___
## Repository content and structure



___
## Dataset 

Custom dataset was created using [Stonefish](https://github.com/patrykcieslak/stonefish/) marine robotics simulator. 

> Note: Dataset contains sonar data without noise. It was adding during training, using code located in: `./.......`.


Dataset is available [HERE](https://drive.google.com/drive/folders/1BgosYlaRkQkSa43Jpgb6hoGK88n3bXLL?usp=sharing).

___
## How to run 








