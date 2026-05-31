# Sonar Odometry 

___
## Project overview 

**This is repositorium contains source code for master engineering thesis.** 

This project focus on implementation of odometry system based on **Forward Looking Sonar**, using **Deep Neural Networks**.

Architecture core is inspired by [**DPVO** (*Deep Patch-based Visual Odometry*)](https://github.com/princeton-vl/DPVO), in order to use proven and efficient visual odometry techniques to process sonar data in underwater environments. 

___
## Repository content and structure

```text

│
├── config/     # configuration files
│   ├── model.yaml       # parameters of models
│   ├── sonar.yaml       # parameters of sonar
│
├── notebooks/   # tests, visualisations, etc.
│   └── test/     
│         ├── BA_test.ipynb                 # Test and visualisation of bundle adjustment module
│         ├── graph_training_test.ipynb     # Visualisation of reprojection and trajectory prediction
│         ├── key_points.ipynb              # Visualisation of patches selection 
│         └─ sonar_noise.ipynb              # Analysis of sonar noise
|                     
├── src/   
│   ├── data_loader/   
│   |       ├── data_module_lightning.py 
│   |       ├── dataset.py 
│   |       ├── evaluation_data_generator.py    # dataloader for evaluation (long sequence)
│   |       ├── lightning_module.py             # pytorch lightning module
│   |       ├── metrics.py                      # metrics for model evaluation
│   |       ├── transforms.py                   # augumentation, noise for sonar data
│   |       └── utils.py 
│   │
|   └── models/  
│           ├── bundle_adjustment_v1.py         # obsolete
│           ├── bundle_adjustment_v2.py         # bundle adjustment, based on 1st order optimizer (Adam)
│           ├── bundle_adjustment_v3.py         # bundle adjustment, based on 2nd order optimizer (LM)
│           ├── dpso_inference.py               # model - inference version (for long sequence)
│           ├── dpso_train.py                   # model - train version (for short sequence only)
│           ├── encodrs.py                      # encoder for feature extraction
│           ├── graph_inference.py              # pose graph - inference version (for long sequence)
│           ├── graph_train.py                  # pose graph - train version (for short sequence only)
│           ├── logger.py                       # logger for saving data to csv 
│           ├── patchifier.py                   # keypoints selection, patch extraction
│           ├── update.py                       # update operator (graph network)
│           ├── utils.py                        
|
├── training/   
│       ├── checkpoints/   
│       ├── lightning_logs/  
│       ├── output/  
│       ├── test_lightning/                     # model evaluation script - obsolete!
│       ├── train_lightning/                    # training script
|
├── .gitignore                    
├── requirements/    
│       ├── ubuntu.txt 
│       └── windows_nogpu.txt ia projektu
└── README.md                     
```

___
## Dataset 

Custom dataset was created using [Stonefish](https://github.com/patrykcieslak/stonefish/) marine robotics simulator. 

> Note: Dataset contains sonar data without noise. It was adding during training, using code located in: `./src/data_loader/transforms.py`.

Dataset is available [HERE](https://drive.google.com/drive/folders/1BgosYlaRkQkSa43Jpgb6hoGK88n3bXLL?usp=sharing).

___
## How to run 








