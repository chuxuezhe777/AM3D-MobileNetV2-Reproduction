# Transfer Learning Based Defect Detection in 3D Printed Objects

## 1 Introduction

Visual inspection supports quality control in additive manufacturing by identifying defective printed parts. Training deep neural networks from scratch requires substantial labelled data, whereas transfer learning reuses features learned from larger image collections. Ahsan et al. [1] investigated this approach for classifying photographs of 3D printed objects as defective or normal.

This group project reproduces the workflow published by Ahsan et al. [1], [4]. Group members implement different pretrained convolutional neural networks using shared data and evaluation protocols. The project examines how transfer learning and optimizer choice affect image-level defect classification across the three experimental datasets.

Python supports the implementation through Keras for pretrained networks and training, OpenCV for image processing, NumPy for numerical operations, and scikit-learn for splitting and evaluation.

## 2 Project Description

### 2.1 Objective and Approach

The objective is to reproduce transfer learning results for binary defect classification. Supervised learning maps labelled photographs to defective or normal classes. Pretrained networks extract visual features, and task-specific classifiers learn from the training subset. Performance is assessed against the corresponding published experiment.

### 2.2 Shared Datasets

Study One uses original cylinder photographs from AMdataset. Study Two uses the authors' augmented cylinder collection, in which normal images outnumber defects. Study Three uses a separate collection of 3D printing images from Kaggle [3], [6]. Each study uses a separate training, validation and test split.

Table 1  Shared datasets and experimental splits

| Study | Defective | Normal | Training | Validation | Test |
| --- | --- | --- | --- | --- | --- |
| One | 136 | 135 | 189 | 27 | 55 |
| Two | 947 | 2,070 | 2,111 | 308 | 598 |
| Three | 759 | 798 | 1,089 | 159 | 309 |

Within each study, the sorted image paths were split with seed 42. The first split held out 30% of images, and the second assigned 66% of the holdout to the test set [4], [7].

### 2.3 Model

#### 2.3.1 MobileNetV2

The reference freezes ImageNet MobileNetV2 [2], scales RGB to [0, 1] and uses 4 × 4 average pooling. All runs share a 64-unit ReLU layer, 0.5 dropout and binary softmax.

The fusion model scales inputs to [−1, 1] [5] and combines each channel's 7 × 7 mean (75%) with its four strongest responses averaged (25%). The resulting 1,280 features preserve the head size.

We compare reference, global average pooling and fusion with Adam, SGD and RMSprop. Each uses seeds 42–44, learning rate 0.001 and binary cross-entropy. Epochs/batch sizes are 30/8, 50/50 and 30/30. The global average control uses corrected input scaling.

#### 2.3.2 Other Models

## 3 Results and Discussion

### 3.1 MobileNetV2

Each cell shows fusion (top, three-seed mean) and paper (bottom) [4], scaled by 100. Precision, recall, F1 and AP use normal as positive. Defect recall is the authors' specificity.

Table 2  Fusion MobileNetV2 and author test scores

| Study and optimizer | Acc. | Prec. | Recall | F1 | Defect recall | MCC | Kappa | AUC | AP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| One  Adam | 100.00 / 100.00 | 100.00 / 100.00 | 100.00 / 100.00 | 100.00 / 100.00 | 100.00 / 100.00 | 100.00 / 100.00 | 100.00 / 100.00 | 100.00 / 100.00 | 100.00 / 100.00 |
| One  SGD | 100.00 / 98.18 | 100.00 / 96.55 | 100.00 / 100.00 | 100.00 / 98.25 | 100.00 / 96.30 | 100.00 / 96.42 | 100.00 / 96.36 | 100.00 / 100.00 | 100.00 / 100.00 |
| One  RMSprop | 100.00 / 100.00 | 100.00 / 100.00 | 100.00 / 100.00 | 100.00 / 100.00 | 100.00 / 100.00 | 100.00 / 100.00 | 100.00 / 100.00 | 100.00 / 100.00 | 100.00 / 100.00 |
| Two  Adam | 99.78 / 99.33 | 99.67 / 99.02 | 100.00 / 100.00 | 99.83 / 99.51 | 99.33 / 97.95 | 99.50 / 98.48 | 99.50 / 98.47 | 100.00 / 99.74 | 100.00 / 99.75 |
| Two  SGD | 99.28 / 99.33 | 98.93 / 99.02 | 100.00 / 100.00 | 99.46 / 99.51 | 97.81 / 97.95 | 98.37 / 98.48 | 98.35 / 98.47 | 100.00 / 99.90 | 100.00 / 99.95 |
| Two  RMSprop | 99.94 / 99.83 | 99.92 / 99.75 | 100.00 / 100.00 | 99.96 / 99.88 | 99.83 / 99.49 | 99.87 / 99.62 | 99.87 / 99.62 | 100.00 / 100.00 | 100.00 / 100.00 |
| Three  Adam | 100.00 / 99.35 | 100.00 / 98.73 | 100.00 / 100.00 | 100.00 / 99.36 | 100.00 / 98.69 | 100.00 / 98.71 | 100.00 / 98.71 | 100.00 / 99.98 | 100.00 / 99.98 |
| Three  SGD | 98.49 / 95.15 | 97.51 / 95.48 | 99.57 / 94.87 | 98.52 / 95.18 | 97.39 / 95.42 | 97.01 / 90.29 | 96.98 / 90.29 | 99.95 / 99.44 | 99.95 / 99.45 |
| Three  RMSprop | 100.00 / 99.35 | 100.00 / 98.73 | 100.00 / 100.00 | 100.00 / 99.36 | 100.00 / 98.69 | 100.00 / 98.71 | 100.00 / 98.71 | 100.00 / 99.97 | 100.00 / 99.98 |

Accuracy rose in 6 settings, tied in 2 and fell in 1. Changes ranged from -0.06 pp (Two, SGD) to +3.34 pp (Three, SGD). Fusion beat matched global averaging in 2 of nine settings. Study Two's results use image-level splits.

### 3.2 Other Models

## References

[1] M. M. Ahsan, S. Raman, Y. Liu, and Z. Siddique, “Defect analysis of 3D printed object using transfer learning approaches,” Expert Systems with Applications, vol. 253, Art. no. 124293, 2024, doi: 10.1016/j.eswa.2024.124293.

[2] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, “MobileNetV2: Inverted residuals and linear bottlenecks,” in Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR), 2018, pp. 4510–4520, doi: 10.1109/CVPR.2018.00474.

[3] M. M. Ahsan, “AMdataset,” GitHub, commit 52fe75d. Accessed: Nov. 20, 2026. [Online]. Available: https://github.com/mahsan2/AMdataset

[4] M. M. Ahsan, “AM3d,” GitHub, commit dd1025d. Accessed: Nov. 20, 2026. [Online]. Available: https://github.com/mahsan2/AM3d

[5] Keras, “MobileNet models,” Keras documentation. Accessed: Nov. 20, 2026. [Online]. Available: https://keras.io/api/applications/mobilenet/mobilenet_models/

[6] justin900429, “3D-Printer Defected Dataset,” Kaggle, version 1. [Online]. Available: https://www.kaggle.com/datasets/justin900429/3d-printer-defected-dataset

[7] RachelYim67, “IDAT7215-3D-Defect-Detection,” GitHub. [Online]. Available: https://github.com/RachelYim67/IDAT7215-3D-Defect-Detection

## Run the three studies

Keep the three Python files together. Each is an entry point for the named study.
Study Two and Study Three import the common training and evaluation functions from Study One.
Run the commands with Python 3.12.

```sh
python -m pip install keras==3.15.1 torch==2.8.0 ml-dtypes==0.5.3 numpy==2.1.3 scikit-learn==1.6.1 opencv-python-headless==4.11.0.86
python mobilenetv2_study_one.py train
python mobilenetv2_study_two.py train
python mobilenetv2_study_three.py train
```

Studies One and Two share the pinned AMdataset ZIP. Study Three downloads Kaggle version 1.
All runs verify the shared image inventory. Use `--archive` to reuse a downloaded ZIP,
`--data` for files under the two class directories, or `--weights` for the Keras no-top
MobileNetV2 ImageNet weights. The default run directory is `runs/study_one`,
`runs/study_two` or `runs/study_three`. Use a fresh `--out` directory for each repeat.

Each run writes `paper_comparison.csv`, `metrics.csv`, `predictions.csv`, `split.csv` and
nine fusion models. The comparison CSV includes unrounded metrics, seed standard deviations
and both matched controls. In the README table, each pair reads **fusion / paper**.

```sh
python mobilenetv2_study_one.py predict --model runs/study_one/global_topk_adam_seed42.keras --image path/to/print.jpg
```

The editable report is `MobileNetV2_Report.docx`.
