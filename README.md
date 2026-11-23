# Transfer Learning Based Defect Detection in 3D Printed Objects

## 1 Introduction

Visual inspection supports quality control in additive manufacturing by identifying defective printed parts. Training deep neural networks from scratch requires substantial labelled data, whereas transfer learning reuses features learned from larger image collections. Ahsan et al. [1] investigated this approach for classifying photographs of 3D printed objects as defective or normal.

This group project reproduces the published workflow using the authors' AMdataset and AM3d repositories [3], [4]. Group members implement different pretrained convolutional neural networks on a shared dataset. The project examines whether the reported classification performance can be reproduced and how model and optimizer choices affect the results. The task is image classification, with one predicted label per photograph.

Python supports the implementation through Keras for pretrained networks and training, OpenCV for image processing, NumPy for numerical operations, and scikit-learn for splitting and evaluation.

## 2 Project Description

### 2.1 Objective and Approach

The objective is to reproduce transfer learning results for binary defect classification. Supervised learning maps labelled photographs to defective or normal classes. Pretrained networks extract visual features, and task-specific classifiers learn from the training subset. Performance is assessed against the corresponding published experiment.

### 2.2 Shared Dataset

All models use Dataset One from AMdataset [3], comprising cylinder photographs labelled as defective or normal. The working collection contains 271 images, including 136 defective and 135 normal examples. Table 1 documents the split used in the completed MobileNetV2 experiment.

Table 1  Dataset composition and experimental split

| Subset | Defective | Normal | Total |
| --- | --- | --- | --- |
| Training | 92 | 97 | 189 |
| Validation | 16 | 11 | 27 |
| Test | 28 | 27 | 55 |
| Total | 136 | 135 | 271 |

Following the reference notebook [4], sorted image paths are split by holding out 30%, then allocating 66% of that subset to testing. Both operations use random_state = 42 without stratification. Validation data monitor training, while test data evaluate the final model.

### 2.3 Model

#### 2.3.1 MobileNetV2

MobileNetV2 uses inverted residuals and linear bottlenecks for efficient feature extraction [2]. Following Study One [1], [4], the Keras ImageNet backbone [5] is frozen and its original classifier removed. RGB images are resized to 224 × 224 and scaled to [0, 1], without online augmentation. The head applies 4 × 4 average pooling, flattening, a 64-unit ReLU layer, dropout of 0.5 and two-unit softmax classification.

Adam, SGD and RMSprop each train a freshly initialized head for 50 epochs, using batch size 30, learning rate 0.001 and binary cross-entropy. Seed 42 fixes initialization and splitting. The final epoch is evaluated using accuracy, precision, recall, F1 and ROC-AUC, with defects treated as positive.

The backbone and training design come from Keras and AM3d [4], [5]. Newly written reproduction scripts handle checksum verification, split export, cached feature extraction, optimizer runs, metric export and prediction.

Algorithm 1  Reproduction pseudocode

1  Load labelled images and split the dataset.

2  Resize, scale and extract frozen backbone features.

3  For each optimizer, initialize and train the classifier.

4  Evaluate test predictions and export metrics.

#### 2.3.2 Other Models

## 3 Results and Discussion

### 3.1 MobileNetV2

Table 2 compares our test accuracy with the corresponding Study One results [1], [4]. Adam and RMSprop classify all 55 test images correctly; SGD misses one of 28 defects. All three accuracies match the published values to two decimal places.

Table 2  MobileNetV2 reproduction versus Study One

| Optimizer | Our accuracy % | Paper accuracy % | Difference pp |
| --- | --- | --- | --- |
| Adam | 100.00 | 100.00 | 0.00 |
| SGD | 98.18 | 98.18 | 0.00 |
| RMSprop | 100.00 | 100.00 | 0.00 |

Adam and RMSprop achieve the highest test accuracy under the fixed training settings. These results support effective transfer learning on this dataset, although one split and 55 test images provide limited evidence of generalization to independently printed objects. Differences in Table 2 are percentage points (pp).

Optimizer choice is the tested parameter variation. Further learning-rate or dropout tuning should use validation performance, reserving test data for final evaluation; such tuning has not yet been tested.

### 3.2 Other Models

## References

[1] M. M. Ahsan, S. Raman, Y. Liu, and Z. Siddique, “Defect analysis of 3D printed object using transfer learning approaches,” Expert Systems with Applications, vol. 253, Art. no. 124293, 2024, doi: 10.1016/j.eswa.2024.124293.

[2] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, “MobileNetV2: Inverted residuals and linear bottlenecks,” in Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR), 2018, pp. 4510–4520, doi: 10.1109/CVPR.2018.00474.

[3] M. M. Ahsan, “AMdataset,” GitHub, commit 52fe75d. Accessed: Nov. 20, 2026. [Online]. Available: https://github.com/mahsan2/AMdataset

[4] M. M. Ahsan, “AM3d,” GitHub, commit dd1025d. Accessed: Nov. 20, 2026. [Online]. Available: https://github.com/mahsan2/AM3d

[5] Keras, “MobileNet models,” Keras documentation. Accessed: Nov. 20, 2026. [Online]. Available: https://keras.io/api/applications/mobilenet/mobilenet_models/
