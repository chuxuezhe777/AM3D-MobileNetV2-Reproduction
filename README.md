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

The Study One baseline [1], [4] freezes an ImageNet MobileNetV2 backbone [2], [5]. RGB inputs are resized to 224 × 224 and divided by 255. The head uses 4 × 4 average pooling, flattening, 64 ReLU units, dropout of 0.5 and a two-class softmax.

Each optimizer trains for 50 epochs with batch size 30, learning rate 0.001 and binary cross-entropy. We use seed 42 and evaluate the final epoch.

Project extension. We adapted the EfficientNetB0 team's robustness protocol [6] to MobileNetV2 on the shared Study One split. Both arms use the same architecture, initialization and image order for each of seeds 42, 43 and 44. Each arm uses Adam for 30 epochs and 210 updates. The clean arm uses original images. The augmented arm samples one of five views per image per epoch: original, blurred, JPEG compressed, downsampled or rotated. Brightness changes are held out of training.

Algorithm 1  Paired robustness training

1  Extract frozen features for each training view.

2  For each seed, train both heads with the same image order.

3  Select the arm using validation F1 and recall.

4  Evaluate both arms on seven test conditions.

#### 2.3.2 Other Models

## 3 Results and Discussion

### 3.1 MobileNetV2

The original reproduction matches the corresponding Study One accuracies [1], [4]. Adam and RMSprop classify all 55 images correctly. SGD misses one defect.

Table 2  MobileNetV2 reproduction versus Study One

| Optimizer | Our accuracy % | Paper accuracy % | Difference pp |
| --- | --- | --- | --- |
| Adam | 100.00 | 100.00 | 0.00 |
| SGD | 98.18 | 98.18 | 0.00 |
| RMSprop | 100.00 | 100.00 | 0.00 |

The validation rule selected augmentation after mean F1 rose from 0.908 to 0.995, with clean F1 maintained and worst-condition recall improved. Selection preceded test scoring.

Table 3  Mean test accuracy across three seeds (%)

| Test condition | Clean training | Augmented | Change (pp) |
| --- | --- | --- | --- |
| Clean | 100.00 | 100.00 | 0.00 |
| Dark (×0.5) | 97.58 | 98.18 | +0.61 |
| Bright (×1.5) | 100.00 | 98.18 | -1.82 |
| Blur (9 × 9, σ = 2) | 83.03 | 100.00 | +16.97 |
| JPEG quality 15 | 90.30 | 100.00 | +9.70 |
| Downsample to 56 × 56 | 93.33 | 100.00 | +6.67 |
| Rotate 10° | 100.00 | 100.00 | 0.00 |
| Seven-condition mean | 94.89 | 99.48 | +4.59 |

Augmentation raises mean test accuracy by 4.59 percentage points. Blur shows the largest gain, clean accuracy stays at 100%, and bright-image accuracy falls. The seven conditions reuse the same 55 images. These synthetic tests do not establish performance on new print runs.

### 3.2 Other Models

## References

[1] M. M. Ahsan, S. Raman, Y. Liu, and Z. Siddique, “Defect analysis of 3D printed object using transfer learning approaches,” Expert Systems with Applications, vol. 253, Art. no. 124293, 2024, doi: 10.1016/j.eswa.2024.124293.

[2] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, “MobileNetV2: Inverted residuals and linear bottlenecks,” in Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR), 2018, pp. 4510–4520, doi: 10.1109/CVPR.2018.00474.

[3] M. M. Ahsan, “AMdataset,” GitHub, commit 52fe75d. Accessed: Nov. 20, 2026. [Online]. Available: https://github.com/mahsan2/AMdataset

[4] M. M. Ahsan, “AM3d,” GitHub, commit dd1025d. Accessed: Nov. 20, 2026. [Online]. Available: https://github.com/mahsan2/AM3d

[5] Keras, “MobileNet models,” Keras documentation. Accessed: Nov. 20, 2026. [Online]. Available: https://keras.io/api/applications/mobilenet/mobilenet_models/

[6] AHJ8273528, “IDAT7215-EfficientNetB0,” GitHub, commit c48cdc9. Accessed: Nov. 23, 2026. [Online]. Available: https://github.com/AHJ8273528/IDAT7215-EfficientNetB0
