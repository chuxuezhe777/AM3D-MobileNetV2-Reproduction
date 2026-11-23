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

The authors' reference setup uses batch size 30, learning rate 0.001 and binary cross-entropy. The robustness experiment retains those settings and trains each arm for 30 epochs.

Project extension. We adapted the EfficientNetB0 team's robustness protocol [6] to MobileNetV2 on the shared Study One split. For Adam, SGD and RMSprop, the two arms share initialization, image order and 210 updates per seed. The clean arm uses original images. The augmented arm samples one view per image per epoch from originals, blur, JPEG compression, downsampling and rotation. Brightness changes are held out of training.

Algorithm 1  Paired robustness training

1  Extract frozen features for each training view.

2  For each seed, train both heads with the same image order.

3  Record the validation decision using F1 and defect recall.

4  Compare augmented clean-image accuracy with Study One for each optimizer.

#### 2.3.2 Other Models

## 3 Results and Discussion

### 3.1 MobileNetV2

Table 2 compares the augmented MobileNetV2 with the authors' Study One result for each optimizer on clean test images [1], [4]. Our accuracy is averaged over seeds 42, 43 and 44 on the same fixed split. The paper column reports the corresponding published accuracy.

Table 2  Augmented MobileNetV2 versus the original Study One results

| Optimizer | Augmented accuracy (%) | Paper accuracy (%) | Change (pp) |
| --- | --- | --- | --- |
| Adam | 100.00 | 100.00 | 0.00 |
| SGD | 98.79 | 98.18 | +0.61 |
| RMSprop | 99.39 | 100.00 | -0.61 |

Relative to the paper, the nonzero accuracy differences are SGD (+0.61 pp), RMSprop (-0.61 pp). The experiment changes the training views and uses paired repeats, so the table compares accuracy under the same optimizer and clean image condition. The paper provides no matching corruption scores. The fixed test split does not establish performance on independent print runs.

### 3.2 Other Models

## References

[1] M. M. Ahsan, S. Raman, Y. Liu, and Z. Siddique, “Defect analysis of 3D printed object using transfer learning approaches,” Expert Systems with Applications, vol. 253, Art. no. 124293, 2024, doi: 10.1016/j.eswa.2024.124293.

[2] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, “MobileNetV2: Inverted residuals and linear bottlenecks,” in Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR), 2018, pp. 4510–4520, doi: 10.1109/CVPR.2018.00474.

[3] M. M. Ahsan, “AMdataset,” GitHub, commit 52fe75d. Accessed: Nov. 20, 2026. [Online]. Available: https://github.com/mahsan2/AMdataset

[4] M. M. Ahsan, “AM3d,” GitHub, commit dd1025d. Accessed: Nov. 20, 2026. [Online]. Available: https://github.com/mahsan2/AM3d

[5] Keras, “MobileNet models,” Keras documentation. Accessed: Nov. 20, 2026. [Online]. Available: https://keras.io/api/applications/mobilenet/mobilenet_models/

[6] AHJ8273528, “IDAT7215-EfficientNetB0,” GitHub, commit c48cdc9. Accessed: Nov. 23, 2026. [Online]. Available: https://github.com/AHJ8273528/IDAT7215-EfficientNetB0

## Running the experiment

Use Python 3.12. Run these commands from the project directory:

```sh
python -m pip install keras==3.15.1 torch==2.8.0 ml-dtypes==0.5.3 numpy==2.1.3 scikit-learn==1.6.1 opencv-python-headless==4.11.0.86
python mobilenetv2.py prepare
python mobilenetv2.py train
python mobilenetv2.py predict --model runs/augmented_adam_seed42.keras --image AMdataset/dataset/badcylinder/IMG_9191.JPG
```

The first command installs the dependencies in the active Python environment. The second downloads the pinned AMdataset images and verifies them. Initial training downloads Keras ImageNet weights. Use `--weights` with a local no-top MobileNetV2 weights file or `--data` with an existing dataset directory. The script recreates the fixed Study One split with seed 42, runs matched clean and augmented fits for three seeds per optimizer, and writes the trained models and audit files to `runs/`.

`runs/paper_comparison.csv` contains the clean-image comparison shown above. Validation decisions, per-image predictions and scores for synthetic conditions are generated as audit evidence. Use a fresh path with `train --out another-run` for subsequent runs.

## RMSprop repeat

A second RMSprop run with the same split, initial weights, and seeds reproduced all 3,444 per-image validation and test predictions. The augmented mean clean-image accuracy was **99.39%**, compared with **100.00%** in the original Study One. Seeds 42 and 44 scored 55/55. Seed 43 scored 54/55 and again misclassified the normal image `goodcylinder/IMG_8933.JPG` as defective.

Repeat this check with `python mobilenetv2.py train --optimizer rmsprop --out rmsprop-repeat`.

The project report is `MobileNetV2_Report.docx`.
