# Running the MobileNetV2 reproduction

The README reproduces the English text and tables from `docs/MobileNetV2_Report_Tables.docx`. This file contains the execution instructions.

## Environment

Use Python 3.12. The saved experiment used Python 3.12.14; package versions are pinned in `requirements.txt`. Run commands from the repository root.

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe prepare_data.py
.\.venv\Scripts\python.exe mobilenetv2.py train --optimizer adam sgd rmsprop
```

Linux or macOS:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python prepare_data.py
.venv/bin/python mobilenetv2.py train --optimizer adam sgd rmsprop
```

The data preparation script downloads and verifies the original cylinder images from `mahsan2/AMdataset` at commit `52fe75dec70c1e8046493b4e3256f8f752a4d39d`. The first download is approximately 849 MB. The dataset is fetched from the authors' repository instead of duplicated here. ImageNet initialization weights are downloaded by Keras when first needed. Internet access is required for package installation and the initial downloads.

## Saved results and inference

`results/` contains the completed experiment: the data split, configuration, summary metrics, and the Adam, SGD and RMSprop models. Each optimizer directory contains the saved Keras model, epoch history, test predictions, test metrics and evaluation plot. New training outputs are written to `runs/` by default.

Use a saved model without retraining (after installing dependencies and downloading the images):

```powershell
.\.venv\Scripts\python.exe mobilenetv2.py predict --model results/adam/model.keras --image AMdataset/dataset/badcylinder/IMG_9191.JPG
```

Prediction returns the class and defect probability. On Linux or macOS, replace the Python executable with `.venv/bin/python`.

The default training configuration is seed 42, 50 epochs, batch size 30 and learning rate 0.001. The code freezes the ImageNet MobileNetV2 backbone and trains a new classification head. To use a local Keras MobileNetV2 no-top weights file, pass `--weights /path/to/weights.h5`; use `--data` and `--out` to change input and output directories. Do not point `--out` at the retained `results/` directory unless intentionally replacing the recorded experiment.

## Matched robustness experiment

`robustness.py` adapts the [EfficientNetB0 Study Three protocol](https://github.com/AHJ8273528/IDAT7215-EfficientNetB0/tree/c48cdc9ae66d9281d839a1ac55468215927a0c15) to the original MobileNetV2 architecture and the fixed Study One assignments in `results/split.csv`. Its six fresh fits are separate from the 50-epoch optimizer reproduction above.

After downloading the pinned images and installing dependencies, run:

```powershell
.\.venv\Scripts\python.exe robustness.py --out robust-runs
.\.venv\Scripts\python.exe mobilenetv2.py predict --model results/robustness/selected_seed42.keras --image AMdataset/dataset/badcylinder/IMG_9191.JPG
```

On Linux or macOS, use `.venv/bin/python` in place of the Windows executable. The training script uses Keras ImageNet weights by default. Pass `--weights` a path to the local MobileNetV2 no-top weights to avoid that download. The default output directory must be new. Use a new `--out` path for repeat runs.

Each matched pair shares the initial head weights, per-epoch image order, optimizer and update count. One view is used per training image per epoch. The control always uses clean input. The augmentation arm samples equally among clean, Gaussian blur (9 × 9, sigma 2), JPEG quality 15, downsampling to 56 × 56 followed by resizing back to 224 × 224, and rotation by +10° with reflection padding. Both arms are scored on those five conditions plus brightness multipliers 0.5 and 1.5. Evaluation applies each condition independently to the same original images.

The validation decision requires at least a 0.01 increase in mean F1 across seeds and conditions, with clean F1 and minimum condition-wise defect recall maintained. The script writes `selection.json` before extracting test features. `results/robustness/` contains the protocol, fit summaries, histories, selection, condition summaries, per-image predictions, saved seed-42 model and its inference checks. The exported model uses a single input view and preserves the original output order (defect, normal). Repeated seeds quantify training variation on a fixed split, and the corrupted tests measure synthetic robustness on those images.

## Sources

- Original implementation: https://github.com/mahsan2/AM3d/tree/dd1025d7180d91f1128c95817b4d3dd268626b1e
- Dataset: https://github.com/mahsan2/AMdataset/tree/52fe75dec70c1e8046493b4e3256f8f752a4d39d
- Paper: https://doi.org/10.1016/j.eswa.2024.124293

`mobilenetv2.py` and `prepare_data.py` implement this reproduction. The README and report include the methodological attribution and references. Upstream data and pretrained weights remain subject to their respective terms.
