# Multi-task CNN for Plant Species Images

CNNs with transfer learning for joint plant species classification and bounding-box localization (CSI 5341, University of Ottawa, Fall 2023).

## The Assignment

Given 1,000 plant seedling images (4 species × 250: Cleavers, Common Chickweed, Maize, Shepherd's Purse — a subset of the [Aarhus Plant Seedlings dataset](https://vision.eng.au.dk/plant-seedlings-dataset/)), split into 750 train / 125 validation / 125 test, the assignment asked to:

- **3.1 — VGG-16 classifier**: use the first two blocks of VGG-16 (frozen, no fine-tuning) as a backbone, add at most five fully-connected/convolutional layers (no regularization), train on images ≤128×128, and report learning curves plus confusion matrices on train/validation/test.
- **3.2 — ResNet-50 classifier**: same classification task using all but the top fully-connected layers of ResNet-50 (frozen), adding at most two layers; report learning curves.
- **3.3 — Discussion**: judge from the learning curves whether the small training set causes overfitting in 3.1 and 3.2.
- **4.1 — VGG-16 bounding-box regression**: same backbone as 3.1 with a regression head predicting the normalized bounding-box height and width (two floats in [0, 1], ground truth from `bbox.json`); report learning curves and MSE on train/validation/test.
- **4.2 — Multi-task + regularization**: combine the 3.1 and 4.1 networks into one multi-task model and add a second regularization method (BatchNorm, Dropout); report learning curves and metrics.
- **4.3 — Discussion**: quantify how much the multi-task setup and the extra regularization improve over 4.1.

## Key Findings

- **Without regularization, the VGG classifier overfits**: training accuracy hits ~0.98 while validation stalls around 0.75 and even dips late in training (test ~72%). The gap between the curves is the whole story — the model memorizes the 750 training images instead of learning species features.
- **ResNet-50 was too much model for too little data**: validation/test accuracy collapsed to ~30% with wild loss fluctuations. A 25M-parameter backbone fine-tuned on 750 images just thrashes — capacity must match dataset size.
- **Multi-task learning fixed most of it**: sharing the backbone between classification and localization forces it to learn features useful for *both* tasks, which acts as regularization. With BatchNorm + Dropout added, validation accuracy rose from ~75% to ~90% and test from ~72% to ~85%, with validation/test loss dropping ~0.4.
- **Localization improved too, modestly**: bbox regression MSE fell from 0.07 → 0.06 (validation) and 0.1 → 0.07 (test) — the shared representation helped both heads, not just classification.

## Dataset

The images are not bundled in this repo (200MB+). Only four species folders are used, with bounding-box annotations in `bbox.json`:

```
a3_dataset/
├── img/
│   ├── Cleavers/            # 250 images
│   ├── Common_Chickweed/    # 250 images
│   ├── Maize/               # 250 images
│   └── Shepherds_Purse/     # 250 images
└── bbox.json                # {class: {image: {height, width}}} ground-truth boxes
```

Source: subset of the Aarhus University Plant Seedlings dataset (Giselsson et al., 2017), distributed via the course. To run the notebook, place `a3_dataset/` at the repo root as shown above (images are resized to 64×64 and normalized at load time).

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook multi_task_cnn_plant_species.ipynb
```

## Repository Structure

```
├── multi_task_cnn_plant_species.ipynb  # classification, regression, multi-task
└── requirements.txt
```
