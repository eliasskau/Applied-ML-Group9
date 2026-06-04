# Dog Emotion Detection — Applied ML Group 9

## Introduction

This project is for the Applied Machine Learning course at the Artificial Intelligence faculty. The goal is to classify the emotional state of a dog from an image into one of four categories: **angry**, **happy**, **relaxed**, or **sad**.

We trained two models:
- A custom **CNN** (5-block convolutional network with BatchNorm, Dropout, and early stopping)
- An **I-JEPA** based model

The best model is served via a **FastAPI** REST API that accepts a dog image and returns the predicted emotion with confidence scores.

---

## Results

| Model | Val Accuracy | Notes |
|---|---|---|
| Vanilla CNN  | ~65% | Early stopping, L1+L2 regularization |
| Random baseline | 25% | 4 equal classes |

---

## Repo structure

```bash
├── data/               # DVC-tracked image data
├── dog_emotion/
│   ├── models/         # CNN and I-JEPA model definitions
│   └── data/           # Preprocessing utilities
├── models/             # Saved .pth weight files
├── notebooks/          # Exploratory notebooks
├── tests/              # Unit tests
├── api.py              # FastAPI app
├── train_model.py      # CNN training script
├── hyperparam_search.py# LR hyperparameter search
├── Pipfile             # Dependency management
└── README.md
```

---

## Prerequisites to collaborate
Make sure you have the following software and tools installed:

- **Pipenv**: used for dependency management and virtual environments.

- **DVC**: manages data versioning. The dataset is stored on Google Drive via DVC. Contact a team member for access credentials.

---

