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

## Installation

```bash
git clone https://github.com/eliasskau/Applied-ML-Group9
cd Applied-ML-Group9
pip install pipenv
pipenv install
pipenv shell
```

Pull the dataset (requires DVC Google Drive access & json key file availible on request):
```bash
dvc pull
```

---

## Training

Train the CNN from scratch:
```bash
python train_model.py
```

This will:
- Train for up to 100 epochs with early stopping (patience 7)
- Save best weights to `models/best_CNN_dog_model.pth`
- Output train/val accuracy plot to `models/train_val_accuracy.png`
- Output confusion matrix to `models/confusion_matrix.png`
- Print test accuracy with 95% bootstrap confidence interval

Run learning rate search:
```bash
python hyperparam_search.py
```

---

## Running the API

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

API will be available at `http://localhost:8000`  
Interactive docs at `http://localhost:8000/docs`

Example request:
```bash
curl -X POST http://localhost:8000/predict \
     -H "accept: application/json" \
     -F "file=@dog.jpg"
```

Example response:
```json
{
  "predicted_emotion": "happy",
  "confidence": 0.82,
  "probabilities": {
    "angry": 0.04,
    "happy": 0.82,
    "relaxed": 0.11,
    "sad": 0.03
  }
}
```
