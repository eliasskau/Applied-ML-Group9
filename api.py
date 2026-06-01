"""
GET /  — API info and available endpoints
GET /health    — health check (confirm model is loaded)
GET /classes   — list of supported emotion classes
POST /predict   — upload an image, receive emotion prediction

How to run locally:
    pip install fastapi uvicorn torch torchvision pillow
    uvicorn api:app --reload --host 0.0.0.0 --port 8000

open: http://localhost:8000/docs for interactive Swagger documentation.
use in terminal: curl.exe -X POST http://localhost:8000/predict -F "file=@<filename>"
"""

import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

MODEL_PATH  = "models/best_distilled_student.pth"
IMG_SIZE    = 224
NUM_CLASSES = 4
CLASSES     = ["angry", "happy", "relaxed", "sad"]
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/webp"}

preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _load_model(weights_path: str) -> nn.Module:
    model = models.mobilenet_v3_small(weights=None)
    num_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_features, NUM_CLASSES)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

try:
    _model = _load_model(MODEL_PATH)
    _model_loaded = True
except FileNotFoundError:
    # when app starts w/o weights /health should report w/o a crash
    _model = None
    _model_loaded = False

class EmotionProbabilities(BaseModel):
    angry:   float = Field(..., ge=0.0, le=1.0, description="Probability for 'angry'")
    happy:   float = Field(..., ge=0.0, le=1.0, description="Probability for 'happy'")
    relaxed: float = Field(..., ge=0.0, le=1.0, description="Probability for 'relaxed'")
    sad:     float = Field(..., ge=0.0, le=1.0, description="Probability for 'sad'")

class PredictionResponse(BaseModel):
    """
    predicted_emotion   : emotion label with highest confidence
    confidence  : softmax probability of predicted class
    probabilities   : softmax distribution over all emotion classes
    """
    predicted_emotion: str = Field(
        ...,
        description="Predicted emotion label: angry | happy | relaxed | sad",
        example="happy",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence for the predicted emotion (0.0 – 1.0)",
        example=0.82,
    )
    probabilities: EmotionProbabilities = Field(
        ...,
        description="Softmax probability for every emotion class",
    )

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    num_classes: int

class InfoResponse(BaseModel):
    name: str
    description: str
    version: str
    endpoints: dict

app = FastAPI(
    title="Dog Emotion Detection API",
    description=(
        "Classifies the emotional state of a dog from an uploaded image using "
        "a distilled MobileNetV3-Small model. "
        "Supported emotions: **angry**, **happy**, **relaxed**, **sad**. "
        "The API handles all preprocessing internally (resizing to 224×224 and "
        "ImageNet normalisation) — just send a raw JPEG or PNG image and receive "
        "a human-readable emotion label with confidence scores."
    ),
    version="1.0.1",
)

@app.get("/", response_model=InfoResponse, tags=["Info"])
def root() -> InfoResponse:
    """
    Returns API information, endpoints
    """
    return InfoResponse(
        name="Dog Emotion Detection API",
        description=(
            "Upload a dog image and receive its predicted emotional state. "
            "The model is a distilled MobileNetV3-Small trained on four emotion categories: "
            "angry, happy, relaxed, and sad."
        ),
        version="1.0.1",
        endpoints={
            "GET /"     : "API info (this response)",
            "GET /health"   : "Health check — confirms the model is loaded and ready",
            "GET /classes"  : "List of supported emotion class labels with descriptions",
            "POST /predict"     : "Upload a dog image and receive an emotion prediction",
        },
    )

@app.get("/health", response_model=HealthResponse, tags=["Info"])
def health() -> HealthResponse:
    """
    Returns current API status, model health, or code 503 if the weights could not be found
    """
    if not _model_loaded:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Model weights not found at '{MODEL_PATH}'. "
                "Place best_distilled_student.pth in the repo root and restart the API."
            ),
        )
    return HealthResponse(
        status="ok",
        model_loaded=_model_loaded,
        device=str(device),
        num_classes=NUM_CLASSES,
    )

@app.get("/classes", tags=["Info"])
def get_classes() -> dict:
    """
    Returns list of emotion classes the model can predict, short description of each class
    """
    return {
        "classes": CLASSES,
        "descriptions": {
            "angry":   "The dog shows signs of aggression or irritation (e.g. bared teeth, tense posture).",
            "happy":   "The dog looks content and cheerful (e.g. relaxed mouth, bright eyes).",
            "relaxed": "The dog is calm and at ease (e.g. loose body, soft gaze).",
            "sad":     "The dog appears downcast or withdrawn (e.g. lowered head, drooping ears).",
        },
    }

@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Classify dog emotion from an uploaded image",
)
async def predict(
    file: UploadFile = File(
        ...,
        description=(
            "A JPEG, PNG, BMP, or WebP image of a dog. "
            "The image can be any resolution — the API resizes it to 224×224 internally. "
            "No normalisation is required from the caller."
        ),
    ),
) -> PredictionResponse:
    """
    Returns most likely emotion, confidence, and probability of all emotions
    Error codes:
    400 — uploaded file is empty or cannot be decoded
    415 — unsupported file type
    503 — model weights not loaded at startup

    Example curl request:
    ```
    curl -X POST http://localhost:8000/predict \\
         -H "accept: application/json" \\
         -F "file=@dog.jpg"
    ```
    """
    if not _model_loaded or _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Check /health for details.",
        )

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type '{file.content_type}'. "
                f"Accepted types: {sorted(ALLOWED_CONTENT_TYPES)}"
            ),
        )

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail=(
                "Could not decode the uploaded file as an image. "
                "Ensure the file is a valid JPEG, PNG, BMP, or WebP."
            ),
        )

    tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = _model(tensor)
        probs  = F.softmax(logits, dim=1)[0]

    top_idx   = int(torch.argmax(probs).item())
    top_label = CLASSES[top_idx]
    top_conf  = float(probs[top_idx].item())
    prob_dict = {cls: round(float(probs[i].item()), 4) for i, cls in enumerate(CLASSES)}

    return PredictionResponse(
        predicted_emotion=top_label,
        confidence=round(top_conf, 4),
        probabilities=EmotionProbabilities(**prob_dict),
    )