import logging
import warnings
from pathlib import Path
from typing import Any

import fasttext
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Suppress fasttext warnings
warnings.filterwarnings("ignore", message=".*warn_on_stderr.*")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="fastText Language Detection API", version="1.0.0")

# Load model at startup
model = None
MODEL_PATH = "/models/lid.176.bin"


@app.on_event("startup")
async def load_model():
    global model
    if not Path(MODEL_PATH).exists():
        logger.error(f"Model file not found at {MODEL_PATH}")
        raise RuntimeError(
            "Model file not found. Please ensure download_model.py ran successfully."
        )

    logger.info("Loading fastText language detection model...")
    model = fasttext.load_model(MODEL_PATH)
    logger.info("Model loaded successfully")


class TextInput(BaseModel):
    text: str
    k: int = 1  # Number of top predictions to return


class LanguageDetection(BaseModel):
    language: str
    confidence: float


class DetectionResponse(BaseModel):
    predictions: list[LanguageDetection]


@app.post("/detect", response_model=DetectionResponse)
async def detect_language(input_data: TextInput):
    """
    Detect language of input text

    - **text**: Text to analyze
    - **k**: Number of top predictions to return (default: 1, max: 10)
    """
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not input_data.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if input_data.k < 1 or input_data.k > 10:
        raise HTTPException(status_code=400, detail="k must be between 1 and 10")

    # Predict language (replace newlines to avoid issues)
    predictions = model.predict(input_data.text.replace("\n", " "), k=input_data.k)

    # Format results
    languages = [lang.replace("__label__", "") for lang in predictions[0]]
    confidences = [float(conf) for conf in predictions[1]]

    results = [
        LanguageDetection(language=lang, confidence=conf)
        for lang, conf in zip(languages, confidences, strict=True)
    ]

    return DetectionResponse(predictions=results)


@app.post("/batch-detect")
async def batch_detect_language(texts: list[str], k: int = 1):
    """
    Detect language for multiple texts in batch

    - **texts**: List of texts to analyze
    - **k**: Number of top predictions per text (default: 1)
    """
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not texts:
        raise HTTPException(status_code=400, detail="Texts list cannot be empty")

    if len(texts) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 texts per batch")

    if k < 1 or k > 10:
        raise HTTPException(status_code=400, detail="k must be between 1 and 10")

    results: list[dict[str, Any]] = []
    for text in texts:
        if not text.strip():
            results.append({"predictions": []})
            continue

        predictions = model.predict(text.replace("\n", " "), k=k)
        languages = [lang.replace("__label__", "") for lang in predictions[0]]
        confidences = [float(conf) for conf in predictions[1]]

        text_results = [
            {"language": lang, "confidence": conf}
            for lang, conf in zip(languages, confidences, strict=True)
        ]
        results.append({"predictions": text_results})

    return results


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
    }


@app.get("/")
async def root():
    """API information"""
    return {
        "name": "fastText Language Detection API",
        "version": "1.0.0",
        "supported_languages": 176,
        "model": "lid.176.bin",
        "endpoints": {
            "POST /detect": "Detect language of single text",
            "POST /batch-detect": "Detect language of multiple texts",
            "GET /health": "Health check",
            "GET /docs": "Interactive API documentation",
        },
    }
