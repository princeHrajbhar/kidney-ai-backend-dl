import os
import io
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from .model_loader import kidney_ai  # Importing our logic
from .schemas import PredictionResponse # Importing our validation

app = FastAPI(title="Kidney AI REST API", version="2.0.0")

# CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- FIX: Root Route to stop 404 errors ---
@app.get("/")
async def root():
    return {
        "message": "Kidney AI API is Online",
        "docs": "/docs",
        "status": "ready"
    }

# --- Health Check ---
@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File is not an image")

    try:
        # Read image
        data = await file.read()
        image = Image.open(io.BytesIO(data)).convert("RGB").resize(kidney_ai.img_size)
        
        # Convert to array
        img_array = np.expand_dims(tf.keras.utils.img_to_array(image), axis=0)
        
        # Predict
        predictions = kidney_ai.predict(img_array)
        score = float(predictions[0][0])
        
        label = "Tumor" if score > 0.5 else "Normal"
        confidence = score if score > 0.5 else 1 - score

        return PredictionResponse(
            filename=file.filename,
            prediction=label,
            confidence=f"{confidence:.2%}",
            is_tumor=(score > 0.5)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))