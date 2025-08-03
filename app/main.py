from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import logging
import os
import shutil
from typing import List
from datetime import datetime

from src.prediction import predict_single, CLASS_NAMES
from src.model import train_model, retrain_on_new_images

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI(title="Plant Disease Classifier API")
START_TIME = datetime.utcnow()


@app.get("/", tags=["Welcome"])
def show_welcome():
    """Returns a welcome message."""
    return {"message": "Welcome to the Plant Disease Classification API!"}


@app.post("/predict/", tags=["Prediction"])
async def predict_image(file: UploadFile = File(...)):
    """
    Receives an image, makes a prediction, and returns the result.
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
    
    try:
        image_bytes = await file.read()
        
        class_name, confidence = predict_single(image_bytes)
        
        logger.info(f"Prediction successful: {class_name}, Confidence: {confidence:.2f}")
        
        return JSONResponse(content={
            "predicted_class": class_name,
            "confidence_score": confidence
        })
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during prediction.")


@app.get("/status/", tags=["Status"])
def get_status():
    """Returns the current status and uptime of the API."""
    uptime = datetime.utcnow() - START_TIME
    return {
        "status": "online",
        "uptime_seconds": uptime.total_seconds()
    }


@app.post("/retrain/", tags=["Retraining"])
async def trigger_retraining(files: List[UploadFile] = File(...)):
    """
    Receives new images, saves them to class-specific folders based on filename,
    and triggers the retraining process.
    """
    NEW_DATA_DIR = 'data/new_uploads'
    
    if os.path.exists(NEW_DATA_DIR):
        shutil.rmtree(NEW_DATA_DIR)
    os.makedirs(NEW_DATA_DIR)

    logger.info("--- Starting File Save Process ---")
    saved_file_count = 0
    
    for file in files:
        logger.info(f"Processing file: {file.filename}") # DEBUG: Log the original filename
        file_class = None
        for name in CLASS_NAMES:
            if file.filename.startswith(name):
                file_class = name
                break
        
        # DEBUG: Log whether a class was found
        if file_class:
            logger.info(f"   -> Found matching class: {file_class}")
            class_dir = os.path.join(NEW_DATA_DIR, file_class)
            os.makedirs(class_dir, exist_ok=True)
            
            file_path = os.path.join(class_dir, file.filename)
            contents = await file.read()
            with open(file_path, "wb") as buffer:
                buffer.write(contents)
            logger.info(f"   -> Successfully saved to: {file_path}") # DEBUG: Confirm save
            saved_file_count += 1
        else:
            logger.warning(f"   -> No matching class found for file: {file.filename}. Skipping.")

    logger.info(f"--- File Save Process Complete. Saved {saved_file_count} files. ---")

    # Check if any files were actually saved before trying to retrain
    if saved_file_count == 0:
        raise HTTPException(status_code=400, detail="No valid files were provided for retraining. Ensure filenames start with a valid class name.")

    try:
        retrain_on_new_images(new_data_dir=NEW_DATA_DIR)
        return JSONResponse(content={"message": "Model retraining on new images initiated successfully."})
    except Exception as e:
        logger.error(f"Error during retraining: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during retraining.")