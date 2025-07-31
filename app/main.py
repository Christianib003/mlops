from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import logging

from src.prediction import predict_single
from src.model import retrain_model


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(title="Plant Disease Classifier API")


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


@app.post("/retrain/", tags=["Retraining"])
def trigger_retraining():
    """
    Triggers the model retraining process.
    (Note: This is a blocking operation and in a real-world scenario,
    this would be handled by a background worker.)
    """
    try:
        logger.info("Retraining endpoint triggered.")
        retrain_model()
        
        return JSONResponse(content={"message": "Model retraining initiated successfully."})
    except Exception as e:
        logger.error(f"Error during retraining: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during retraining.")