from fastapi import FastAPI, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import joblib
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="."), name="static")

# Global variables for model and vectorizer
model = None
vectorizer = None

def load_models():
    global model, vectorizer
    try:
        logger.info("Loading ML models...")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Model file exists: {os.path.exists('ai_detection_model.pkl')}")
        logger.info(f"Vectorizer file exists: {os.path.exists('vectorizer.pkl')}")
        
        model = joblib.load("ai_detection_model.pkl")
        vectorizer = joblib.load("vectorizer.pkl")
        
        # Test the model and vectorizer
        test_text = "This is a test email"
        X_test = vectorizer.transform([test_text])
        prediction = model.predict(X_test)
        logger.info(f"Model test prediction: {prediction}")
        
        logger.info("ML models loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        return False

# Load models on startup
if not load_models():
    raise HTTPException(status_code=500, detail="Error loading ML models")

@app.get("/")
async def read_root():
    return FileResponse('index.html')

@app.post("/predict")
async def predict(email: str = Form(...)):
    try:
        if model is None or vectorizer is None:
            if not load_models():
                raise HTTPException(status_code=500, detail="Error loading models")
            
        logger.info(f"Received prediction request for email: {email[:50]}...")
        X = vectorizer.transform([email])
        prediction = model.predict(X)[0]
        prob = max(model.predict_proba(X)[0])
        result = {"label": "legitimate" if prediction == 1 else "spam", "confidence": round(prob, 2)}
        logger.info(f"Prediction completed: {result}")
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

@app.get("/checking")
async def checking():
    return FileResponse('checking.html')
