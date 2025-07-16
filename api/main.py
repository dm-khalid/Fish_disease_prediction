from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import logging
import sys

# Ensure UTF-8 encoding for stdout (if possible)
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

# Initialize the app and configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app = FastAPI()

# Configure CORS
origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the EfficientNetB4 model
try:
    MODEL = tf.keras.models.load_model("model/efficientnetB3V3.keras")
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    raise RuntimeError("Failed to load the model. Check the model path or file integrity.")

# Define class names and solutions
CLASS_NAMES = [
    'Aeromoniasis (Bacterial disease)',
    'Bacterial Red disease',
    'Bacterial gill disease',
    'Fin and Tail Rot disease(Viral)',
    'Healthy Fish',
    'Parasitic diseases',
    'Saprolegniasis (Fungal disease)'
]

SOLUTIONS = {
    'Aeromoniasis (Bacterial disease)': 'Use antibiotics like Oxytetracycline in water.',
    'Bacterial Red disease': 'Treat water with potassium permanganate.',
    'Bacterial gill disease': 'Improve water quality and add formalin to water.',
    'Healthy Fish': 'No action needed. Fish is healthy.',
    'Parasitic diseases': 'Use anti-parasitic treatment like Praziquantel.',
    'Saprolegniasis (Fungal disease)': 'Apply malachite green or salt treatment.',
    'Fin and Tail Rot disease(Viral)': 'Isolate fish and improve water sanitation.'
}

# Function to safely handle and print outputs
def safe_print(message):
    try:
        print(message)
    except UnicodeEncodeError:
        print(message.encode('ascii', 'replace').decode('ascii'))

@app.get("/ping")
async def ping():
    """Health check endpoint."""
    return {"message": "Hello, I am alive"}

# Function to read and preprocess the image file
def read_file_as_image(data) -> np.ndarray:
    try:
        image = Image.open(BytesIO(data)).resize((224, 224))
        image = np.array(image) / 255.0
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image
    except Exception as e:
        logging.error(f"Error processing image file: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            raise HTTPException(status_code=415, detail="Unsupported file type. Please upload a PNG or JPG image.")
        
        image = read_file_as_image(await file.read())
        img_batch = np.expand_dims(image, axis=0)
        
        predictions = MODEL.predict(img_batch)
        predicted_class = CLASS_NAMES[np.argmax(predictions)]
        confidence = np.max(predictions)
        solution = SOLUTIONS.get(predicted_class, "No solution available.")
        
        safe_print(f"Predicted class: {predicted_class}, Confidence: {confidence:.2f}, Solution: {solution}")
        
        response_data = {
            'class': predicted_class,
            'confidence': round(float(confidence), 2),
            'solution': solution
        }
        return JSONResponse(content=response_data, headers={"Cache-Control": "no-store"})
    
    except HTTPException as http_err:
        logging.error(f"HTTP error: {http_err.detail}")
        raise http_err
    except Exception as e:
        logging.error(f"Unexpected error in prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)
