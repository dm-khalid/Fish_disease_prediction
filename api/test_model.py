import tensorflow as tf
import numpy as np
from PIL import Image
import os
import sys

# Ensure UTF-8 encoding for stdout (if possible)
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

# Load your trained model
model = tf.keras.models.load_model("model/efficientnetB3final.keras")

# Define class names (as regular strings)
CLASS_NAMES = [
    'Aeromoniasis (Bacterial disease)', 
    'Bacterial Red disease', 
    'Bacterial gill disease', 
    'Healthy Fish', 
    'Parasitic diseases', 
    'Saprolegniasis (Fungal disease)', 
    'White tail disease (Viral)'
]

# Function to preprocess an image for model prediction
def preprocess_image(image_path):
    # Load and resize the image
    image = Image.open(image_path).resize((224, 224))
    image = np.array(image) / 255.0  # Normalize to [0, 1]
    
    # Convert RGBA images to RGB
    if image.shape[-1] == 4:  # Check for RGBA
        image = image[..., :3]  # Drop alpha channel
    
    # Expand dimensions to match model's expected input shape
    return np.expand_dims(image, 0)

# Function to safely handle and print outputs
def safe_print(message):
    try:
        print(message)
    except UnicodeEncodeError:
        # Encode as ASCII and replace unsupported characters
        print(message.encode('ascii', 'replace').decode('ascii'))

# Function to make a prediction on a single image
def predict_image(image_path):
    try:
        # Preprocess the image
        image = preprocess_image(image_path)
        
        # Make a prediction
        predictions = model.predict(image)
        
        # Extract predicted class and confidence score
        predicted_class = CLASS_NAMES[np.argmax(predictions)]
        confidence = np.max(predictions) * 100  # Confidence as percentage
        
        # Print results safely
        safe_print(f"Predicted Class: {predicted_class}, Confidence: {confidence:.2f}%")
        safe_print(f"Raw Predictions: {predictions[0]}")
    except Exception as e:
        safe_print(f"An error occurred: {e}")

# Ensure the image path is valid
image_path = os.path.join("api", "Parasitic diseases.jpeg")  # Replace with your actual test image path

# Call the prediction function
predict_image(image_path)
