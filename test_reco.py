import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image

# Parameters
image_size = (224, 224)

# Update this mapping with actual names of the persons
label_mapping = {
    0: "Anurag",        # Class 0 corresponds to John Doe
    1: "Anurag T",      # Class 1 corresponds to Jane Smith
    2: "Shivangi",   # Class 2 corresponds to Alice Johnson
    3: "Shivangi Singh",       # Class 3 corresponds to Bob Brown
    # Add more labels corresponding to the classes in your dataset
}

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocess a single image for prediction.
    
    Args:
        image_path (str): Path to the image file.
        target_size (tuple): Target size for resizing the image.
        
    Returns:
        numpy.ndarray: Preprocessed image ready for the model.
    """
    try:
        # Load image using PIL
        image = Image.open(image_path).convert("RGB")
        
        # Resize to the target size
        image_resized = image.resize(target_size)
        
        # Convert to numpy array and normalize to range [0, 1]
        image_array = np.array(image_resized).astype("float32") / 255.0
        
        # Expand dimensions for batch size compatibility
        image_batch = np.expand_dims(image_array, axis=0)
        
        # Apply VGG16-specific preprocessing
        preprocessed_image = preprocess_input(image_batch)
        return preprocessed_image
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def verify_model(image_path):
    """
    Verify the functionality of the trained model using a test image.
    
    Args:
        image_path (str): Path to the image to be tested.
    """
    try:
        # Load the trained model
        model = load_model("face_model.h5")
        print("Model loaded successfully.")

        # Preprocess the test image
        preprocessed_image = preprocess_image(image_path, target_size=image_size)
        if preprocessed_image is None:
            print("Image preprocessing failed.")
            return

        # Make a prediction
        predictions = model.predict(preprocessed_image)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)

        # Map the predicted class to the name of the person
        predicted_label = label_mapping.get(predicted_class, "Unknown")
        
        # Output the prediction
        print(f"Predicted Person: {predicted_label}")
        print(f"Confidence: {confidence:.2f}")
    except Exception as e:
        print(f"Error verifying the model: {e}")

# Test the model with an example image
test_image_path = "AS1/WhatsApp Image 2024-06-25 at 16.43.14_8db353d0-Photoroom.jpg"  # Replace with the actual test image path
verify_model(test_image_path)
