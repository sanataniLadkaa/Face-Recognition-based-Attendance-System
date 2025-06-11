from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Replace with the path to your test dataset directory
test_dataset_directory = "AS1/dataset/Shivangi Shandilya"

def load_test_data():
    images = []
    labels = []
    image_size = (224, 224)  # Ensure this matches your model's input size

    # Loop through the dataset directory and load images
    for class_label in os.listdir(test_dataset_directory):
        class_path = os.path.join(test_dataset_directory, class_label)
        if os.path.isdir(class_path):
            for image_file in os.listdir(class_path):
                image_path = os.path.join(class_path, image_file)
                image = load_img(image_path, target_size=image_size)
                image = img_to_array(image) / 255.0  # Normalize the image
                images.append(image)
                labels.append(class_label)
    
    # Convert lists to numpy arrays
    X = np.array(images)
    y = np.array(labels)
    return X, y

# Load the test dataset
testX, testy = load_test_data()

# Encode string labels to numeric values
label_encoder = LabelEncoder()
testy_encoded = label_encoder.fit_transform(testy)

# Load the trained model
model = load_model("face_recognition_model.h5")

# Evaluate the model on test data
loss, accuracy = model.evaluate(testX, testy_encoded, verbose=1)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Make predictions
predictions = model.predict(testX, verbose=1)
predicted_labels = np.argmax(predictions, axis=1)

# Confusion matrix
cm = confusion_matrix(testy_encoded, predicted_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues', 
    xticklabels=label_encoder.classes_, 
    yticklabels=label_encoder.classes_
)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Classification report
print("Classification Report:")
print(classification_report(testy_encoded, predicted_labels, target_names=label_encoder.classes_))
