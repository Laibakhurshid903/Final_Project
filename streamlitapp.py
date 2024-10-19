import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the saved CNN model
cnn_model = tf.keras.models.load_model('/content/Final_Project/cnn_model.h5')

# Define class names for the CNN model (update these based on your dataset)
cnn_class_names = ['Card Type A', 'Card Type B']  # Replace with your actual class names

# Streamlit app title
st.title("Card Classification App")
st.write("Upload a card image to classify it using the CNN model.")

# File uploader for image upload
uploaded_image = st.file_uploader("Choose a card image...", type=["jpg", "jpeg", "png"])

# Preprocess image function
def preprocess_image(image):
    img = image.resize((64, 64))  # Resize to match the model input size (update if needed)
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# If an image is uploaded, display and classify it
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    img_array = preprocess_image(image)
    
    # Make prediction
    cnn_prediction = cnn_model.predict(img_array)
    predicted_class = cnn_class_names[np.argmax(cnn_prediction)]
    
    st.write(f"The CNN model predicts: **{predicted_class}**")
