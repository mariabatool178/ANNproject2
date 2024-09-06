import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load the saved model
model = load_model('ann_model.h5')

# Function to preprocess the image
def preprocess_image(image):
    # Convert to grayscale and resize to 8x8 pixels
    image = image.convert('L')
    image = image.resize((8, 8))  # Resize to 8x8 pixels as in the dataset
    image = np.array(image) / 16.0  # Normalize (assuming the dataset uses 16 grayscale values)
    image = image.flatten().reshape(1, 64)  # Flatten to 64 features and reshape for model input
    return image

# Streamlit app interface
st.title("Digit Recognition App")

# Allow user to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess and predict
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        predicted_digit = np.argmax(prediction)

        st.write(f'Predicted Digit: {predicted_digit}')
    except Exception as e:
        st.error(f"An error occurred: {e}")
