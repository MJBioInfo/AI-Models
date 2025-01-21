import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import tempfile

# Constants
IMG_WIDTH = 150
IMG_HEIGHT = 150
MODEL_PATH = "malaria_AI_model.h5"

# Load the model
@st.cache_resource
def load_malaria_model():
    model = load_model(MODEL_PATH)
    return model

# Function to evaluate the uploaded image
def evaluate_user_image(image, model, img_width, img_height):
    """
    Evaluates a single image provided by the user.

    Parameters:
        image (UploadedFile): Image file uploaded by the user.
        model (keras.Model): Trained model for prediction.
        img_width (int): Target width for resizing the image.
        img_height (int): Target height for resizing the image.
    """
    # Load and preprocess the image
    image = load_img(image, target_size=(img_width, img_height))
    img_arr = img_to_array(image)
    img_arr /= 255  # Normalize pixel values

    # Make a prediction
    pred = model.predict(img_arr.reshape(1, *img_arr.shape), verbose=0).flatten()
    label = "Parasitised" if pred < 0.5 else "Uninfected"

    # Return the label and image array for display
    return label, img_arr

# Streamlit App
def main():
    st.title("Malaria Cell Image Classifier")
    st.write("Upload a cell image to classify it as Parasitized or Uninfected.")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Load the model
        model = load_malaria_model()

        # Save uploaded image temporarily for processing
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_image_path = tmp_file.name

        # Evaluate the image
        label, img_arr = evaluate_user_image(tmp_image_path, model, IMG_WIDTH, IMG_HEIGHT)

        # Display the uploaded image
        st.image(img_arr, caption=f"Uploaded Image ({label})", use_column_width=True)

        # Display the prediction
        st.subheader(f"Prediction: {label}")

if __name__ == "__main__":
    main()

