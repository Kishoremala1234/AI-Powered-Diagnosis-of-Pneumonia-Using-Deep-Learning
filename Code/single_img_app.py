import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import pandas as pd
import os
from PIL import Image

# ----------------------------------
# Streamlit setup (classic theme)
# ----------------------------------
st.set_page_config(page_title="Pneumonia Detection", layout="centered")
st.title("🫁 Pneumonia Detection – Single Image")

st.info("Prediction threshold is fixed at 0.3")

# ----------------------------------
# Fixed configuration
# ----------------------------------
THRESHOLD = 0.3
OUTPUT_DIR = "Output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "single_test_results.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------
# User inputs
# ----------------------------------
model_file = st.file_uploader("Upload trained model (.h5)", type=["h5"])
image_file = st.file_uploader("Upload chest X-ray image", type=["png", "jpg", "jpeg"])
patient_name = st.text_input("Enter patient name")

output_cols = ['S.no', 'Patient name', 'Pneumonia or Normal']

# ----------------------------------
# Prediction function (SAME LOGIC)
# ----------------------------------
def predict_pneumonia(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return "Pneumonia" if prediction[0][0] > THRESHOLD else "Normal"

# ----------------------------------
# Run prediction
# ----------------------------------
if st.button("Detect Pneumonia"):
    if not model_file or not image_file or not patient_name:
        st.error("Model file, image, and patient name are required.")
    else:
        with st.spinner("Running prediction..."):
            # ---- Save model ----
            model_path = os.path.join(OUTPUT_DIR, model_file.name)
            with open(model_path, "wb") as f:
                f.write(model_file.getbuffer())

            model = tf.keras.models.load_model(model_path)

            # ---- Save image ----
            image_path = os.path.join(OUTPUT_DIR, image_file.name)
            with open(image_path, "wb") as f:
                f.write(image_file.getbuffer())

            # ---- Predict ----
            result = predict_pneumonia(image_path, model)

            # ---- Load or create CSV ----
            if os.path.exists(OUTPUT_FILE):
                df = pd.read_csv(OUTPUT_FILE)
                sno = df.shape[0] + 1
            else:
                df = pd.DataFrame(columns=output_cols)
                sno = 1

            new_row = {
                'S.no': sno,
                'Patient name': patient_name,
                'Pneumonia or Normal': result
            }

            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv(OUTPUT_FILE, index=False)

        # ----------------------------------
        # Output
        # ----------------------------------
        st.image(Image.open(image_path), caption="Uploaded X-ray", use_container_width=True)
        st.success(f"Prediction: **{result}**")
        st.write("📁 **Result saved at:**")
        st.code(os.path.abspath(OUTPUT_FILE))
