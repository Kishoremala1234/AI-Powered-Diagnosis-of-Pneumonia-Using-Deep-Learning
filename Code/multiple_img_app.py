import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import pandas as pd
import os

# ----------------------------------
# Streamlit setup (classic theme)
# ----------------------------------
st.set_page_config(page_title="Batch Pneumonia Testing", layout="centered")
st.title("🫁 Batch Pneumonia Testing")

st.info("Prediction threshold is fixed at 0.3")

# ----------------------------------
# Fixed configuration
# ----------------------------------
THRESHOLD = 0.3
OUTPUT_DIR = "Output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "test_results.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------
# User inputs
# ----------------------------------
model_file = st.file_uploader("Upload trained model (.h5)", type=["h5"])
input_folder = st.text_input("Enter folder path containing test images")

output_cols = ['S.no', 'Patient name', 'Pneumonia or Normal']

# ----------------------------------
# Prediction function (UNCHANGED LOGIC)
# ----------------------------------
def predict_pneumonia(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return "Pneumonia" if prediction[0][0] > THRESHOLD else "Normal"

# ----------------------------------
# Run batch testing
# ----------------------------------
if st.button("Run Batch Testing"):
    if not model_file or not input_folder:
        st.error("Model file and input folder are required.")
    elif not os.path.exists(input_folder):
        st.error("Input folder path does not exist.")
    else:
        with st.spinner("Running batch predictions..."):
            # ---- Save model to disk ----
            model_path = os.path.join(OUTPUT_DIR, model_file.name)
            with open(model_path, "wb") as f:
                f.write(model_file.getbuffer())

            model = tf.keras.models.load_model(model_path)

            # ---- Load or create CSV ----
            if os.path.exists(OUTPUT_FILE):
                df = pd.read_csv(OUTPUT_FILE)
                sno = df.shape[0] + 1
            else:
                df = pd.DataFrame(columns=output_cols)
                sno = 1

            log_lines = []

            # ---- Walk through folder ----
            for subdir, _, files in os.walk(input_folder):
                for fname in files:
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(subdir, fname)
                        class_name = os.path.basename(subdir)
                        patient_name = f"{class_name}_{os.path.splitext(fname)[0]}"

                        result = predict_pneumonia(img_path, model)

                        new_row = {
                            'S.no': sno,
                            'Patient name': patient_name,
                            'Pneumonia or Normal': result
                        }

                        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                        log_lines.append(f"{sno}, {patient_name}, {result}")
                        sno += 1

            # ---- Save CSV ----
            df.to_csv(OUTPUT_FILE, index=False)

        # ----------------------------------
        # Output
        # ----------------------------------
        st.success("Batch testing completed successfully")
        st.write("📁 **Results saved at:**")
        st.code(os.path.abspath(OUTPUT_FILE))
        st.subheader("Prediction Log")
        st.text("\n".join(log_lines))
