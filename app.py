"""
Brain Tumor MRI Image Classification — Streamlit application.

Loads the project's selected deployment model (MobileNetV2 transfer-learning
checkpoint, see README.md / Evaluation/ for why it was chosen over the
Custom CNN, EfficientNetB0, and ResNet50 candidates) and classifies an
uploaded brain MRI image into one of four classes: Glioma, Meningioma,
No Tumor, or Pituitary.

Educational project only — NOT a medical diagnostic tool. Predictions must
never be used for real clinical decisions.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "Models" / "mobilenetv2_best_validation.keras"
CLASS_NAMES_PATH = APP_DIR / "Models" / "class_names.json"
VALIDATION_SUMMARY_PATH = APP_DIR / "Models" / "mobilenetv2_validation_summary.json"

IMAGE_SIZE = (224, 224)
MAX_UPLOAD_MB = 10
ALLOWED_TYPES = ["jpg", "jpeg", "png", "bmp"]

CLASS_INFO = {
    "Glioma": "A tumor arising from glial (supporting) brain cells.",
    "Meningioma": "A tumor arising from the meninges, the membranes covering the brain.",
    "No Tumor": "No tumor pattern detected in this scan.",
    "Pituitary": "A tumor arising from the pituitary gland at the base of the brain.",
}

st.set_page_config(
    page_title="Brain Tumor MRI Classification",
    page_icon="🧠",
    layout="centered",
)


# --------------------------------------------------------------------------
# Cached resources
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model (first run only)...")
def load_model_and_metadata():
    """Load the trained Keras model and its class-order metadata once per
    session. Cached so the ~21 MB model is not reloaded on every click."""
    import tensorflow as tf  # imported lazily so the page skeleton renders fast

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            "Make sure Models/mobilenetv2_best_validation.keras is present in the repository."
        )
    if not CLASS_NAMES_PATH.is_file():
        raise FileNotFoundError(f"Class names file not found at {CLASS_NAMES_PATH}.")

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    class_names = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))

    validation_accuracy = None
    if VALIDATION_SUMMARY_PATH.is_file():
        summary = json.loads(VALIDATION_SUMMARY_PATH.read_text(encoding="utf-8"))
        validation_accuracy = summary.get("validation_accuracy")

    return model, class_names, validation_accuracy


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Reproduce the exact preprocessing used when MobileNetV2 was trained
    in Notebooks 1/3: decode as RGB, resize to 224x224 (float pixels in
    [0, 255]), then apply Keras's mobilenet_v2.preprocess_input, which
    scales pixels to [-1, 1]. Must stay consistent with training."""
    import tensorflow as tf

    image = image.convert("RGB")
    array = np.asarray(image).astype("float32")
    resized = tf.image.resize(array, IMAGE_SIZE).numpy()
    preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(resized)
    return np.expand_dims(preprocessed, axis=0)


def run_prediction(model, class_names, image: Image.Image):
    batch = preprocess_image(image)
    probabilities = model.predict(batch, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    return class_names[predicted_index], probabilities


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------

st.title("🧠 Brain Tumor MRI Image Classification")
st.caption(
    "Upload a brain MRI scan to classify it as **Glioma**, **Meningioma**, "
    "**No Tumor**, or **Pituitary** tumor, using a MobileNetV2 transfer-learning model."
)

st.warning(
    "⚠️ **Educational project only.** This tool is not a certified medical device and "
    "its predictions must never be used for real diagnosis or treatment decisions. "
    "Always consult a qualified radiologist or physician.",
    icon="⚠️",
)

with st.expander("ℹ️ About this model"):
    st.markdown(
        """
This app uses the **MobileNetV2** transfer-learning model from this project's
model comparison (see `README.md` and the `Evaluation/` folder for the full
comparison against the Custom CNN, EfficientNetB0, and ResNet50 candidates).

MobileNetV2 was selected for deployment because it combines strong accuracy
with a small (~21 MB), CPU-friendly footprint suited to free hosting —
full reasoning and measured numbers are documented in `README.md`.
        """
    )

uploaded_file = st.file_uploader(
    "Upload an MRI image",
    type=ALLOWED_TYPES,
    accept_multiple_files=False,
    help=f"Supported formats: {', '.join(ALLOWED_TYPES).upper()}. Max size: {MAX_UPLOAD_MB} MB.",
)

if uploaded_file is not None:
    file_size_mb = uploaded_file.size / (1024 * 1024)

    if file_size_mb > MAX_UPLOAD_MB:
        st.error(
            f"File is {file_size_mb:.1f} MB, which exceeds the {MAX_UPLOAD_MB} MB limit. "
            "Please upload a smaller image."
        )
    else:
        try:
            image = Image.open(uploaded_file)
            image.load()  # force decode now so corrupted files raise immediately
        except UnidentifiedImageError:
            st.error("This file could not be read as an image. Please upload a valid JPG, PNG, or BMP file.")
        except Exception:
            st.error("The uploaded file appears to be corrupted or unsupported. Please try a different image.")
        else:
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(image, caption="Uploaded MRI image", use_container_width=True)

            try:
                model, class_names, validation_accuracy = load_model_and_metadata()
            except FileNotFoundError as error:
                st.error(f"Model could not be loaded: {error}")
            else:
                with st.spinner("Classifying..."):
                    try:
                        predicted_class, probabilities = run_prediction(model, class_names, image)
                    except Exception as error:
                        st.error(f"Inference failed on this image: {error}")
                    else:
                        with col2:
                            confidence = float(np.max(probabilities)) * 100
                            st.metric("Predicted class", predicted_class, f"{confidence:.1f}% confidence")
                            st.caption(CLASS_INFO.get(predicted_class, ""))

                        st.subheader("Confidence by class")
                        ranked = sorted(
                            zip(class_names, probabilities), key=lambda item: item[1], reverse=True
                        )
                        for class_name, probability in ranked:
                            st.write(f"**{class_name}** — {probability * 100:.1f}%")
                            st.progress(float(probability))

                        if validation_accuracy is not None:
                            st.caption(
                                f"Model validation accuracy during training: {validation_accuracy * 100:.1f}% "
                                "(see README.md for full test-set metrics)."
                            )
else:
    st.info("Upload an MRI image above to get a prediction.")

st.divider()
st.caption(
    "Brain Tumor MRI Image Classification — educational deep learning project. "
    "Dataset: Labeled MRI Brain Tumor Dataset (Roboflow, CC BY 4.0)."
)
