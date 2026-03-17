import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# -----------------------------
# SEITENKONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="YOLOv8 Objekterkennung",
    page_icon="🤖",
    layout="centered"
)

# -----------------------------
# TITEL & BESCHREIBUNG
# -----------------------------
st.title("🤖 YOLOv8 Objekterkennung")
st.markdown(
    "Lade ein Bild hoch und die KI erkennt automatisch alle Objekte im Bild."
)

# -----------------------------
# YOLO MODELL LADEN
# (wird nur einmal geladen)
# -----------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # kleines & schnelles Modell

model = load_model()

# -----------------------------
# DATEI-UPLOAD
# -----------------------------
uploaded_file = st.file_uploader(
    "📤 Lade ein Bild hoch",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# WENN BILD HOCHGELADEN
# -----------------------------
if uploaded_file is not None:

    # Bild laden
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    st.subheader("📷 Originalbild")
    st.image(image, use_container_width=True)

    # -----------------------------
    # OBJEKTERKENNUNG
    # -----------------------------
    with st.spinner("🔍 Erkenne Objekte..."):
        results = model(image_np)

    # Ergebnisbild (mit Boxen)
    result_image = results[0].plot()

    st.subheader("🎯 Erkannte Objekte")
    st.image(result_image, use_container_width=True)

    # -----------------------------
    # ERGEBNISSE AUSLESEN
    # -----------------------------
    boxes = results[0].boxes

    if boxes is not None:

        st.subheader("📋 Details")

        for box in boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            label = model.names[class_id]

            st.write(
                f"**Objekt:** {label}  \n"
                f"**Sicherheit:** {confidence * 100:.2f}%"
            )

    else:
        st.warning("Keine Objekte erkannt.")
