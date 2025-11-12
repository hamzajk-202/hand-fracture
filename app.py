import streamlit as st
from PIL import Image
import torch
import tempfile
import os

# -----------------------------
# 🎯 Load YOLOv11 Model
# -----------------------------
@st.cache_resource
def load_model():
    model_path = "best (1).pt"
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False)
    return model

model = load_model()

# -----------------------------
# 🌟 Streamlit Page Config
# -----------------------------
st.set_page_config(
    page_title="🩻 Fracture Detection using YOLOv11",
    page_icon="💀",
    layout="wide"
)

# -----------------------------
# 🎨 Custom CSS for Styling
# -----------------------------
st.markdown("""
    <style>
        body { background-color: #f8fafc; }
        .main-title {
            text-align: center;
            font-size: 36px;
            color: #0f172a;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            font-size: 18px;
            color: #475569;
            margin-bottom: 40px;
        }
        .stButton>button {
            background-color: #2563eb;
            color: white;
            font-size: 18px;
            border-radius: 10px;
            padding: 10px 25px;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #1e40af;
            transform: scale(1.05);
        }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# 🏷️ App Title
# -----------------------------
st.markdown("<h1 class='main-title'>🩻 Fracture Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload an X-ray image to automatically detect fractures using YOLOv11</p>", unsafe_allow_html=True)

# -----------------------------
# 📤 File Upload Section
# -----------------------------
uploaded_file = st.file_uploader("📸 Upload an X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_path = temp_file.name
        image.save(temp_path)

    # -----------------------------
    # 🚀 Detection Button
    # -----------------------------
    if st.button("🔍 Detect Fracture"):
        with st.spinner("Detecting fracture... Please wait..."):
            results = model(temp_path)
            results.render()  # Render detection boxes

            # Get the result image
            detected_img = Image.fromarray(results.ims[0])

            # Display result side-by-side
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original Image", use_container_width=True)
            with col2:
                st.image(detected_img, caption="Detected Fracture", use_container_width=True)

            # Optional: Display detection details
            st.subheader("📋 Detection Summary")
            st.dataframe(results.pandas().xyxy[0][["name", "confidence", "xmin", "ymin", "xmax", "ymax"]])

        st.success("✅ Detection Completed!")

else:
    st.info("👆 Please upload an image to begin fracture detection.")

# -----------------------------
# 🧾 Footer
# -----------------------------
st.markdown("""
---
💡 *Developed by Hamza Javed — Powered by YOLOv11 and Streamlit*
""")
