import os
import streamlit as st
from PIL import Image
from datetime import datetime
import csv

# Import our custom prediction function
from predict import predict

# Paths
CHECKPOINT_PATH = "checkpoints/best_model.pth"
FEEDBACK_PATH = "results/feedback_log.csv"
CONFUSION_MATRIX_PATH = "results/confusion_matrix.png"
METRICS_PATH = "results/metrics.txt"

# 1. Startup Checkpoint Validation
if not os.path.exists(CHECKPOINT_PATH):
    st.warning("⚠️ No trained model found. Please run `python train.py` first to train the model and save the checkpoint.")
    st.stop()

# Set page layout and aesthetics
st.set_page_config(
    page_title="Drone Flood Damage Assessment",
    page_icon="🛸",
    layout="wide"
)

# Custom Sleek CSS Styles
st.markdown("""
<style>
    .title-text {
        font-family: 'Outfit', 'Inter', sans-serif;
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 0px;
    }
    .subtitle-text {
        color: #4B5563;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .flooded-label {
        font-size: 1.8rem;
        font-weight: 800;
        color: #DC2626;
        background-color: #FEE2E2;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        display: inline-block;
        border: 2px solid #FCA5A5;
        margin-bottom: 1.2rem;
    }
    .not-flooded-label {
        font-size: 1.8rem;
        font-weight: 800;
        color: #16A34A;
        background-color: #DCFCE7;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        display: inline-block;
        border: 2px solid #86EFAC;
        margin-bottom: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to append feedback
def log_feedback(filename, predicted_label, confidence, feedback_type):
    os.makedirs(os.path.dirname(FEEDBACK_PATH) or ".", exist_ok=True)
    file_exists = os.path.exists(FEEDBACK_PATH)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(FEEDBACK_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "image_filename", "predicted_label", "confidence", "feedback"])
        writer.writerow([timestamp, filename, predicted_label, f"{confidence:.2f}%", feedback_type])

# Sidebar - System Control Panel
st.sidebar.title("🛸 System Control Center")
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎓 Course Details")
st.sidebar.info("**Course:** SOF106 Principles of AI\n\n**Project:** Drone Aerial Flood Damage Assessment")

# Display validation metrics if available
if os.path.exists(METRICS_PATH) or os.path.exists(CONFUSION_MATRIX_PATH):
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Model Performance")
    
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r") as f:
                lines = f.readlines()
            for line in lines[:2]:
                st.sidebar.caption(line.strip())
        except Exception:
            pass
            
    show_cm = st.sidebar.checkbox("Show Confusion Matrix")
    if show_cm and os.path.exists(CONFUSION_MATRIX_PATH):
        st.sidebar.image(CONFUSION_MATRIX_PATH, use_column_width=True)

# Main Application Headers
st.markdown("<h1 class='title-text'>Drone Aerial Flood Damage Assessment System</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle-text'>A deep learning demonstrator for post-disaster scene understanding. "
            "Upload drone/UAV images to detect building damage, mapping areas for emergency response.</p>", unsafe_allow_html=True)

# File Uploader
uploaded_file = st.file_uploader("Upload an aerial JPG or PNG image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded file to a temporary location for prediction function compatibility
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    # Set up session states to prevent duplicate feedback logging
    img_name = uploaded_file.name
    if "current_image" not in st.session_state or st.session_state.current_image != img_name:
        st.session_state.current_image = img_name
        st.session_state.feedback_submitted = False
        st.session_state.logged_feedback_type = None
        
    # Execute Model Prediction
    with st.spinner("Classifying aerial image via ResNet-50..."):
        try:
            label, confidence = predict(temp_path, CHECKPOINT_PATH)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()
            
    # Divide view into Columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📸 Drone View")
        image = Image.open(temp_path)
        st.image(image, use_column_width=True, caption=f"Uploaded File: {img_name}")
        
    with col2:
        st.markdown("### 🎯 Assessment Results")
        
        # Color-coded labels
        if label == "Flooded":
            st.markdown(f"<div class='flooded-label'>🚨 {label}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='not-flooded-label'>✅ {label}</div>", unsafe_allow_html=True)
            
        # Confidence score visualization
        st.markdown("**Assessment Confidence**")
        st.progress(confidence / 100.0)
        st.markdown(f"**Confidence Score:** `{confidence:.2f}%`")
        
        # Human feedback logging section
        st.markdown("---")
        st.markdown("#### 👤 Human-in-the-loop Feedback")
        st.write("Does this prediction match the visual ground truth?")
        
        if not st.session_state.feedback_submitted:
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if st.button("✅ Correct", use_container_width=True):
                    log_feedback(img_name, label, confidence, "Correct")
                    st.session_state.feedback_submitted = True
                    st.session_state.logged_feedback_type = "Correct"
                    st.rerun()
            with btn_col2:
                if st.button("❌ Wrong", use_container_width=True):
                    log_feedback(img_name, label, confidence, "Wrong")
                    st.session_state.feedback_submitted = True
                    st.session_state.logged_feedback_type = "Wrong"
                    st.rerun()
        else:
            st.success(f"Feedback logged ({st.session_state.logged_feedback_type}). Thank you!")
            
    # Clean up temp file
    try:
        os.remove(temp_path)
    except Exception:
        pass
else:
    st.info("💡 Please upload an aerial drone image to initiate triage analysis.")
