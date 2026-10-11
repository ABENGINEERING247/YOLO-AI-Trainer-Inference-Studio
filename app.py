import streamlit as st
import os
import zipfile
import tempfile
import shutil
import cv2
from PIL import Image
import numpy as np
from ultralytics import YOLO

# --- Page Configuration ---
st.set_page_config(
    page_title="YOLO AI Power Studio | Engr. Bilal Mehmood",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- High-Contrast & Energetic Styling ---
st.markdown("""
    <style>
    /* Global App Background & High Contrast Font */
    html, body, [data-testid="stAppViewContainer"], .stApp {
        background-color: #0B0F19 !important;
        color: #FFFFFF !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Force General Text Elements to White */
    h1, h2, h3, h4, h5, h6, p, span, label, div {
        color: #FFFFFF !important;
    }

    /* Vibrant Top Header Banner */
    .app-header-banner {
        background: linear-gradient(135deg, #FF4500 0%, #FF8C00 50%, #FFD700 100%);
        padding: 30px 35px;
        border-radius: 20px;
        color: #000000 !important;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px rgba(255, 69, 0, 0.4);
    }
    .app-header-banner h1, .app-header-banner p, .app-header-banner b {
        color: #000000 !important;
    }
    .app-header-banner h1 {
        font-size: 2.6rem;
        font-weight: 900;
        text-transform: uppercase;
        margin: 0;
    }
    .app-header-banner p {
        font-size: 1.15rem;
        font-weight: 800;
        margin-top: 8px;
    }

    /* Sidebar High Contrast Theme */
    [data-testid="stSidebar"] {
        background-color: #111827 !important;
        border-right: 2px solid #1F2937;
    }
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] span {
        color: #F3F4F6 !important;
        font-weight: 600 !important;
    }

    /* High Contrast Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #1F2937 !important;
        border: 2px solid #3B82F6 !important;
        border-radius: 12px !important;
        padding: 15px !important;
    }
    div[data-testid="stMetricValue"] {
        color: #60A5FA !important;
        font-weight: 900 !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #9CA3AF !important;
        font-weight: 700 !important;
    }

    /* Tab Header High Contrast */
    button[data-baseweb="tab"] {
        font-weight: 800 !important;
        font-size: 1.1rem !important;
        color: #60A5FA !important;
    }

    /* UPLOAD BUTTONS & BOXES WITH BLACK TEXT */
    [data-testid="stFileUploader"] {
        background-color: #F8FAFC !important;
        border: 2px dashed #3B82F6 !important;
        border-radius: 12px !important;
        padding: 20px !important;
        color: #000000 !important;
    }
    [data-testid="stFileUploader"] section, 
    [data-testid="stFileUploader"] span, 
    [data-testid="stFileUploader"] small, 
    [data-testid="stFileUploader"] div, 
    [data-testid="stFileUploader"] p {
        color: #000000 !important;
        font-weight: 700 !important;
    }
    [data-testid="stFileUploader"] button {
        background-color: #FF8C00 !important;
        color: #000000 !important;
        font-weight: 900 !important;
        border: 2px solid #000000 !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- Session State Management ---
if "trained_model_path" not in st.session_state:
    st.session_state.trained_model_path = None

if "show_welcome_popup" not in st.session_state:
    st.session_state.show_welcome_popup = True

# --- POPUP ALERT / BANNER MESSAGE ---
if st.session_state.show_welcome_popup:
    st.markdown("""
        <div style="
            background: linear-gradient(135deg, #10B981 0%, #059669 100%);
            border: 3px solid #34D399;
            border-radius: 14px;
            padding: 20px;
            text-align: center;
            margin-bottom: 25px;
            box-shadow: 0 10px 25px rgba(16, 185, 129, 0.4);
        ">
            <h2 style="color: #FFFFFF !important; margin: 0; font-size: 1.8rem; font-weight: 900;">
                🚀 AI BASED DETECTION SYSTEM BNAANA HUA AB BILKUL ASAAN! 🔥
            </h2>
            <p style="color: #ECFDF5 !important; margin: 8px 0 0 0; font-size: 1.1rem; font-weight: 700;">
                Welcome to your ultimate computer vision studio. Train models and run real-time inference effortlessly!
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    col_pop1, col_pop2, col_pop3 = st.columns([2, 1, 2])
    with col_pop2:
        if st.button("✨ Got it! Let's Start", use_container_width=True, type="primary"):
            st.session_state.show_welcome_popup = False
            st.rerun()

# --- TOP HEADER BANNER ---
st.markdown("""
    <div class="app-header-banner">
        <h1>🔥 ULTRALYTICS YOLO AI POWER STUDIO 🔥</h1>
        <p>Next-Gen Model Training & Real-Time Computer Vision Inference | Architected by <b>Engr. Bilal Mehmood</b></p>
    </div>
""", unsafe_allow_html=True)

# Create Tabs for Navigation
tab_train, tab_infer, tab_guide = st.tabs(["🚀 Train Custom YOLO Model", "🎥 High-Speed Media Inference Studio", "📖 Comprehensive Guide Book"])

# ==========================================
# TAB 1: TRAIN MODEL
# ==========================================
with tab_train:
    st.markdown("### ⚡ Step 1: Configure Your Neural Architecture & Dataset")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        model_size = st.selectbox(
            "Select YOLO Base Architecture", 
            [
                "yolov8n.pt (Nano - Fastest)", 
                "yolov8s.pt (Small - Balanced)", 
                "yolov8m.pt (Medium - High Accuracy)",
                "yolov8l.pt (Large - Heavy Duty)",
                "yolov9c.pt (YOLOv9 Compact)",
                "yolov9e.pt (YOLOv9 Extended)",
                "yolov10n.pt (YOLOv10 Nano)",
                "yolov10s.pt (YOLOv10 Small)",
                "yolov10m.pt (YOLOv10 Medium)",
                "yolov11n.pt (YOLOv11 Nano 🔥)",
                "yolov11s.pt (YOLOv11 Small 🔥)",
                "yolov11m.pt (YOLOv11 Medium 🔥)",
                "yolov11l.pt (YOLOv11 Large 🔥)"
            ],
            index=9
        )
        clean_model_name = model_size.split(" ")[0]
        epochs = st.slider("Training Epochs", min_value=1, max_value=100, value=10, step=1)
    with col_s2:
        img_size = st.selectbox("Image Resolution (Pixels)", [320, 416, 640, 1024], index=2)
        batch_size = st.selectbox("Batch Size", [2, 4, 8, 16, 32], index=2)

    uploaded_zip = st.file_uploader("📦 Upload YOLO Dataset (.zip containing images, labels, and data.yaml)", type=["zip"])

    if uploaded_zip is not None:
        temp_dir = tempfile.mkdtemp()
        dataset_dir = os.path.join(temp_dir, "dataset")
        os.makedirs(dataset_dir, exist_ok=True)

        zip_path = os.path.join(temp_dir, "uploaded_dataset.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.getbuffer())

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_dir)

        yaml_path = None
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                if file.endswith(".yaml") or file.endswith(".yml"):
                    yaml_path = os.path.join(root, file)
                    break
            if yaml_path:
                break

        if yaml_path:
            st.success(f"🎯 Dataset configuration locked: `{os.path.basename(yaml_path)}`")
            
            if st.button("🚀 IGNITE MODEL TRAINING NOW!", type="primary", use_container_width=True):
                with st.spinner("🔥 Training neural network in progress... Crunching tensors and optimizing weights!"):
                    model = YOLO(clean_model_name)
                    results = model.train(
                        data=yaml_path,
                        epochs=epochs,
                        imgsz=img_size,
                        batch=batch_size,
                        project=os.path.join(temp_dir, "runs"),
                        name="detect_model",
                        verbose=True
                    )
                    
                    best_weights_path = os.path.join(temp_dir, "runs", "detect_model", "weights", "best.pt")
                    
                    if os.path.exists(best_weights_path):
                        persistent_weights = os.path.join(tempfile.gettempdir(), "best_trained_model.pt")
                        shutil.copy(best_weights_path, persistent_weights)
                        st.session_state.trained_model_path = persistent_weights
                        
                        st.balloons()
                        st.success("🎉 Training complete! Your custom high-performance .pt model is fully primed and ready.")
                        
                        zip_output_path = os.path.join(temp_dir, "trained_model_weights.zip")
                        with zipfile.ZipFile(zip_output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                            zipf.write(persistent_weights, arcname="best.pt")

                        with open(zip_output_path, "rb") as f:
                            bytes_data = f.read()
                            
                        st.download_button(
                            label="📥 DOWNLOAD TRAINED MODEL WEIGHTS (.ZIP)",
                            data=bytes_data,
                            file_name="yolo_energized_model.zip",
                            mime="application/zip",
                            use_container_width=True
                        )
                    else:
                        st.error("⚠️ Training finished, but weights file was not generated.")
        else:
            st.warning("⚠️ Could not find a `data.yaml` configuration file in your zip folder.")

# ==========================================
# TAB 2: TEST ON PICTURES & VIDEOS
# ==========================================
with tab_infer:
    st.markdown("### 🎯 Step 2: High-Speed Vision Inference Studio")
    
    model_option = st.radio("Choose Model Source", ["⚡ Use Recently Trained Model", "📂 Upload Custom .pt File"], horizontal=True)
    
    active_model = None
    if model_option == "⚡ Use Recently Trained Model":
        if st.session_state.trained_model_path and os.path.exists(st.session_state.trained_model_path):
            active_model = YOLO(st.session_state.trained_model_path)
            st.success("⚡ Active model loaded successfully from training session!")
        else:
            st.warning("⚠️ No active trained model found. Train one in Tab 1 or upload a `.pt` file below.")
    else:
        uploaded_pt = st.file_uploader("Upload custom `.pt` weights", type=["pt"])
        if uploaded_pt is not None:
            pt_temp_path = os.path.join(tempfile.gettempdir(), uploaded_pt.name)
            with open(pt_temp_path, "wb") as f:
                f.write(uploaded_pt.getbuffer())
            active_model = YOLO(pt_temp_path)
            st.success(f"🔥 Custom model loaded: {uploaded_pt.name}")

    if active_model is not None:
        st.markdown("---")
        media_type = st.selectbox("Select Target Media Stream", ["🖼️ Picture (Image)", "🎞️ Video Stream"])
        confidence_threshold = st.slider("Detection Confidence Threshold", min_value=0.05, max_value=1.0, value=0.35, step=0.05)

        if media_type == "🖼️ Picture (Image)":
            uploaded_image = st.file_uploader("Upload Image File", type=["jpg", "jpeg", "png"])
            if uploaded_image is not None:
                image = Image.open(uploaded_image)
                st.image(image, caption="Original Input Image", use_container_width=True)
                
                if st.button("🔍 EXECUTE OBJECT DETECTION", type="primary", use_container_width=True):
                    with st.spinner("⚡ Running computer vision inference..."):
                        results = active_model(image, conf=confidence_threshold)
                        res_plotted = results[0].plot()
                        res_image = Image.fromarray(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB))
                        
                        st.image(res_image, caption="🔥 Detected Output with Bounding Boxes", use_container_width=True)

        elif media_type == "🎞️ Video Stream":
            uploaded_video = st.file_uploader("Upload Video File", type=["mp4", "avi", "mov"])
            if uploaded_video is not None:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded_video.read())
                video_path = tfile.name

                st.video(video_path)

                if st.button("🚀 PROCESS & RENDER VIDEO INFERENCE", type="primary", use_container_width=True):
                    output_video_path = os.path.join(tempfile.gettempdir(), "output_detected.mp4")
                    
                    with st.spinner("⚡ Processing video frames in real-time... Hang tight!"):
                        cap = cv2.VideoCapture(video_path)
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        if fps == 0 or np.isnan(fps):
                            fps = 30
                        
                        fourcc = cv2.VideoWriter_fourcc(*'avc1')
                        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

                        progress_bar = st.progress(0)
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        frame_count = 0

                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                break
                            
                            results = active_model(frame, conf=confidence_threshold, verbose=False)
                            annotated_frame = results[0].plot()
                            out.write(annotated_frame)
                            
                            frame_count += 1
                            if total_frames > 0:
                                progress_bar.progress(min(frame_count / total_frames, 1.0))

                        cap.release()
                        out.release()

                    st.success("🎉 Video inference complete!")
                    st.video(output_video_path)
                    
                    with open(output_video_path, "rb") as f:
                        st.download_button(
                            label="📥 DOWNLOAD PROCESSED VIDEO",
                            data=f.read(),
                            file_name="yolo_detected_output.mp4",
                            mime="video/mp4",
                            use_container_width=True
                        )

# ==========================================
# TAB 3: COMPREHENSIVE ENGLISH GUIDE BOOK
# ==========================================
with tab_guide:
    st.markdown("""
        <div style="
            background-color: #FFFDF0; 
            border: 3px solid #F59E0B; 
            border-radius: 16px; 
            padding: 30px; 
            margin-top: 15px;
            color: #000000;
        ">
            <h1 style="color: #B45309; margin-top: 0; font-size: 2.2rem; font-weight: 900;">
                📖 Comprehensive Guide Book: YOLO AI Power Studio
            </h1>
            <p style="color: #000000; font-size: 1.1rem; font-weight: 600;">
                <b>Lead Architect & Developer:</b> Engr. Bilal Mehmood (Certified Generative & Agentic AI Application Developer)
            </p>
            <hr style="border-top: 2px solid #F59E0B; margin: 20px 0;">

            <h3 style="color: #1E3A8A; font-weight: 800;">1. Project Overview & Architecture</h3>
            <p style="color: #000000; font-size: 1.05rem; line-height: 1.7;">
                The <b>YOLO AI Power Studio</b> is a high-performance Streamlit application designed for end-to-end computer vision workflows. 
                It bridges the gap between custom dataset training and real-time media inference without requiring complex command-line scripts. 
                The platform supports multiple generations of Ultralytics YOLO models (YOLOv8, YOLOv9, YOLOv10, and YOLOv11) with automated <code>.pt</code> weight packaging.
            </p>

            <h3 style="color: #1E3A8A; font-weight: 800; margin-top: 25px;">2. Dataset Structure and Preparation</h3>
            <p style="color: #000000; font-size: 1.05rem; line-height: 1.7;">
                To successfully train a custom object detection model, your dataset must strictly adhere to the standard <b>YOLO Directory Format</b> 
                and be compressed into a single <code>.zip</code> archive.
            </p>

            <h4 style="color: #B45309; font-weight: 700;">Required Directory Layout inside the .zip:</h4>
        </div>
    """, unsafe_allow_html=True)

    st.code("""
my_custom_dataset.zip
│
├── data.yaml          # Configuration file defining classes and paths
├── train/
│   ├── images/        # Training pictures (.jpg, .png)
│   └── labels/        # Bounding box text files (.txt matching image names)
└── val/
    ├── images/        # Validation pictures for evaluation
    └── labels/        # Validation bounding box text files
    """, language="text")

    st.markdown("""
        <div style="
            background-color: #FFFDF0; 
            border: 3px solid #F59E0B; 
            border-radius: 16px; 
            padding: 30px; 
            margin-top: 15px;
            color: #000000;
        ">
            <h4 style="color: #B45309; font-weight: 700; margin-top: 0;">Detailed Explanation of Components:</h4>
            <ul style="color: #000000; font-size: 1.05rem; line-height: 1.8;">
                <li><b>data.yaml:</b> Specifies paths to training/validation directories, total class count (<code>nc</code>), and class names list.</li>
                <li><b>Label Files (.txt):</b> Each image must have a corresponding <code>.txt</code> file where every line represents an object: <code>[class_id x_center y_center width height]</code> normalized between 0 and 1.</li>
            </ul>

            <h3 style="color: #1E3A8A; font-weight: 800; margin-top: 25px;">3. Step-by-Step Guide: Training a Custom Model</h3>
            <ul style="color: #000000; font-size: 1.05rem; line-height: 1.8;">
                <li><b>Choose Base Architecture:</b> Select lightweight models (e.g., <code>yolov11n.pt</code>) for speed or heavy models (e.g., <code>yolov11l.pt</code>) for maximum accuracy.</li>
                <li><b>Configure Hyperparameters:</b> Set Epochs (e.g., 10–50+), Image Resolution (320px, 640px, or 1024px), and Batch Size.</li>
                <li><b>Upload & Train:</b> Upload your <code>.zip</code> dataset and click <b>IGNITE MODEL TRAINING NOW</b>.</li>
                <li><b>Download Weights:</b> After training completes, download your <code>best.pt</code> weights in a zip archive.</li>
            </ul>

            <h3 style="color: #1E3A8A; font-weight: 800; margin-top: 25px;">4. Step-by-Step Guide: Inference Studio</h3>
            <ul style="color: #000000; font-size: 1.05rem; line-height: 1.8;">
                <li><b>Model Selection:</b> Load your session-trained model or upload external <code>.pt</code> weights.</li>
                <li><b>Confidence Threshold:</b> Adjust the slider (0.05 to 1.0) to set sensitivity.</li>
                <li><b>Image & Video Inference:</b> Upload media files, execute real-time detection, preview bounding boxes, and download annotated output files.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
