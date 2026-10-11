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

if "show_welcome_screen" not in st.session_state:
    st.session_state.show_welcome_screen = True

# --- BIG CENTERED WELCOME SCREEN OVERLAY ---
if st.session_state.show_welcome_screen:
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    st.markdown("""
        <div style="
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 60px 40px;
            background: linear-gradient(135deg, #047857 0%, #059669 50%, #10B981 100%);
            border: 5px solid #34D399;
            border-radius: 30px;
            text-align: center;
            margin: 0 auto;
            max-width: 950px;
            box-shadow: 0 25px 60px rgba(16, 185, 129, 0.6);
        ">
            <h1 style="color: #FFFFFF !important; font-size: 3.2rem; font-weight: 900; text-transform: uppercase; margin-bottom: 25px; line-height: 1.2; text-shadow: 0 4px 10px rgba(0,0,0,0.3);">
                🚀 AI BASED DETECTION SYSTEM BNAANA HUA AB BILKUL ASAAN! 🔥
            </h1>
            <p style="color: #ECFDF5 !important; font-size: 1.5rem; font-weight: 800; max-width: 800px; margin-bottom: 10px; line-height: 1.5;">
                Welcome to your ultimate high-performance Ultralytics YOLO training and real-time computer vision inference studio.
            </p>
            <p style="color: #FEF3C7 !important; font-size: 1.2rem; font-weight: 700; margin-top: 0;">
                Architected by <b>Engr. Bilal Mehmood</b>
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_w1, col_w2, col_w3 = st.columns([1, 2, 1])
    with col_w2:
        if st.button("✨ ENTER STUDIO NOW 🔥", use_container_width=True, type="primary"):
            st.session_state.show_welcome_screen = False
            st.rerun()

else:
    # --- TOP HEADER BANNER ---
    st.markdown("""
        <div class="app-header-banner">
            <h1>🔥 ULTRALYTICS YOLO AI POWER STUDIO 🔥</h1>
            <p>Next-Gen Model Training & Real-Time Computer Vision Inference | Architected by <b>Engr. Bilal Mehmood</b></p>
        </div>
    """, unsafe_allow_html=True)

    # Create Tabs for Navigation
    tab_train, tab_infer = st.tabs(["🚀 Train Custom YOLO Model", "🎥 High-Speed Media Inference Studio"])

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
