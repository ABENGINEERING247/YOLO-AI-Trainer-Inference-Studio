import streamlit as st
import os
import zipfile
import tempfile
import cv2
from PIL import Image
import numpy as np
from ultralytics import YOLO

# --- Page Configuration ---
st.set_page_config(
    page_title="YOLO AI Trainer & Inference Studio",
    page_icon="🤖",
    layout="wide"
)

# --- Custom Styling ---
st.markdown("""
    <style>
    .main-header {
        font-size: 2.4rem;
        color: #D97706;
        font-weight: 800;
        margin-bottom: 0rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🤖 YOLO Model Trainer & Media Inference Studio</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Train custom YOLO .pt models from datasets, and test them instantly on pictures and videos.</p>', unsafe_allow_html=True)

# --- Session State Management ---
if "trained_model_path" not in st.session_state:
    st.session_state.trained_model_path = None

# Create Tabs for Navigation
tab_train, tab_infer = st.tabs(["🚀 Train Model (.pt)", "🎥 Test on Pictures & Videos"])

# ==========================================
# TAB 1: TRAIN MODEL
# ==========================================
with tab_train:
    st.markdown("### Step 1: Upload Dataset & Configure Training")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        model_size = st.selectbox("Base YOLO Model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"], index=0)
        epochs = st.slider("Training Epochs", min_value=1, max_value=50, value=5, step=1)
    with col_s2:
        img_size = st.selectbox("Image Resolution", [320, 416, 640], index=2)
        batch_size = st.selectbox("Batch Size", [2, 4, 8, 16], index=1)

    uploaded_zip = st.file_uploader("Upload YOLO Dataset (.zip containing images, labels, and data.yaml)", type=["zip"])

    if uploaded_zip is not None:
        temp_dir = tempfile.mkdtemp()
        dataset_dir = os.path.join(temp_dir, "dataset")
        os.makedirs(dataset_dir, exist_ok=True)

        zip_path = os.path.join(temp_dir, "uploaded_dataset.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.getbuffer())

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_dir)

        # Locate data.yaml
        yaml_path = None
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                if file.endswith(".yaml") or file.endswith(".yml"):
                    yaml_path = os.path.join(root, file)
                    break
            if yaml_path:
                break

        if yaml_path:
            st.info(f"Dataset configuration located: `{os.path.basename(yaml_path)}`")
            
            if st.button("🚀 Start Model Training", type="primary"):
                with st.spinner("Training model... Please wait while epochs complete."):
                    model = YOLO(model_size)
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
                        # Save to persistent session state or persistent temp file
                        persistent_weights = os.path.join(tempfile.gettempdir(), "best_trained_model.pt")
                        import shutil
                        shutil.copy(best_weights_path, persistent_weights)
                        st.session_state.trained_model_path = persistent_weights
                        
                        st.success("Training completed successfully! Your .pt model is ready.")
                        
                        # Package into zip for download
                        zip_output_path = os.path.join(temp_dir, "trained_model_weights.zip")
                        with zipfile.ZipFile(zip_output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                            zipf.write(persistent_weights, arcname="best.pt")

                        with open(zip_output_path, "rb") as f:
                            bytes_data = f.read()
                            
                        st.download_button(
                            label="📦 Download Trained Model Weights (.zip)",
                            data=bytes_data,
                            file_name="yolo_trained_model.zip",
                            mime="application/zip",
                            use_container_width=True
                        )
                    else:
                        st.error("Training completed, but weights file was not generated.")
        else:
            st.warning("Could not find a `data.yaml` configuration file in your zip folder.")

# ==========================================
# TAB 2: TEST ON PICTURES & VIDEOS
# ==========================================
with tab_infer:
    st.markdown("### Step 2: Test Your Trained Model on Media")
    
    # Model Source Selection
    model_option = st.radio("Select Model Source", ["Use Recently Trained Model from Tab 1", "Upload Custom .pt File"])
    
    active_model = None
    if model_option == "Use Recently Trained Model from Tab 1":
        if st.session_state.trained_model_path and os.path.exists(st.session_state.trained_model_path):
            active_model = YOLO(st.session_state.trained_model_path)
            st.success("Loaded your recently trained model!")
        else:
            st.warning("No trained model found in session. Please train a model in Tab 1 or upload a `.pt` file below.")
    else:
        uploaded_pt = st.file_uploader("Upload your `.pt` model file", type=["pt"])
        if uploaded_pt is not None:
            pt_temp_path = os.path.join(tempfile.gettempdir(), uploaded_pt.name)
            with open(pt_temp_path, "wb") as f:
                f.write(uploaded_pt.getbuffer())
            active_model = YOLO(pt_temp_path)
            st.success(f"Loaded custom model: {uploaded_pt.name}")

    if active_model is not None:
        st.markdown("---")
        media_type = st.selectbox("Select Media Type for Detection", ["Picture (Image)", "Video"])
        confidence_threshold = st.slider("Confidence Threshold", min_value=0.1, max_value=1.0, value=0.25, step=0.05)

        if media_type == "Picture (Image)":
            uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
            if uploaded_image is not None:
                image = Image.open(uploaded_image)
                st.image(image, caption="Original Image", use_container_width=True)
                
                if st.button("🔍 Run Detection on Image", type="primary"):
                    with st.spinner("Processing image..."):
                        results = active_model(image, conf=confidence_threshold)
                        res_plotted = results[0].plot() # BGR format numpy array
                        res_image = Image.fromarray(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB))
                        
                        st.image(res_image, caption="Detected Output", use_container_width=True)

        elif media_type == "Video":
            uploaded_video = st.file_uploader("Upload a Video File", type=["mp4", "avi", "mov"])
            if uploaded_video is not None:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded_video.read())
                video_path = tfile.name

                st.video(video_path)

                if st.button("🚀 Process & Run Detection on Video", type="primary"):
                    output_video_path = os.path.join(tempfile.gettempdir(), "output_detected.mp4")
                    
                    with st.spinner("Processing video frames... This may take a moment."):
                        cap = cv2.VideoCapture(video_path)
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        if fps == 0 or np.isnan(fps):
                            fps = 30
                        
                        fourcc = cv2.VideoWriter_fourcc(*'avc1') # H.264 compatible codec for web
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

                    st.success("Video processing complete!")
                    st.video(output_video_path)
                    
                    with open(output_video_path, "rb") as f:
                        st.download_button(
                            label="📥 Download Processed Video",
                            data=f.read(),
                            file_name="detected_output_video.mp4",
                            mime="video/mp4",
                            use_container_width=True
                        )
