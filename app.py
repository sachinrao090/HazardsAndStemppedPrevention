import streamlit as st
import cv2
from ultralytics import YOLO
import datetime
import pandas as pd
import time

# ==========================================
# 1. PAGE CONFIGURATION & STATE MANAGEMENT
# ==========================================
st.set_page_config(
    page_title="HSP Control Center", 
    page_icon="🛡️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize persistent session state for analytics
if 'log_df' not in st.session_state:
    st.session_state.log_df = pd.DataFrame(columns=["Timestamp", "Hazard Type", "Confidence", "Status"])
if 'session_start' not in st.session_state:
    st.session_state.session_start = time.time()

# ==========================================
# 2. CORE FUNCTIONS (MODULAR DESIGN)
# ==========================================
@st.cache_resource
def load_vision_model(weights_path: str):
    """Loads the YOLOv8 model and caches it in GPU memory to prevent reloading."""
    try:
        model = YOLO(weights_path)
        return model
    except Exception as e:
        st.error(f"Failed to load model architecture: {e}")
        return None

def process_frame(frame, model, conf_threshold: float):
    """Executes inference and returns the annotated frame alongside detection telemetry."""
    start_time = time.time()
    
    # Run inference on GPU (device=0)
    results = model.predict(frame, conf=conf_threshold, device=0, verbose=False)
    annotated_frame = results[0].plot()
    
    # Calculate FPS
    fps = 1.0 / (time.time() - start_time)
    
    # Extract detection telemetry
    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id]
        detections.append({"label": label, "conf": conf})
        
    return annotated_frame, detections, fps

def log_hazard(label: str, conf: float):
    """Appends critical safety violations to the session state DataFrame."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_entry = pd.DataFrame([{
        "Timestamp": timestamp, 
        "Hazard Type": label, 
        "Confidence": f"{conf:.2f}",
        "Status": "UNRESOLVED"
    }])
    st.session_state.log_df = pd.concat([new_entry, st.session_state.log_df], ignore_index=True)

# ==========================================
# 3. SIDEBAR NAVIGATION & CONTROLS
# ==========================================
with st.sidebar:
    # Now it just looks inside the current project folder!
    st.image("attachment_159171845.png", use_container_width=True) 
    st.title("System Controls")
    st.markdown("---")
    
    engine_choice = st.radio("Core Engine Selection", ["PyTorch (.pt)", "TensorRT (.engine)"], index=0)
    model_path = "runs/detect/Unified_Safety_Model/weights/best.pt" if "pt" in engine_choice else "runs/detect/Unified_Safety_Model/weights/best.engine"
    
    st.markdown("### Vision Parameters")
    conf_thresh = st.slider("Confidence Threshold (Filtering)", 0.05, 0.95, 0.25, step=0.05)
    
    st.markdown("---")
    is_streaming = st.toggle("🎥 Initialize Camera Feed", value=False)
    
    st.markdown("---")
    st.caption("Hardware: NVIDIA RTX Series")
    st.caption("Environment: CUDA Accelerated")

# Load the core model
vision_model = load_vision_model(model_path)

# ==========================================
# 4. MAIN DASHBOARD UI (TABS)
# ==========================================
st.title("🛡️ Hazards and Stepped Prevention (HSP)")
st.markdown("Real-Time Unified Industrial & Public Safety Monitoring Station")

tab_live, tab_analytics, tab_architecture = st.tabs([
    "🔴 Live Monitoring Station", 
    "📊 Analytics & Hazard Logs", 
    "🧠 System Architecture & Motive"
])

# ------------------------------------------
# TAB 1: LIVE MONITORING
# ------------------------------------------
with tab_live:
    col_video, col_metrics = st.columns([3, 1])
    
    with col_metrics:
        st.subheader("System Telemetry")
        metric_fps = st.empty()
        metric_hazards = st.empty()
        metric_active = st.empty()
        
        st.markdown("---")
        st.markdown("**Recent Critical Alerts**")
        alert_container = st.empty()

    with col_video:
        video_placeholder = st.empty()
        
        if is_streaming and vision_model:
            cap = cv2.VideoCapture(0)
            
            while cap.isOpened() and is_streaming:
                ret, frame = cap.read()
                if not ret:
                    st.error("Camera connection lost.")
                    break
                
                # Process the frame
                annotated, detections, current_fps = process_frame(frame, vision_model, conf_thresh)
                
                # Update Video Stream
                rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                video_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)
                
                # Analyze detections
                active_objects = len(detections)
                hazard_count_frame = 0
                
                for det in detections:
                    if "NO-" in det["label"]:
                        hazard_count_frame += 1
                        # Prevent log spam: only log if we haven't logged recently or it's a new instance
                        if len(st.session_state.log_df) == 0 or (datetime.datetime.now() - pd.to_datetime(st.session_state.log_df.iloc[0]["Timestamp"])).total_seconds() > 2:
                            log_hazard(det["label"], det["conf"])
                
                # Update Metrics Live
                metric_fps.metric("Inference Speed", f"{current_fps:.1f} FPS")
                metric_active.metric("Tracked Entities", active_objects)
                metric_hazards.metric("Total Violations Logged", len(st.session_state.log_df))
                
                # Update Alert Banner
                if len(st.session_state.log_df) > 0:
                    latest_alert = st.session_state.log_df.iloc[0]
                    alert_container.error(f"🚨 {latest_alert['Timestamp']} | {latest_alert['Hazard Type']}")
                else:
                    alert_container.success("✅ Secure. No active hazards.")
                    
            cap.release()
        elif not is_streaming:
            video_placeholder.info("Camera feed is currently offline. Toggle 'Initialize Camera Feed' in the sidebar to begin monitoring.")

# ------------------------------------------
# TAB 2: ANALYTICS & LOGS
# ------------------------------------------
with tab_analytics:
    st.subheader("Incident History")
    
    if len(st.session_state.log_df) > 0:
        col_chart, col_data = st.columns([1, 2])
        
        with col_chart:
            st.write("**Violations by Type**")
            hazard_counts = st.session_state.log_df["Hazard Type"].value_counts()
            st.bar_chart(hazard_counts)
            
        with col_data:
            st.dataframe(st.session_state.log_df, use_container_width=True)
            
            # Export functionality
            csv = st.session_state.log_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Official Incident Report (CSV)",
                data=csv,
                file_name=f"HSP_Incident_Report_{datetime.date.today()}.csv",
                mime="text/csv",
            )
    else:
        st.success("The hazard database is currently clean. No incidents recorded.")

# ------------------------------------------
# TAB 3: ARCHITECTURE & MOTIVES
# ------------------------------------------
with tab_architecture:
    st.markdown("""
    ## The Motive: "Hazards and Stepped Prevention"
    Industrial sites and crowded public spaces share a common critical failure point: **human monitoring fatigue**. Traditional CCTV systems require continuous human observation, leading to delayed responses to Personal Protective Equipment (PPE) violations or heavy machinery proximity hazards.
    
    This project was developed at **Gautam Buddha University** to transition safety protocols from reactive to **proactive real-time prevention**.
    
    ### System Architecture
    This application is powered by a custom-trained **YOLOv8** Convolutional Neural Network. Rather than relying on separate models for different domains, this engine utilizes a heavily engineered dataset comprising **21 Unified Classes**.
    
    **The Three Pillars of the Unified Model:**
    1. **Industrial Machinery Logic:** Tracking complex, high-density objects like `EXCAVATOR`, `dump truck`, and `wheel loader` to prevent struck-by accidents.
    2. **PPE Compliance Tracking:** High-precision bounding boxes for `Hardhat` and `Safety Vest`, accompanied by logical inversion classes (`NO-Hardhat`) designed specifically to trigger automated alarms.
    3. **Public Health & Crowd Dynamics:** Integration of `Mask` and `NO-Mask` data, preparing the system for versatile deployment across both construction zones and densely populated indoor facilities.
    
    ### Hardware Optimization
    To achieve the real-time inference speeds (FPS) required for life-safety systems, the training and deployment pipeline is highly optimized for local Edge computing, bypassing cloud-latency completely. The model is engineered to run seamlessly on consumer-grade NVIDIA architectures (such as the **RTX 5060**), utilizing CUDA acceleration and optional **TensorRT layer fusion** for production deployment.
    """)