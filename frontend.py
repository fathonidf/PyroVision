import streamlit as st
import requests
from PIL import Image
import io
import cv2
import numpy as np
import time
from datetime import datetime
from streamlit_extras.stylable_container import stylable_container

# Konfigurasi Halaman dengan tema gelap
st.set_page_config(
    page_icon="assets/logo.png",
    page_title="🔥 PyroVision - AI Vision Smoke & Fire Detection",
    layout="wide"
)

st.image("assets/pyrovision_logo.png", width=220)
st.title("PyroVision - AI Vision Smoke & Fire Detection")
st.markdown("Sistem pemantau CCTV cerdas berbasis YOLOv11 dengan deteksi real-time.")

# Custom CSS untuk tampilan modern
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    /* Global Styling */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Background */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    /* Card Styling */
    .stApp > header {
        background-color: transparent;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #7D5A00 0%, #FFD464 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Title Styling */
    h1 {
        background: linear-gradient(135deg, #FFD464 0%, #FFD464 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800;
        font-size: 3.5rem !important;
        text-align: start;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #ffffff;
        font-size: 1.2rem;
        font-weight: 400;
        margin-bottom: 2rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    
    /* Card Container */
    .card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        margin-bottom: 1.5rem;
    }
    
    /* Traffic Light Container */
    .traffic-light {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 20px;
        padding: 15px;
        background: rgba(0, 0, 0, 0.8);
        border-radius: 15px;
        margin-bottom: 15px;
    }
    
    /* Individual Light */
    .light {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        border: 3px solid rgba(255, 255, 255, 0.3);
        transition: all 0.3s ease;
        box-shadow: 0 0 10px rgba(0,0,0,0.3);
    }
    
    .light-red {
        background-color: #4a4a4a;
    }
    
    .light-red.active {
        background-color: #ff3333;
        box-shadow: 0 0 30px #ff3333, 0 0 50px #ff3333;
        animation: pulse-red 1s infinite;
    }
    
    .light-yellow {
        background-color: #4a4a4a;
    }
    
    .light-yellow.active {
        background-color: #ffcc00;
        box-shadow: 0 0 30px #ffcc00, 0 0 50px #ffcc00;
        animation: pulse-yellow 2s infinite;
    }
    
    .light-green {
        background-color: #4a4a4a;
    }
    
    .light-green.active {
        background-color: #00ff00;
        box-shadow: 0 0 30px #00ff00, 0 0 50px #00ff00;
    }
    
    @keyframes pulse-red {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }
    
    @keyframes pulse-yellow {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    /* Card Title */
    .card-title {
        color: #FFD464;
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    /* Status Badge */
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1.1rem;
        margin: 1rem 0;
        text-align: center;
        width: 100%;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .status-safe {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
    }
    
    .status-idle {
        background: #000000);
        color: white;
    }
    
    .status-warning {
        background: linear-gradient(135deg, #f093fb 0%, #fee140 100%);
        color: white;
        animation: pulse 2s infinite;
    }
    
    .status-danger {
        background: linear-gradient(135deg, #fa709a 0%, #f5576c 100%);
        color: white;
        animation: pulse 1s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    /* Info Box */
    .info-box {
        background: linear-gradient(135deg, #000000 0%, #000000 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    /* Metrics Styling */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #FFD464;
    }
    
    /* Button Styling */
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        height: 3rem;
        font-weight: 600;
        font-size: 1rem;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    
    .stButton > button:disabled {
        opacity: 0.5;
        cursor: not-allowed;
        transform: none;
    }
    
    /* Image Container */
    .stImage {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    /* Header Section */
    .header-section {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    /* Stats Container */
    .stats-container {
        display: flex;
        justify-content: space-around;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Header Section

# Sidebar dengan desain modern
st.sidebar.markdown("## ⚙️ Control Panel")
st.sidebar.markdown("---")

# Confidence threshold
confidence = st.sidebar.slider(
    "🎯 Confidence Threshold", 
    0.0, 1.0, 0.4,
    help="Tingkat kepercayaan deteksi (semakin tinggi semakin akurat)"
)

# FPS Control
fps_limit = st.sidebar.slider(
    "⚡ Frame Rate (FPS)", 
    1, 30, 10,
    help="Jumlah frame per detik"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎮 Stream Control")

# Session State
if 'streaming' not in st.session_state:
    st.session_state.streaming = False
if 'start_time' not in st.session_state:
    st.session_state.start_time = None
if 'detection_counter' not in st.session_state:
    st.session_state.detection_counter = 0
if 'current_status' not in st.session_state:
    st.session_state.current_status = "SAFE"
if 'current_message' not in st.session_state:
    st.session_state.current_message = "Aman terkendali"

# Tombol kontrol streaming dengan warna berbeda
col_btn1, col_btn2 = st.sidebar.columns(2)
with col_btn1:
    with stylable_container(
            key="pv_start_btn",
            css_styles="""
                .stButton > button { background:#FFCF3E !important; color:#000 !important; }
                .stButton > button:hover { background:#000 !important; color: #fff !important; }
            """
        ):
            start_stream = st.button("Start Stream", key="start_stream", use_container_width=True)
with col_btn2:
    with stylable_container(
            key="pv_stop_btn",
            css_styles="""
                .stButton > button { background:#000 !important; color:#fff; !important; }
                .stButton > button:hover { background:#000 !important; color:#fff !important;}
            """
        ):
            stop_stream = st.button("Stop Stream", key="stop_stream", use_container_width=True)

st.sidebar.markdown("---")

# Info Box di Sidebar
st.sidebar.markdown("""
    <div class="info-box">
        <h4>📊 Status Deteksi</h4>
        <p><strong>🟢 SAFE:</strong> Tidak ada ancaman</p>
        <p><strong>🟡 WARNING:</strong> Asap terdeteksi</p>
        <p><strong>🔴 DANGER:</strong> Api terdeteksi!</p>
    </div>
""", unsafe_allow_html=True)

# Statistik
fps_info = st.sidebar.empty()
uptime_info = st.sidebar.empty()
detection_count = st.sidebar.empty()

if start_stream:
    st.session_state.streaming = True
    st.session_state.start_time = time.time()
    st.session_state.detection_counter = 0
    st.session_state.current_status = "SAFE"
    st.session_state.current_message = "Aman terkendali"
    st.rerun()
    
if stop_stream:
    st.session_state.streaming = False
    st.session_state.current_status = "SAFE"
    st.session_state.current_message = "Aman terkendali"
    st.rerun()

# Fungsi untuk render traffic light
def render_traffic_lights(status):
    red_active = "active" if status == "DANGER" else ""
    yellow_active = "active" if status == "WARNING" else ""
    green_active = "active" if status == "SAFE" else ""
    
    html = f"""
        <div class="traffic-light">
            <div class="light light-red {red_active}"></div>
            <div class="light light-yellow {yellow_active}"></div>
            <div class="light light-green {green_active}"></div>
        </div>
    """
    return html

# Layout 2 Kolom dengan Card
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="card-title">📹 Live Camera Feed</div>', unsafe_allow_html=True)
    traffic_light_cam = st.empty()
    camera_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card-title">🤖 AI Detection Result</div>', unsafe_allow_html=True)
    traffic_light_ai = st.empty()
    result_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

# Status Display yang lebih menonjol
status_placeholder = st.empty()

# Fungsi Streaming
def process_stream():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("❌ Tidak dapat mengakses kamera! Pastikan kamera terhubung.")
        return
    
    frame_delay = 1.0 / fps_limit
    
    while st.session_state.streaming:
        start_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Gagal membaca frame dari kamera")
            break
        
        # Update traffic lights PERTAMA
        traffic_light_cam.markdown(render_traffic_lights(st.session_state.current_status), unsafe_allow_html=True)
        traffic_light_ai.markdown(render_traffic_lights(st.session_state.current_status), unsafe_allow_html=True)
        
        # Tampilkan frame asli
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        camera_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        
        # Encode frame
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = io.BytesIO(buffer)
        
        try:
            # Kirim ke backend
            files = {"file": ("frame.jpg", frame_bytes, "image/jpeg")}
            response = requests.post("http://127.0.0.1:8000/detect_stream", files=files, timeout=2)
            
            if response.status_code == 200:
                detection_status = response.headers.get("X-Detection-Status", "SAFE")
                detection_message = response.headers.get("X-Detection-Message", "Aman")
                
                # Update status global
                st.session_state.current_status = detection_status
                st.session_state.current_message = detection_message
                
                # Tampilkan hasil deteksi
                result_image = Image.open(io.BytesIO(response.content))
                result_placeholder.image(result_image, use_container_width=True)
                
                # Update status dengan styling modern
                if detection_status == "DANGER":
                    st.session_state.detection_counter += 1
                    status_placeholder.markdown(f"""
                        <div class="status-badge status-danger">
                            🚨 {detection_message} 🚨
                        </div>
                    """, unsafe_allow_html=True)
                elif detection_status == "WARNING":
                    st.session_state.detection_counter += 1
                    status_placeholder.markdown(f"""
                        <div class="status-badge status-warning">
                            ⚠️ {detection_message}
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    status_placeholder.markdown(f"""
                        <div class="status-badge status-safe">
                            ✅ {detection_message}
                        </div>
                    """, unsafe_allow_html=True)
            
        except Exception as e:
            status_placeholder.error(f"🔌 Connection Error: {str(e)[:100]}")
        
        # Update statistik
        process_time = time.time() - start_time
        actual_fps = 1 / process_time if process_time > 0 else 0
        fps_info.metric("⚡ FPS", f"{actual_fps:.1f}")
        
        if st.session_state.start_time:
            uptime = int(time.time() - st.session_state.start_time)
            uptime_info.metric("⏱️ Uptime", f"{uptime}s")
        
        detection_count.metric("🎯 Detections", st.session_state.detection_counter)
        
        # Delay untuk FPS
        if process_time < frame_delay:
            time.sleep(frame_delay - process_time)
    
    cap.release()
    status_placeholder.info("⏸️ Stream dihentikan")

# Jalankan streaming
if st.session_state.streaming:
    process_stream()
else:
    # Tampilkan traffic lights standby (hijau)
    traffic_light_cam.markdown(render_traffic_lights("SAFE"), unsafe_allow_html=True)
    traffic_light_ai.markdown(render_traffic_lights("SAFE"), unsafe_allow_html=True)
    
    # Placeholder saat tidak streaming
    camera_placeholder.image("https://via.placeholder.com/640x480/667eea/ffffff?text=Press+START+to+begin", use_container_width=True)
    result_placeholder.image("https://via.placeholder.com/640x480/764ba2/ffffff?text=AI+Detection+Ready", use_container_width=True)
    
    status_placeholder.markdown("""
        <div class="status-badge status-idle">
            💤 System Standby - Press START to activate monitoring
        </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: white; padding: 1rem;'>
        <p>🔥 <strong>PyroVision AI</strong> - Powered by YOLOv11 & FastAPI</p>
        <p style='font-size: 0.9rem; opacity: 0.8;'>Deteksi kebakaran real-time untuk keamanan maksimal 🛡️</p>
    </div>
""", unsafe_allow_html=True)