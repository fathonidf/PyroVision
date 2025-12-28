import streamlit as st
import requests
from PIL import Image
import io
import cv2
import numpy as np
import time

# Konfigurasi Halaman
st.set_page_config(page_title="🔥 PyroVision - AI Vision Smoke & Fire Detection", layout="wide")

st.title("🔥 PyroVision - AI Vision Smoke & Fire Detection")
st.markdown("Sistem pemantau CCTV cerdas berbasis YOLOv11 dengan deteksi real-time.")

# Sidebar
st.sidebar.header("Kontrol Panel")
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.4)
fps_limit = st.sidebar.slider("FPS (Frame per Second)", 1, 30, 10)

# Tombol kontrol streaming
col_btn1, col_btn2 = st.sidebar.columns(2)
start_stream = col_btn1.button("▶️ Start Stream")
stop_stream = col_btn2.button("⏹️ Stop Stream")

# Layout 2 Kolom
col1, col2 = st.columns(2)

with col1:
    st.subheader("📡 Live Camera Input")
    camera_placeholder = st.empty()

with col2:
    st.subheader("🖥️ AI Monitoring Result")
    result_placeholder = st.empty()
    status_placeholder = st.empty()

# Info FPS
fps_info = st.sidebar.empty()

# Session State untuk kontrol streaming
if 'streaming' not in st.session_state:
    st.session_state.streaming = False

if start_stream:
    st.session_state.streaming = True
    
if stop_stream:
    st.session_state.streaming = False

# Fungsi Streaming
def process_stream():
    cap = cv2.VideoCapture(0)  # 0 untuk webcam default
    
    if not cap.isOpened():
        st.error("Tidak dapat mengakses kamera!")
        return
    
    frame_delay = 1.0 / fps_limit
    
    while st.session_state.streaming:
        start_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            st.warning("Gagal membaca frame dari kamera")
            break
        
        # Tampilkan frame asli di kolom kiri
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        camera_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        
        # Encode frame ke JPEG untuk dikirim ke backend
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = io.BytesIO(buffer)
        
        try:
            # Kirim ke backend
            files = {"file": ("frame.jpg", frame_bytes, "image/jpeg")}
            response = requests.post("http://127.0.0.1:8000/detect_stream", files=files, timeout=2)
            
            if response.status_code == 200:
                # Baca hasil deteksi
                detection_status = response.headers.get("X-Detection-Status", "SAFE")
                detection_message = response.headers.get("X-Detection-Message", "Aman")
                
                # Tampilkan gambar hasil dengan bounding box
                result_image = Image.open(io.BytesIO(response.content))
                result_placeholder.image(result_image, use_container_width=True)
                
                # Update status
                if detection_status == "DANGER":
                    status_placeholder.error(f"🚨 {detection_message} 🚨")
                elif detection_status == "WARNING":
                    status_placeholder.warning(f"⚠️ {detection_message}")
                else:
                    status_placeholder.success(f"✅ {detection_message}")
            
        except Exception as e:
            status_placeholder.error(f"Connection Error: {str(e)[:50]}")
        
        # Hitung FPS aktual
        process_time = time.time() - start_time
        actual_fps = 1 / process_time if process_time > 0 else 0
        fps_info.metric("FPS Aktual", f"{actual_fps:.1f}")
        
        # Delay untuk mengatur FPS
        if process_time < frame_delay:
            time.sleep(frame_delay - process_time)
    
    cap.release()
    status_placeholder.info("Stream dihentikan")

# Jalankan streaming jika aktif
if st.session_state.streaming:
    process_stream()
else:
    camera_placeholder.image("https://via.placeholder.com/640x480?text=Press+Start+Stream", use_container_width=True)
    result_placeholder.image("https://via.placeholder.com/640x480?text=AI+Detection+Result", use_container_width=True)