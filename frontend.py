import streamlit as st
import requests
from PIL import Image
import io
import cv2
import time
from streamlit_extras.stylable_container import stylable_container

st.set_page_config(
    page_icon="assets/logo.png",
    page_title="🔥 PyroVision - AI Vision Smoke & Fire Detection",
    layout="wide"
)

st.image("assets/pyrovision_logo.png", width=220)
st.title("PyroVision - AI Vision Smoke & Fire Detection")
st.markdown("Sistem pemantau CCTV cerdas berbasis YOLOv11 dengan deteksi real-time.")

# session state
if "streaming" not in st.session_state:
    st.session_state.streaming = False

# =========================
# MAIN LAYOUT (Camera first)
# =========================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📡 Live Camera Input")
    camera_placeholder = st.empty()

with col2:
    st.subheader("🖥️ AI Monitoring Result")
    result_placeholder = st.empty()
    status_placeholder = st.empty()

fps_box = st.empty()

# =========================
# CONTROL PANEL (white background) BELOW camera/result
# =========================
from streamlit_extras.stylable_container import stylable_container

# =========================
# CONTROL PANEL (CLEAN)
# =========================
with stylable_container(
    key="pv_control_card",
    css_styles="""
        {
            background: #FFFBEC;
            border-radius: 22px;
            padding: 22px;
            box-shadow: 0 18px 45px rgba(0,0,0,0.15);
            border: 1px solid rgba(0,0,0,0.06);
        }

        /* Ensure rows/columns have spacing and don't collide */
        [data-testid="stHorizontalBlock"] { gap: 18px; }
        [data-testid="column"] { padding-top: 0px; }

        /* Buttons */
        .stButton { margin-bottom: 0px; }
        .stButton > button {
            width: 100%;
            height: 64px;
            border-radius: 14px;
            font-size: 18px;
            font-weight: 700;
            border: none !important;
        }

        /* IMPORTANT: add spacing before sliders */
        .pv-spacer { height: 18px; }

        /* Slider label typography */
        .pv-label {
            font-weight: 700;
            margin: 0 0 8px 0;
            line-height: 1.2;
        }

        /* Reduce slider internal padding so it stays compact */
        [data-testid="stSlider"] {
            padding-top: 0px !important;
            padding-bottom: 0px !important;
        }
        
        /* Slider base */
        [data-testid="stSlider"] {
            --slider-active: #000000;
            --slider-inactive: #e5e7eb;
        }

        /* Active track */
        [data-testid="stSlider"] div[role="slider"] {
            background: var(--slider-active) !important;
            border-color: var(--slider-active) !important;
        }

        /* Filled bar */
        [data-testid="stSlider"] div[role="slider"] ~ div {
            background: var(--slider-active) !important;
        }

        /* Inactive bar */
        [data-testid="stSlider"] > div > div > div {
            background: var(--slider-inactive) !important;
        }

        /* Value */
        [data-testid="stSlider"] span {
            color: var(--slider-active) !important;
            font-weight: 700;
        }
            """
):
    # --- Row 1: Buttons (ONLY buttons here)
    b1, b2 = st.columns(2, gap="large")

    with b1:
        with stylable_container(
            key="pv_start_btn",
            css_styles="""
                .stButton > button { background:#FFCF3E !important; color:#111 !important; }
                .stButton > button:hover { background:#000 !important; color: #fff !important; }
            """
        ):
            start_stream = st.button("Start Stream", key="start_stream", use_container_width=True)

    with b2:
        with stylable_container(
            key="pv_stop_btn",
            css_styles="""
                .stButton > button { background:#000 !important; color:#fff !important; }
                .stButton > button:hover { background:#000 !important; color:#fff !important;}
            """
        ):
            stop_stream = st.button("Stop Stream", key="stop_stream", use_container_width=True)

    # --- Spacer between rows (THIS prevents overlap)
    st.markdown('<div class="pv-spacer"></div>', unsafe_allow_html=True)

    # --- Row 2: Sliders (labels + sliders)
    s1, s2 = st.columns(2, gap="large")

    with s1:
        st.markdown('<div class="pv-label" style="color:#000000; margin-top:20px;">Confidence Threshold</div>', unsafe_allow_html=True)
        confidence = st.slider(
            "confidence_hidden",
            0.0, 1.0, 0.4,
            key="conf",
            label_visibility="collapsed"
        )

    with s2:
        st.markdown('<div class="pv-label" style="color:#000000; margin-top:20px;">FPS (Frame per Second)</div>', unsafe_allow_html=True)
        fps_limit = st.slider(
            "fps_hidden",
            1, 30, 10,
            key="fps",
            label_visibility="collapsed"
        )

    st.caption("Control Panel")


# update streaming state
if start_stream:
    st.session_state.streaming = True
if stop_stream:
    st.session_state.streaming = False



# =========================
# Streaming function
# =========================
def process_stream():
    cap = cv2.VideoCapture(0)  # 0 = default webcam
    if not cap.isOpened():
        st.error("Tidak dapat mengakses kamera!")
        return

    frame_delay = 1.0 / max(1, int(fps_limit))

    while st.session_state.streaming:
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            st.warning("Gagal membaca frame dari kamera")
            break

        # Show original frame (left)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        camera_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        # Encode JPEG for backend
        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            status_placeholder.error("Gagal encode frame")
            continue

        frame_bytes = io.BytesIO(buffer.tobytes())

        try:
            files = {"file": ("frame.jpg", frame_bytes, "image/jpeg")}
            response = requests.post(
                "http://127.0.0.1:8000/detect_stream",
                files=files,
                timeout=2
            )

            if response.status_code == 200:
                detection_status = response.headers.get("X-Detection-Status", "SAFE")
                detection_message = response.headers.get("X-Detection-Message", "Aman")

                result_image = Image.open(io.BytesIO(response.content))
                result_placeholder.image(result_image, use_container_width=True)

                if detection_status == "DANGER":
                    status_placeholder.error(f"🚨 {detection_message} 🚨")
                elif detection_status == "WARNING":
                    status_placeholder.warning(f"⚠️ {detection_message}")
                else:
                    status_placeholder.success(f"✅ {detection_message}")
            else:
                status_placeholder.error(f"Backend error: {response.status_code}")

        except Exception as e:
            status_placeholder.error(f"Connection Error: {str(e)[:120]}")

        # FPS actual
        process_time = time.time() - start_time
        actual_fps = (1 / process_time) if process_time > 0 else 0
        fps_box.metric("FPS Aktual", f"{actual_fps:.1f}")

        # Limit FPS
        if process_time < frame_delay:
            time.sleep(frame_delay - process_time)

    cap.release()
    status_placeholder.info("Stream dihentikan")

# =========================
# Run
# =========================
if st.session_state.streaming:
    process_stream()
else:
    camera_placeholder.image(
        "https://via.placeholder.com/640x480?text=Press+Start+Stream",
        use_container_width=True
    )
    result_placeholder.image(
        "https://via.placeholder.com/640x480?text=AI+Detection+Result",
        use_container_width=True
    )
