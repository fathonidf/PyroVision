from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
from PIL import Image
import io
import numpy as np
import cv2

app = FastAPI()

# Load model kamu (Pastikan path-nya benar)
model = YOLO("fire_smoke_daffa_yolov11_model.pt")

@app.get("/")
def home():
    return {"status": "AI System Ready"}

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    # 1. Baca gambar dari upload
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    
    # 2. Lakukan Prediksi
    # conf=0.4 agar lebih sensitif sedikit untuk PoC
    results = model.predict(image, conf=0.4)
    result = results[0]
    
    # 3. Cek apakah ada Api/Asap?
    # Anggap class_id 0 = fire, 1 = smoke (sesuaikan dengan data.yaml kamu)
    detected_classes = result.boxes.cls.cpu().numpy().astype(int)
    class_names = result.names
    
    status = "SAFE"
    message = "Aman terkendali"
    
    # Logika Alarm: Jika ada 'fire'
    for cls_id in detected_classes:
        label = class_names[cls_id]
        if "fire" in label.lower(): # Sesuaikan nama kelas di datasetmu
            status = "DANGER"
            message = "KEBAKARAN TERDETEKSI!"
            break
        elif "smoke" in label.lower():
            status = "WARNING"
            message = "Terdeteksi Asap"

    # 4. Gambar Bounding Box (Render)
    # Kita minta Ultralytics gambarin box-nya langsung
    img_array = result.plot() # Ini format BGR (OpenCV standard)
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB) # Ubah ke RGB untuk dikirim balik
    img_pil = Image.fromarray(img_rgb)
    
    # 5. Konversi gambar hasil ke bytes untuk dikirim balik ke API
    img_byte_arr = io.BytesIO()
    img_pil.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    
    # Kita tidak bisa return bytes langsung dengan mudah di JSON campuran, 
    # jadi kita return Response khusus gambar di endpoint lain, 
    # TAPI untuk kemudahan PoC, kita return JSON info saja,
    # dan Frontend yang akan plot manual? 
    # ATAU LEBIH MUDAH: Kita return bytes image menggunakan Response FastAPI.
    
    from fastapi.responses import Response
    
    # Kirim status lewat Header HTTP agar frontend tau harus bunyikan alarm atau tidak
    return Response(content=img_byte_arr.getvalue(), media_type="image/jpeg", headers={
        "X-Detection-Status": status,
        "X-Detection-Message": message
    })

# Cara jalanin: uvicorn backend:app --reload
# ENDPOINT BARU untuk Stream Video
@app.post("/detect_stream")
async def detect_stream(file: UploadFile = File(...)):
    """Endpoint untuk menerima frame per frame dari video stream"""
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    
    # Prediksi dengan confidence threshold
    results = model.predict(image, conf=0.4, verbose=False)
    result = results[0]
    
    # Deteksi status
    detected_classes = result.boxes.cls.cpu().numpy().astype(int) if len(result.boxes) > 0 else []
    class_names = result.names
    
    status = "SAFE"
    message = "Aman terkendali"
    
    for cls_id in detected_classes:
        label = class_names[cls_id]
        if "fire" in label.lower():
            status = "DANGER"
            message = "KEBAKARAN TERDETEKSI!"
            break
        elif "smoke" in label.lower():
            status = "WARNING"
            message = "Terdeteksi Asap"
    
    # Render gambar dengan bounding box
    img_array = result.plot()
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    
    # Convert ke JPEG bytes
    img_byte_arr = io.BytesIO()
    img_pil.save(img_byte_arr, format='JPEG', quality=85)
    img_byte_arr.seek(0)
    
    from fastapi.responses import Response
    
    return Response(
        content=img_byte_arr.getvalue(), 
        media_type="image/jpeg",
        headers={
            "X-Detection-Status": status,
            "X-Detection-Message": message
        }
    )