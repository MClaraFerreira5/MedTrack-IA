import cv2
import easyocr
import uvicorn
from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import numpy as np

reader = easyocr.Reader(['en', 'pt'], gpu=True)
model = YOLO(r'C:\Users\clara\Documents\GitHub\MedTrack-IA\runs\detect\train5\weights\best.pt')

app = FastAPI(title="MedTrack AI - OCR API")


@app.post("/detect")
async def extract_medicine_info(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model.predict(img, conf=0.5)

    data_extracted = {
        "status": "success",
        "data": {},
        "count": 0
    }

    for item in results:
        for box in item.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label_name = model.names[int(box.cls[0])]

            roi = img[y1:y2, x1:x2]
            if roi.size > 0:
                result_ocr = reader.readtext(roi, detail=0, paragraph=True)
                text = " ".join(result_ocr).strip()
                data_extracted["data"][label_name] = text
                data_extracted["count"] += 1

    return data_extracted


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)