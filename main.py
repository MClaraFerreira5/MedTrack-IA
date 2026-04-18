from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')

model = YOLO(r'C:\Users\clara\Documents\GitHub\MedTrack-IA\runs\detect\train5\weights\best.pt')

img_path = '20260330_115955.jpg'

results = model.predict(source=img_path, conf=0.5, save=True)

for r in results:
    im_array = r.plot()

    cv2.namedWindow("Resultado", cv2.WINDOW_NORMAL)

    cv2.resizeWindow("Resultado", 800, 600)

    cv2.imshow("Resultado", im_array)
    cv2.waitKey(0)
cv2.destroyAllWindows()