import torch
import cv2
from ultralytics import YOLO
import os


model = YOLO('yolov8n.pt')
def detect_crop(img_path):
    img = cv2.imread(img_path)

    if img is None:
        print("Erro: Imagem nao encontrada")
        return None

    h_orig, w_orig, _ = img.shape
    results = model.predict(img, conf=0.2)
    biggest = 0
    better_box = None

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (y2 - y1) * (x2 - x1)
            if area > biggest:
                biggest = area
                better_box = (x1, y1, x2, y2)

    if better_box:
        x1, y1, x2, y2 = better_box

        largura_box = x2 - x1
        altura_box = y2 - y1

        padding_w = int(largura_box)
        padding_h = int(altura_box * 0.2)

        nx1 = max(0, x1 - padding_w)
        ny1 = max(0, y1 - padding_h)
        nx2 = min(w_orig, x2 + padding_w)
        ny2 = min(h_orig, y2 + padding_h)

        crop_img = img[ny1:ny2, nx1:nx2]
        cv2.imwrite('croped_medicine.jpg', crop_img)
        print("Sucesso! Recorte salvo como 'croped_medicine.jpg'")
        # Desenha o que o YOLO achou (Vermelho)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        # Desenha o novo corte com margem (Verde)
        cv2.rectangle(img, (nx1, ny1), (nx2, ny2), (0, 255, 0), 3)
        cv2.imwrite('debug_deteccao.jpg', img)
        return crop_img

    print("Erro: Nenhum Objeto detectado pelo YOLO")
    return None

detect_crop('data/Puran T4 - 50mcg/20260329_110141.jpg')