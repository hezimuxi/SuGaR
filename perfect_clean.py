import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import argparse

def aggressive_cloud_cleaner(roi_bgr):
    """
    更强力地滤除白云，同时利用边缘检测保护铁塔结构。
    """
    hls = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HLS)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    # --- 1. 强力云朵识别 ---
    cloud_mask = (l_channel > 165) & (s_channel < 60)

    # --- 2. 边缘保护逻辑 ---
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    edge_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    edge_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edges = cv2.magnitude(edge_x, edge_y)
    
    edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, edge_mask = cv2.threshold(edges, 20, 255, cv2.THRESH_BINARY)
    
    safe_zone = cv2.dilate(edge_mask, np.ones((3, 3), np.uint8), iterations=1)

    # --- 3. 最终云朵掩码 ---
    final_cloud_mask = cloud_mask.astype(np.uint8) * 255
    final_cloud_mask[safe_zone > 0] = 0
    final_cloud_mask = cv2.dilate(final_cloud_mask, np.ones((3, 3), np.uint8), iterations=1)

    # 4. 涂白结果
    res_roi = roi_bgr.copy()
    res_roi[final_cloud_mask > 0] = [255, 255, 255]
    return res_roi

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--yolo_model', type=str, default='/home/gaoli/tower_detection_20250721_1900/weights/best.pt')
    args = parser.parse_args()

    # 统一变量名
    IMAGE_DIR = args.input_dir
    OUTPUT_DIR = args.output_dir
    YOLO_MODEL_PATH = args.yolo_model

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Loading YOLO model: {YOLO_MODEL_PATH} on {device}")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    
    image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    for filename in image_files:
        img_path = os.path.join(IMAGE_DIR, filename)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None: continue
        h, w = img_bgr.shape[:2]
        
        results = yolo_model.predict(source=img_path, conf=0.3, device=device, verbose=False)
        if not results or len(results[0]) == 0:
            # 如果没检测到，保持全白背景（或者你可以选择保留原图，目前逻辑是全白）
            final_img = np.ones_like(img_bgr) * 255
        else:
            final_img = np.ones_like(img_bgr) * 255
            res = results[0]
            items = res.obb if (hasattr(res, 'obb') and res.obb is not None) else res.boxes
            
            for i in range(len(items)):
                box = items.xyxy[i].cpu().numpy().flatten()
                ix1, iy1, ix2, iy2 = map(int, box)
                
                # 如果检测到的是铁塔，自动延展到底部，防止砍断
                if yolo_model.names[int(items.cls[i])] == 'tower':
                    iy2 = h
                
                ix1, iy1, ix2, iy2 = max(0, ix1), max(0, iy1), min(w, ix2), min(h, iy2)
                roi = img_bgr[iy1:iy2, ix1:ix2]
                if roi.size == 0: continue
                
                clean_roi = aggressive_cloud_cleaner(roi)
                final_img[iy1:iy2, ix1:ix2] = clean_roi

        cv2.imwrite(os.path.join(OUTPUT_DIR, filename), final_img)
        print(f"🧹 强力清洗已完成: {filename}")

if __name__ == "__main__":
    main()