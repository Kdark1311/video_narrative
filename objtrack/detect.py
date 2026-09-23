# Bước 1: phát hiện đối tượng mỗi frame bằng YOLO11m-VisDrone + SAHI (cắt ô 640) -> detections.json
import argparse, glob, json, os
import cv2
from huggingface_hub import hf_hub_download
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

VEH = {'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'}
ap = argparse.ArgumentParser()
ap.add_argument('video'); ap.add_argument('out')
ap.add_argument('--max_seconds', type=int, default=None)
ap.add_argument('--conf', type=float, default=0.2)
ap.add_argument('--crop_top', type=int, default=30)      # bỏ dải chữ chèn
ap.add_argument('--min_ped_h', type=int, default=28)     # người đi bộ nhỏ hơn -> bỏ (ô giả trên cỏ)
a = ap.parse_args()
dm = AutoDetectionModel.from_pretrained(model_type='ultralytics', model_path=hf_hub_download('dronefreak/visdrone-yolov11m', 'best.pt'),
                                        confidence_threshold=a.conf, device='cuda:0', image_size=960)
from ultralytics import YOLO
coco = YOLO('yolo11x.pt')   # COCO phát hiện người tốt hơn VisDrone (người đứng, CSGT)
fs = sorted(glob.glob(f'{a.video}/*.webp'))[:a.max_seconds]
out = []
for t, f in enumerate(fs):
    img = cv2.imread(f)
    res = get_sliced_prediction(img, dm, slice_height=640, slice_width=640, overlap_height_ratio=0.25, overlap_width_ratio=0.25,
                                perform_standard_pred=True, postprocess_type='GREEDYNMM', postprocess_match_metric='IOS',
                                postprocess_match_threshold=0.6, verbose=0)
    dets = []
    for p in res.object_prediction_list:
        x1, y1, x2, y2 = p.bbox.to_xyxy(); c = p.category.name
        if y2 < a.crop_top: continue
        if c not in VEH: continue      # người lấy từ COCO bên dưới
        dets.append({'box': [int(round(float(v))) for v in (x1, y1, x2, y2)], 'cls': 'pedestrian' if c == 'people' else c, 'conf': round(float(p.score.value), 3)})
    r = coco.predict(img, classes=[0, 9], imgsz=1920, conf=0.3, verbose=False)[0]
    lights = []
    for (x1, y1, x2, y2), cf, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.int().tolist()):
        if c == 9:                                        # đèn giao thông -> dùng khi camera chưa khai báo
            if max(x2 - x1, y2 - y1) >= 15: lights.append([int(x1), int(y1), int(x2), int(y2)])
            continue
        if y2 < a.crop_top or (y2 - y1) < a.min_ped_h: continue
        dets.append({'box': [int(x1), int(y1), int(x2), int(y2)], 'cls': 'pedestrian', 'conf': round(float(cf), 3)})
    out.append({'t': t, 'frame': os.path.basename(f), 'dets': dets, 'lights': lights})
json.dump(out, open(a.out, 'w'))
print(f'{len(fs)} frame, trung bình {sum(len(o["dets"]) for o in out)/len(out):.1f} đối tượng/frame')
