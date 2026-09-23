# Đọc màu đèn giao thông bằng pixel trong vùng ROI đã khai báo: điểm sáng + bão hoà -> hue -> đỏ/vàng/xanh
import cv2, numpy as np

def light_color(img, roi):
    x1, y1, x2, y2 = roi
    hsv = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1], hsv[..., 2]
    lit = (v > 170) & (s > 90)                     # pixel đèn đang sáng
    if lit.sum() < 4: return 'không rõ'
    hh = h[lit]
    votes = {'đỏ': ((hh <= 8) | (hh >= 165)).sum(), 'vàng': ((hh > 8) & (hh <= 35)).sum(), 'xanh': ((hh >= 45) & (hh <= 100)).sum()}
    best = max(votes, key=votes.get)
    return best if votes[best] >= 3 else 'không rõ'

if __name__ == '__main__':
    import sys, json, glob
    video, cfg = sys.argv[1], json.load(open(sys.argv[2]))
    fs = sorted(glob.glob(f'{video}/*.webp'))[:int(sys.argv[3]) if len(sys.argv) > 3 else None]
    for name, L in cfg['lights'].items():
        print(name, ' '.join(f'{t}:{light_color(cv2.imread(f), L["roi"])}' for t, f in enumerate(fs)))



def lights_from_detections(det, max_lights=3, min_frames=3):
    """Camera chưa khai báo: gom các khung đèn giao thông mà detect.py tìm được qua các frame, giữ đèn xuất hiện ổn định."""
    rois = []   # [box, số frame]
    for fr in det:
        for b in fr.get('lights', []):
            for r in rois:
                x = max(0, min(b[2], r[0][2]) - max(b[0], r[0][0])) * max(0, min(b[3], r[0][3]) - max(b[1], r[0][1]))
                if x > 0: r[1] += 1; break
            else: rois.append([b, 1])
    rois = [r for r in rois if r[1] >= min_frames]
    rois.sort(key=lambda r: -(r[0][2] - r[0][0]) * (r[0][3] - r[0][1]))
    return [r[0] for r in rois[:max_lights]]
