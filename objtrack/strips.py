# Kiểm tra độ "sạch" của ID: mỗi hàng = ảnh cắt của 1 ID qua các giây. Hàng nào lẫn xe khác là ID bị tráo.
import sys, json, glob, cv2, numpy as np
video, trk, out = sys.argv[1], sys.argv[2], sys.argv[3]; n = int(sys.argv[4]) if len(sys.argv) > 4 else 14
fs = sorted(glob.glob(f'{video}/*.webp')); T = json.load(open(trk)); cache = {}
def disp(tr):
    c = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in tr['boxes']]; return np.hypot(c[-1][0] - c[0][0], c[-1][1] - c[0][1])
moving = sorted([t for t in T if len(t['t']) >= 4 and disp(t) > 150], key=lambda t: -len(t['t']))
pick = moving[::max(1, len(moving) // n)][:n]
rows = []
for tr in pick:
    cells = []
    for t, (x1, y1, x2, y2) in list(zip(tr['t'], tr['boxes']))[:12]:
        im = cache.setdefault(t, cv2.imread(fs[t])); p = 10
        c = cv2.resize(im[max(0, y1 - p):y2 + p, max(0, x1 - p):x2 + p], (90, 90)); cv2.putText(c, str(t), (2, 14), 0, 0.45, (0, 255, 255), 1); cells.append(c)
    cells += [np.zeros((90, 90, 3), np.uint8)] * (12 - len(cells))
    lab = np.zeros((90, 110, 3), np.uint8); cv2.putText(lab, f'#{tr["id"]}', (4, 40), 0, 0.7, (255, 255, 255), 2); cv2.putText(lab, tr['cls'][:9], (4, 70), 0, 0.5, (0, 255, 0), 1)
    rows.append(np.hstack([lab] + cells))
cv2.imwrite(out, np.vstack(rows)); print(len(moving), 'ID di chuyển; vẽ', len(pick))
