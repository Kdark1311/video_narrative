# Vẽ ID track lên các frame để kiểm tra bằng mắt: python objtrack/viz.py video trk.json out.jpg t0 t1 [id...]
import sys, json, cv2, numpy as np, glob
video, trk, out, t0, t1 = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
only = set(map(int, sys.argv[6:]))
fs = sorted(glob.glob(f'{video}/*.webp')); T = json.load(open(trk)); tiles = []
for t in range(t0, t1):
    im = cv2.imread(fs[t])
    for tr in T:
        if t in tr['t'] and (not only or tr['id'] in only) and len(tr['t']) >= 2:
            x1, y1, x2, y2 = tr['boxes'][tr['t'].index(t)]
            col = tuple(int(c) for c in np.random.RandomState(tr['id']).randint(60, 255, 3))
            cv2.rectangle(im, (x1, y1), (x2, y2), col, 3); cv2.putText(im, str(tr['id']), (x1, y1 - 5), 0, 0.9, col, 2)
    cv2.putText(im, f'giay {t}', (20, 90), 0, 2, (0, 255, 255), 4)
    tiles.append(cv2.resize(im[80:800, 150:1920], (885, 360)))
while len(tiles) % 2: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(out, np.vstack([np.hstack(tiles[k:k + 2]) for k in range(0, len(tiles), 2)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
