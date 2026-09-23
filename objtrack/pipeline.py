# Bước 3: từ các ID đã kiểm chứng -> (a) chuỗi hành động từng ID (hình học + màu đèn, không dùng LLM)
# -> (b) ngoại hình từng ID (VLM trên ảnh cắt, kèm cờ hợp lệ để loại vật không phải xe/người)
# -> (c) mô tả từng đối tượng -> (d) mô tả từng đoạn (VLM trên frame có vẽ ID + ảnh phóng to + bảng dữ kiện).
import argparse, glob, json, os, re, time
from datetime import datetime

import cv2, numpy as np
from PIL import Image

from lights import light_color, lights_from_detections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument('video'); ap.add_argument('trk'); ap.add_argument('cfg')
ap.add_argument('--model', default='Qwen/Qwen3-VL-32B-Instruct-FP8')
ap.add_argument('--max_seconds', type=int, default=30)
ap.add_argument('--win', type=int, default=10); ap.add_argument('--overlap', type=int, default=1)
ap.add_argument('--min_h', type=int, default=45, help='đối tượng rõ: khung cao nhất >= chừng này px')
ap.add_argument('--runs_dir', default=f'{ROOT}/outputs/runs')
ap.add_argument('--run_name', default=None)
a = ap.parse_args()

VI = {'motor': 'xe máy', 'bicycle': 'xe đạp', 'tricycle': 'xe ba bánh', 'awning-tricycle': 'xe ba gác', 'car': 'ô tô con',
      'van': 'ô tô', 'truck': 'xe tải', 'bus': 'xe buýt', 'pedestrian': 'người đi bộ'}
TWO = {'motor', 'bicycle', 'tricycle', 'awning-tricycle'}
name = os.path.basename(a.video.rstrip('/'))
cfg = json.load(open(a.cfg)) if a.cfg and os.path.exists(a.cfg) else {}   # camera chưa khai báo vẫn chạy được
cfg.setdefault('zones', {})
fs = sorted(glob.glob(f'{a.video}/*.webp'))[:a.max_seconds]
frames = [cv2.imread(f) for f in fs]
H_IMG, W_IMG = frames[0].shape[:2]
T = [t for t in json.load(open(a.trk)) if t['t'][0] < len(fs)]
if cfg.get('primary_light'):
    light = [light_color(im, cfg['lights'][cfg['primary_light']]['roi']) for im in frames]
    AUTO_ROIS, AUTO_LIGHTS = [], []
else:                                                                  # tự tìm đèn, không biết đèn điều khiển hướng nào
    light = ['không rõ'] * len(frames)
    AUTO_ROIS = lights_from_detections(json.load(open(a.trk.replace('_trkv.json', '_det.json'))))
    AUTO_LIGHTS = [[light_color(im, r) for im in frames] for r in AUTO_ROIS]
    print(f'Camera chưa khai báo: tự tìm được {len(AUTO_ROIS)} đèn giao thông')
ZNAME = {z: v.get('name', z) for z, v in cfg['zones'].items()}


def in_poly(pt, poly): return cv2.pointPolygonTest(np.array(poly, np.float32), pt, False) >= 0


def zone_of(box):
    foot = ((box[0] + box[2]) / 2, box[3])
    for z, v in cfg['zones'].items():
        if in_poly(foot, v['poly']): return z
    return None


def where(box):
    cx, cy = (box[0] + box[2]) / 2, box[3]
    lr = 'bên trái' if cx < W_IMG / 3 else 'bên phải' if cx > 2 * W_IMG / 3 else 'giữa'
    fn = 'xa camera' if cy < H_IMG * 0.4 else 'gần camera' if cy > H_IMG * 0.7 else 'giữa khung hình'
    return f'{lr} khung hình, {fn}'


def direction(dx, dy):
    ang = np.degrees(np.arctan2(-dy, dx)) % 360
    names = ['sang phải', 'chéo lên phải (ra xa camera)', 'thẳng lên (ra xa camera)', 'chéo lên trái (ra xa camera)',
             'sang trái', 'chéo xuống trái (lại gần camera)', 'thẳng xuống (lại gần camera)', 'chéo xuống phải (lại gần camera)']
    return names[int(((ang + 22.5) % 360) // 45)]


def exit_side(box):
    x1, y1, x2, y2 = box; m = 60
    if x1 < m: return 'mép trái'
    if x2 > W_IMG - m: return 'mép phải'
    if y2 > H_IMG - m: return 'mép dưới (phía camera)'
    return None


# ---------- (a) chuỗi hành động từng ID ----------
def action_chain(ts, bs):
    h = max(1, np.median([b[3] - b[1] for b in bs]))
    c0, c1 = ((bs[0][0] + bs[0][2]) / 2, bs[0][3]), ((bs[-1][0] + bs[-1][2]) / 2, bs[-1][3])
    if len(ts) > 1 and np.hypot(c1[0] - c0[0], c1[1] - c0[1]) < max(0.5 * h, 20):   # khung chỉ rung tại chỗ
        z = zone_of(bs[0]); place = ZNAME.get(z) if z else where(bs[0])
        lis = [x for x in dict.fromkeys(light[t] for t in ts) if x != 'không rõ']
        return f"đứng yên ở {place}" + (f" (đèn {'→'.join(lis)})" if lis and z == 'approach_near' else '')
    # làm mượt quỹ đạo (trung bình trượt 3 điểm) để khung rung không tạo ra hướng giả
    if len(bs) >= 3:
        arr = np.array(bs, float); sm = arr.copy()
        for k in range(1, len(bs) - 1): sm[k] = arr[k - 1:k + 2].mean(0)
        bs = [list(map(int, b)) for b in sm]
    phases = []
    for k, (t, b) in enumerate(zip(ts, bs)):
        z = zone_of(b); place = ZNAME.get(z) if z else where(b)
        mv = None
        if k:
            p = bs[k - 1]; dx = (b[0] + b[2] - p[0] - p[2]) / 2; dy = (b[1] + b[3] - p[1] - p[3]) / 2
            mv = 'đứng yên' if np.hypot(dx, dy) < max(0.25 * h, 12) else f'di chuyển {direction(dx, dy)}'
        if phases and phases[-1]['place'] == place and (mv is None or phases[-1]['mv'] in (None, mv)):
            ph = phases[-1]; ph['lights'].append(light[t]); ph['mv'] = ph['mv'] or mv
        else:
            phases.append({'place': place, 'zone': z, 'mv': mv, 'lights': [light[t]]})
    # pha di chuyển chỉ 1 giây nằm giữa 2 pha khác -> gộp vào pha trước (thường là nhiễu)
    merged = []
    for ph in phases:
        if merged and len(ph['lights']) == 1 and ph is not phases[-1] and ph['place'] == merged[-1]['place']:
            merged[-1]['lights'] += ph['lights']; continue
        if merged and merged[-1]['place'] == ph['place'] and merged[-1]['mv'] == ph['mv']:
            merged[-1]['lights'] += ph['lights']; continue
        merged.append(ph)
    phases = merged
    parts = []
    for ph in phases:
        lis = [x for x in dict.fromkeys(ph['lights']) if x != 'không rõ']
        s = f"{ph['mv'] if ph['mv'] else 'xuất hiện'} ở {ph['place']}"
        if lis and ph['zone'] == 'approach_near': s += f" (đèn {'→'.join(lis)})"
        parts.append(s)
    for k in range(1, len(ts)):
        if zone_of(bs[k - 1]) == 'approach_near' and zone_of(bs[k]) == 'intersection':
            parts.append(f"qua vạch dừng vào giao lộ khi đèn {light[ts[k - 1]]}→{light[ts[k]]}")
    ex = exit_side(bs[-1])
    if ex and ts[-1] < len(fs) - 1: parts.append(f'rời khung hình ở {ex}')
    return ' → '.join(parts)


# ---------- trường hướng lưu thông học từ chính video (không cần khai báo) ----------
GX, GY = 12, 8
flow = np.zeros((GY, GX, 2)); cnt = np.zeros((GY, GX))
def cell(x, y): return min(GY - 1, max(0, int(y / H_IMG * GY))), min(GX - 1, max(0, int(x / W_IMG * GX)))
def steps(tr):
    for k in range(1, len(tr['t'])):
        if tr['t'][k] - tr['t'][k - 1] != 1: continue
        p, b = tr['boxes'][k - 1], tr['boxes'][k]; h = max(10, b[3] - b[1])
        v = np.array([(b[0] + b[2] - p[0] - p[2]) / 2, (b[3] - p[3])], float)
        if np.linalg.norm(v) < 0.4 * h: continue                     # chỉ tính bước thực sự di chuyển
        yield k, ((b[0] + b[2]) / 2 + (p[0] + p[2]) / 2) / 2, (b[3] + p[3]) / 2, v / np.linalg.norm(v)
for tr in T:
    for k, x, y, u in steps(tr):
        cy, cx = cell(x, y); flow[cy, cx] += u; cnt[cy, cx] += 1
coh = np.linalg.norm(flow, axis=2) / np.maximum(cnt, 1)               # độ thống nhất hướng trong ô (1 = mọi xe cùng hướng)


def wrong_way(tr):
    run = best = 0
    for k, x, y, u in steps(tr):
        cy, cx = cell(x, y)
        if cnt[cy, cx] >= 12 and coh[cy, cx] >= 0.75 and u @ (flow[cy, cx] / np.linalg.norm(flow[cy, cx])) < -0.7:
            run += 1; best = max(best, run)
        else: run = 0
    return best >= 2


objs = []
for tr in T:
    ts = [t for t in tr['t'] if t < len(fs)]; bs = tr['boxes'][:len(ts)]
    if max(b[3] - b[1] for b in bs) < a.min_h: continue          # chỉ đối tượng đã hiện rõ
    zones = [zone_of(b) for b in bs]
    if set(z for z in zones if z) == {'island'}: continue          # vật đứng trên đảo cỏ (cọc, chòi)
    anomalies = []
    for k in range(1, len(ts)):
        if zones[k - 1] == 'approach_near' and zones[k] == 'intersection' and light[ts[k - 1]] == 'đỏ' and light[ts[k]] == 'đỏ':
            anomalies.append('vượt đèn đỏ: đi từ làn tới vào giao lộ khi đèn của hướng này đã đỏ')
    if wrong_way(tr):
        anomalies.append('có thể đi ngược chiều: di chuyển ngược hướng với số đông xe trên cùng phần đường (chưa chắc chắn)')
    objs.append({'id': tr['id'], 'cls': tr['cls'], 'loai': VI[tr['cls']], 't0': ts[0], 't1': ts[-1], 'ts': ts, 'boxes': bs,
                 'chuoi_hanh_dong': action_chain(ts, bs), 'bat_thuong': anomalies})
print(f'{len(objs)} đối tượng rõ / {len(T)} ID')

# ---------- (b) ngoại hình bằng VLM ----------
from vllm import LLM, SamplingParams  # noqa: E402
llm = LLM(model=a.model, max_model_len=32768, gpu_memory_utilization=0.92, limit_mm_per_prompt={'image': 14, 'video': 0},
          max_num_seqs=128, enable_prefix_caching=True)
tok = llm.get_tokenizer()


def chat(content): return tok.apply_chat_template([{'role': 'user', 'content': content}], add_generation_prompt=True, tokenize=False)


def crop(t, box, two, minside=224):
    im = frames[t].copy(); x1, y1, x2, y2 = box; h, w = y2 - y1, x2 - x1
    top = y1 - int(0.9 * h) if two else y1                     # khung xe 2 bánh không bao đầu người lái
    cv2.rectangle(im, (x1 - 2, top - 2), (x2 + 2, y2 + 2), (0, 0, 255), 2)
    p = int(0.5 * max(w, h)) + 6
    c = im[max(0, top - p):y2 + p, max(0, x1 - p):x2 + p]; s = max(1.0, minside / min(c.shape[:2]))
    return Image.fromarray(cv2.cvtColor(cv2.resize(c, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB))


APP_PROMPT = open(f'{ROOT}/prompts/object_appearance_vi.txt').read()
reqs = []
for o in objs:
    two = o['cls'] in TWO
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in o['boxes']]
    k0 = int(np.argmax(areas)); pick = [k for k in (k0 - 1, k0, k0 + 1) if 0 <= k < len(areas)]
    hpx = max(b[3] - b[1] for b in o['boxes']) * (1.9 if two else 1)
    note = '' if hpx >= 70 else 'LƯU Ý: đối tượng nhỏ, KHÔNG nhận xét về mũ bảo hiểm.'
    imgs = [crop(o['ts'][k], o['boxes'][k], two) for k in pick]
    reqs.append({'prompt': chat([{'type': 'image'} for _ in imgs] + [{'type': 'text', 'text': APP_PROMPT.replace('{loai}', o['loai']).replace('{note}', note)}]),
                 'multi_modal_data': {'image': imgs}})
t0 = time.time()
outs = llm.generate(reqs, SamplingParams(temperature=0, max_tokens=80))
for o, r in zip(objs, outs):
    txt = r.outputs[0].text
    ok = re.search(r'HỢP LỆ:\s*(\S+)', txt); ds_ = re.search(r'MÔ TẢ:\s*(.+)', txt)
    o['hop_le'] = not (ok and ok.group(1).lower().startswith('không'))
    desc = (ds_.group(1) if ds_ else txt).strip().rstrip('.')
    # bỏ mọi cụm phủ định ("không đội mũ", "không có áo mưa"...): đã kiểm chứng là hay sai ở độ phân giải này
    desc = ', '.join(c.strip() for c in desc.split(',') if c.strip() and not re.match(r'(không|chưa)\b', c.strip(), re.I))
    o['ngoai_hinh'] = desc
    m = re.match(r'\s*(người đi bộ|CSGT|công an|xe tay ga|xe số|xe máy điện|xe máy|xe đạp|xe ba gác|xe ba bánh|ô tô con|SUV|bán tải|taxi|xe buýt|xe tải|xe van|ô tô)', o['ngoai_hinh'], re.I)
    if m: o['loai'] = m.group(1)
t_app = time.time() - t0
n0 = len(objs); objs = [o for o in objs if o['hop_le']]
print(f'VLM loại {n0 - len(objs)} đối tượng không hợp lệ, còn {len(objs)}')


# ---------- (c) mô tả từng đối tượng (ghép theo mẫu, không để LLM tự do) ----------
def obj_text(o):
    s = f"{o['ngoai_hinh']}: {o['chuoi_hanh_dong']}"
    if o['bat_thuong']: s += '. BẤT THƯỜNG: ' + '; '.join(o['bat_thuong'])
    return s


for o in objs: o['mo_ta'] = obj_text(o)

# ---------- (d) mô tả từng đoạn ----------
SEG_PROMPT = open(f'{ROOT}/prompts/segment_with_facts_vi.txt').read()


def som(t, ids):
    im = frames[t].copy()
    for o in objs:
        if o['id'] in ids and t in o['ts']:
            x1, y1, x2, y2 = o['boxes'][o['ts'].index(t)]
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 255), 2); cv2.putText(im, f"#{o['id']}", (x1, y1 - 4), 0, 0.8, (0, 255, 255), 2)
    return Image.fromarray(cv2.cvtColor(cv2.resize(im[30:], (1280, 700)), cv2.COLOR_BGR2RGB))


step = a.win - a.overlap
starts = [s for s in range(0, len(fs), step) if s == 0 or len(fs) - s > a.win // 2]
seg_reqs, segs = [], []
for s in starts:
    rng = range(s, min(s + a.win, len(fs)))
    inseg = [o for o in objs if o['t0'] <= rng[-1] and o['t1'] >= rng[0]]
    ids = {o['id'] for o in inseg}
    seq = lambda L: ' → '.join(c for i, c in enumerate(L[rng[0]:rng[-1] + 1]) if (i == 0 or c != L[rng[0] + i - 1]) and c != 'không rõ')
    lights = seq(light)
    if cfg.get('primary_light'):
        light_line = (f"- Đèn giao thông chính: {lights or 'không đọc được'}. Đèn này CHỈ điều khiển {cfg['lights'][cfg['primary_light']]['controls']}. "
                      "Xe chạy ngang qua giao lộ thuộc hướng khác, có pha đèn riêng không đo được; KHÔNG dùng màu đèn này để đánh giá xe chạy ngang.")
    elif AUTO_ROIS:
        light_line = '\n'.join(f"- Đèn giao thông tự phát hiện ở {where(r)}: {seq(L) or 'không đọc được màu'}" for r, L in zip(AUTO_ROIS, AUTO_LIGHTS)) + \
                     "\n  (không biết đèn điều khiển hướng nào: chỉ dùng để mô tả, KHÔNG dùng để kết luận vượt đèn đỏ)"
    else:
        light_line = '- Không phát hiện đèn giao thông trong khung hình.'
    def clip(o):
        idx = [k for k, t in enumerate(o['ts']) if rng[0] <= t <= rng[-1]]
        ts_, bs_ = [o['ts'][k] for k in idx], [o['boxes'][k] for k in idx]
        an = [x for x in o['bat_thuong'] if any(rng[0] <= t <= rng[-1] for t in ts_)]
        return f"#{o['id']} {o['ngoai_hinh']}: {action_chain(ts_, bs_)}" + ('. BẤT THƯỜNG: ' + '; '.join(an) if an else '')
    def prio(o):
        n = sum(1 for t in o['ts'] if rng[0] <= t <= rng[-1])
        return (bool(o['bat_thuong']), n >= 3, o['cls'] in ('car', 'van', 'truck', 'bus'), n, max(b[3] - b[1] for b in o['boxes']))
    ranked = sorted(inseg, key=prio, reverse=True)
    top, rest = ranked[:40], ranked[40:]
    facts = '\n'.join(clip(o) for o in top)
    if rest:
        cnt = {}
        for o in rest: cnt[o['loai'].split(',')[0]] = cnt.get(o['loai'].split(',')[0], 0) + 1
        facts += '\n(ngoài ra còn ' + ', '.join(f'{v} {k}' for k, v in cnt.items()) + ' chỉ xuất hiện ngắn, không liệt kê)'
    text = SEG_PROMPT.replace('{n}', str(len(rng))).replace('{light_line}', light_line).replace('{facts}', facts or '(không có)')
    content = []
    for k, t in enumerate(rng, 1): content += [{'type': 'text', 'text': f'[ảnh {k}]'}, {'type': 'image'}]
    zoom_ts = [rng[0], rng[len(rng) // 2], rng[-1]]
    for k in range(len(zoom_ts)): content += [{'type': 'text', 'text': f'[ảnh phóng to vùng giữa {k + 1}]'}, {'type': 'image'}]
    content.append({'type': 'text', 'text': text})
    x1, y1, x2, y2 = cfg.get('focus_roi', [W_IMG // 4, int(H_IMG * 0.12), 3 * W_IMG // 4, int(H_IMG * 0.6)])
    zooms = [Image.fromarray(cv2.cvtColor(cv2.resize(frames[t][y1:y2, x1:x2], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)) for t in zoom_ts]
    seg_reqs.append({'prompt': chat(content), 'multi_modal_data': {'image': [som(t, ids) for t in rng] + zooms}})
    segs.append({'start_sec': rng[0], 'end_sec': rng[-1] + 1, 'objects': sorted(ids), 'den': lights})
t0 = time.time()
outs = llm.generate(seg_reqs, SamplingParams(temperature=0, max_tokens=900, repetition_penalty=1.05))
t_seg = time.time() - t0
for sg, r in zip(segs, outs):
    txt = r.outputs[0].text.strip()
    txt = txt.replace('**', '')
    parts = re.split(r'\n?\s*BẤT THƯỜNG\s*:', txt, maxsplit=1)
    body = re.sub(r'^\s*MÔ TẢ\s*:', '', parts[0].strip())
    lines = [l.strip().lstrip('- ') for l in body.splitlines() if l.strip()]
    # lọc hậu kỳ: câu về mũ bảo hiểm không có căn cứ trong dữ kiện, câu phủ định chung chung
    lines = [l for l in lines if not re.search(r'không (đội|rõ|thấy rõ) mũ', l, re.I)
             and not re.match(r'(không có|không thấy|không xảy ra)\b', l, re.I)]
    sg['caption_lines'] = lines
    b = re.match(r'\s*(.*)', parts[1], re.S) if len(parts) > 1 else None
    sg['caption'] = ' '.join(sg['caption_lines'])
    sg['bat_thuong'] = ' '.join(l.strip().lstrip('- ') for l in b.group(1).splitlines() if l.strip()) if b else ''

# ---------- lưu ----------
run = a.run_name or f"{datetime.now():%Y%m%d-%H%M%S}_{a.model.split('/')[-1]}_objtrack"
d = f'{a.runs_dir}/{run}'; os.makedirs(d, exist_ok=True)
with open(f'{d}/{name}_objects.jsonl', 'w') as f:
    for o in objs:
        f.write(json.dumps({'video': name, 'id': o['id'], 'loai': o['loai'], 'start_sec': o['t0'], 'end_sec': o['t1'] + 1,
                            'ngoai_hinh': o['ngoai_hinh'], 'chuoi_hanh_dong': o['chuoi_hanh_dong'], 'bat_thuong': o['bat_thuong'],
                            'mo_ta': o['mo_ta'], 'frames': [os.path.basename(fs[t]).split('.')[0] for t in o['ts']], 'boxes': o['boxes']},
                           ensure_ascii=False) + '\n')
with open(f'{d}/{name}_segments.jsonl', 'w') as f:
    for sg in segs: f.write(json.dumps({'video': name, **sg}, ensure_ascii=False) + '\n')
with open(f'{d}/{name}.md', 'w') as f:
    f.write(f'# {name}\n\nĐèn chính theo giây: {" ".join(light)}\n')
    for sg in segs:
        f.write(f"\n---\n\n## Đoạn {sg['start_sec']}–{sg['end_sec']}s\n\n### Mô tả đoạn\n\n" + '\n'.join(sg['caption_lines']) +
                f"\n\n**Bất thường:** {sg['bat_thuong']}\n\n### Đối tượng ({len(sg['objects'])})\n\n")
        for o in sorted(objs, key=lambda o: (o['t0'], o['id'])):
            if o['id'] in sg['objects']:
                f.write(f"- **#{o['id']}** ({o['t0']}–{o['t1'] + 1}s) {o['ngoai_hinh']}\n  - {o['chuoi_hanh_dong']}" +
                        (f"\n  - ⚠️ {'; '.join(o['bat_thuong'])}" if o['bat_thuong'] else '') + '\n')
json.dump({'run': run, 'model': a.model, 'args': vars(a), 'camera_cfg': cfg, 'n_objects': len(objs), 'n_ids': len(T),
           'appearance_s': round(t_app, 1), 'segment_s': round(t_seg, 1), 'light': light}, open(f'{d}/meta.json', 'w'), ensure_ascii=False, indent=1)
print(f'Xong -> {d}  (ngoại hình {t_app:.1f}s, đoạn {t_seg:.1f}s)')
