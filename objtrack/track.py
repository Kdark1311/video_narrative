# Bước 2: theo dõi đối tượng ở 1 fps.
# - Chỉ tạo ID khi đối tượng đã hiện rõ (khung đủ lớn, độ tin cậy đủ cao).
# - Ghép giai đoạn 1 (tự động): ngoại hình DINOv2 + histogram màu + khoảng cách tới vị trí dự đoán + IoU (xe đứng yên),
#   chỉ nhận khi KHÔNG mơ hồ.
# - Ghép giai đoạn 2 (--vlm): các xe 2 bánh/người còn mơ hồ -> VLM so từng cặp ảnh (có khung đỏ đánh dấu), chỉ nhận khi
#   một ứng viên chắc chắn >= min_p và hơn hẳn các ứng viên khác.
# Sau bước này vẫn chạy verify_links.py để VLM kiểm tra lại mọi lần ghép.
import argparse, collections, glob, json, math
import cv2, numpy as np, torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from transformers import AutoImageProcessor, AutoModel

GROUP = {'motor': 'two', 'bicycle': 'two', 'tricycle': 'two', 'awning-tricycle': 'two',
         'car': 'four', 'van': 'four', 'truck': 'big', 'bus': 'big', 'pedestrian': 'ped'}
COMPAT = {('four', 'big'), ('big', 'four')}  # xe tải/van/ô tô hay bị nhầm lẫn nhau
ap = argparse.ArgumentParser()
ap.add_argument('video'); ap.add_argument('det'); ap.add_argument('out')
ap.add_argument('--max_miss', type=int, default=2, help='ô tô/xe lớn: số giây được mất dấu; xe 2 bánh/người: không được mất dấu')
ap.add_argument('--max_cost', type=float, default=1.5)
ap.add_argument('--max_app', type=float, default=0.30)
ap.add_argument('--max_anchor', type=float, default=0.40, help='khác biệt tối đa so với ảnh mẫu lúc rõ nhất (chống ID trôi)')
ap.add_argument('--margin', type=float, default=0.05, help='xe 2 bánh/người: ứng viên tốt nhất phải hơn ứng viên thứ 2 chừng này')
ap.add_argument('--min_h_start', type=int, default=45, help='chỉ tạo ID mới khi khung cao >= chừng này px')
ap.add_argument('--min_conf_start', type=float, default=0.45)
ap.add_argument('--vlm', default=None, help='model VLM để phân xử các trường hợp mơ hồ, vd Qwen/Qwen3-VL-32B-Instruct-FP8')
ap.add_argument('--min_p', type=float, default=0.85)
a = ap.parse_args()

llm = None
if a.vlm:
    from vllm import LLM, SamplingParams
    llm = LLM(model=a.vlm, max_model_len=8192, gpu_memory_utilization=0.85, limit_mm_per_prompt={'image': 2, 'video': 0}, max_num_seqs=256)
    Q = ('Hai ảnh trên được cắt từ camera giao thông ở 2 giây liên tiếp. Chỉ xét đối tượng nằm trong KHUNG ĐỎ ở mỗi ảnh. '
         'Hai đối tượng trong khung đỏ có phải là CÙNG MỘT phương tiện (cùng người lái) hoặc cùng một người không? '
         'So sánh loại xe, màu xe, số người, màu áo, màu mũ, hàng hoá. Góc nhìn và kích thước có thể thay đổi. Chỉ trả lời một từ: CÓ hoặc KHÔNG.')
    VPROMPT = llm.get_tokenizer().apply_chat_template([{'role': 'user', 'content': [{'type': 'image'}, {'type': 'image'}, {'type': 'text', 'text': Q}]}],
                                                      add_generation_prompt=True, tokenize=False)
    VSP = SamplingParams(temperature=0, max_tokens=1, logprobs=10)
# DINOv2 khởi tạo CUDA -> phải tạo SAU vLLM, nếu không vLLM phải spawn tiến trình và lỗi
proc = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
enc = AutoModel.from_pretrained('facebook/dinov2-small').cuda().eval().half()


def iou(a_, b_, mode='iou'):
    ix = max(0, min(a_[2], b_[2]) - max(a_[0], b_[0])); iy = max(0, min(a_[3], b_[3]) - max(a_[1], b_[1])); inter = ix * iy
    ar = lambda b: (b[2] - b[0]) * (b[3] - b[1])
    return inter / (min(ar(a_), ar(b_)) if mode == 'ios' else ar(a_) + ar(b_) - inter + 1e-6)


def clean(ds):
    ds = sorted(ds, key=lambda d: -d['conf']); keep = []
    for d in ds:                                                   # NMS không phân biệt lớp: 1 xe chỉ còn 1 khung
        if all(iou(d['box'], k['box']) < 0.6 for k in keep): keep.append(d)
    two = [k['box'] for k in keep if GROUP[k['cls']] == 'two']      # người ngồi trên xe 2 bánh -> gộp vào xe
    return [d for d in keep if not (d['cls'] == 'pedestrian' and any(iou(d['box'], b, 'ios') > 0.4 for b in two))]


def colhist(img, boxes):
    out = []
    for x1, y1, x2, y2 in boxes:
        c = img[max(0, y1 - int(0.8 * (y2 - y1))):y2, max(0, x1):x2]   # gồm cả người lái phía trên
        h = cv2.calcHist([cv2.cvtColor(c, cv2.COLOR_BGR2HSV)], [0, 1], None, [18, 8], [0, 180, 0, 256]).flatten()
        out.append((h / (h.sum() + 1e-6)).astype(np.float32))
    return out


@torch.no_grad()
def embed(img, boxes):
    if not boxes: return np.zeros((0, 384), np.float32)
    crops = [cv2.cvtColor(img[max(0, y1):y2, max(0, x1):x2], cv2.COLOR_BGR2RGB) for x1, y1, x2, y2 in boxes]
    x = proc(images=crops, return_tensors='pt', size={'height': 112, 'width': 112}, do_center_crop=False)['pixel_values'].cuda().half()
    return torch.nn.functional.normalize(enc(pixel_values=x).pooler_output.float(), dim=1).cpu().numpy()


def marked(img, box, two):
    im = img.copy(); x1, y1, x2, y2 = box; h, w = y2 - y1, x2 - x1
    top = y1 - int(0.9 * h) if two else y1
    cv2.rectangle(im, (x1 - 2, top - 2), (x2 + 2, y2 + 2), (0, 0, 255), 2)
    p = int(0.6 * max(w, h)) + 6
    c = im[max(0, top - p):y2 + p, max(0, x1 - p):x2 + p]; s = max(1.0, 224 / min(c.shape[:2]))
    return Image.fromarray(cv2.cvtColor(cv2.resize(c, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB))


def p_yes(o):
    lp = o.outputs[0].logprobs[0]
    py = sum(math.exp(x.logprob) for x in lp.values() if x.decoded_token and x.decoded_token.strip().upper().startswith('C'))
    pn = sum(math.exp(x.logprob) for x in lp.values() if x.decoded_token and x.decoded_token.strip().upper().startswith('KH'))
    return py / (py + pn + 1e-9)


det = json.load(open(a.det))
tracks, active, nid, prev_img = {}, [], 1, None
n_vlm_q = n_vlm_ok = 0
for fr in det:
    t = fr['t']; img = cv2.imread(f"{a.video}/{fr['frame']}")
    ds = clean(fr['dets']); boxes = [d['box'] for d in ds]; E = embed(img, boxes); H = colhist(img, boxes)
    C = np.full((len(active), len(ds)), 9.0)
    G = np.full((len(active), len(ds)), 9.0)          # chi phí chỉ theo vị trí (để lấy ứng viên cho VLM)
    for i, tid in enumerate(active):
        tr = tracks[tid]; lx1, ly1, lx2, ly2 = tr['boxes'][-1]; gap = t - tr['t'][-1]
        if gap > 1 and GROUP[tr['cls_hist'][-1]] in ('two', 'ped'): continue   # xe 2 bánh mất dấu 1 giây: không ghép lại
        cx, cy = (lx1 + lx2) / 2 + tr['v'][0] * gap, (ly1 + ly2) / 2 + tr['v'][1] * gap
        max_d = min(600, max(80, 6 * max(lx2 - lx1, ly2 - ly1))) * gap
        for j, d in enumerate(ds):
            g1, g2 = GROUP[tr['cls_hist'][-1]], GROUP[d['cls']]
            if g1 != g2 and (g1, g2) not in COMPAT: continue
            x1, y1, x2, y2 = d['box']; dist = np.hypot((x1 + x2) / 2 - cx, (y1 + y2) / 2 - cy)
            if dist > max_d: continue
            G[i, j] = dist / max_d
            app = 1 - float(tr['emb'] @ E[j])
            if app > a.max_app or 1 - float(tr['anchor'] @ E[j]) > a.max_anchor: continue
            col = cv2.compareHist(tr['hist'], H[j], cv2.HISTCMP_BHATTACHARYYA)
            if col > 0.55: continue
            C[i, j] = app / 0.5 + dist / max_d + col - (iou(tr['boxes'][-1], d['box']) if gap == 1 else 0)

    def take(i, j):
        tr = tracks[active[i]]; d = ds[j]
        (px1, py1, px2, py2), gap = tr['boxes'][-1], t - tr['t'][-1]
        x1, y1, x2, y2 = d['box']
        nv = (((x1 + x2) - (px1 + px2)) / 2 / gap, ((y1 + y2) - (py1 + py2)) / 2 / gap)
        tr['v'] = (0.5 * tr['v'][0] + 0.5 * nv[0], 0.5 * tr['v'][1] + 0.5 * nv[1])
        tr['emb'] = 0.7 * tr['emb'] + 0.3 * E[j]; tr['emb'] /= np.linalg.norm(tr['emb']); tr['hist'] = 0.6 * tr['hist'] + 0.4 * H[j]
        tr['t'].append(t); tr['boxes'].append(d['box']); tr['cls_hist'].append(d['cls']); tr['conf'].append(d['conf'])
        if y2 - y1 > tr['anchor_h'] and d['conf'] >= a.min_conf_start: tr['anchor'], tr['anchor_h'] = E[j].copy(), y2 - y1

    matched_r, matched_c, ambiguous = set(), set(), []
    if len(active) and len(ds):
        for i, j in zip(*linear_sum_assignment(C)):
            if C[i, j] > a.max_cost: continue
            row = np.sort(C[i]); col_ = np.sort(C[:, j])
            second = min(row[1] if len(row) > 1 else 9, col_[1] if len(col_) > 1 else 9)
            strong = iou(tracks[active[i]]['boxes'][-1], ds[j]['box']) >= 0.6 and t - tracks[active[i]]['t'][-1] == 1
            if GROUP[ds[j]['cls']] in ('two', 'ped') and not strong and second - C[i, j] < a.margin:
                ambiguous.append(i); continue                           # mơ hồ -> để VLM phân xử (hoặc bỏ)
            take(i, j); matched_r.add(i); matched_c.add(j)
    # ---- giai đoạn 2: VLM phân xử các trường hợp mơ hồ ----
    if llm is not None:
        ambiguous = [i for i in range(len(active)) if i not in matched_r and t - tracks[active[i]]['t'][-1] == 1
                     and GROUP[tracks[active[i]]['cls_hist'][-1]] in ('two', 'ped')]
    if llm is not None and ambiguous:
        reqs, meta = [], []
        for i in ambiguous:
            tr = tracks[active[i]]
            cands = [j for j in np.argsort(G[i]) if G[i, j] < 9 and j not in matched_c][:3]
            two = GROUP[tr['cls_hist'][-1]] == 'two'
            for j in cands:
                reqs.append({'prompt': VPROMPT, 'multi_modal_data': {'image': [marked(prev_img, tr['boxes'][-1], two), marked(img, ds[j]['box'], two)]}})
                meta.append((i, j))
        n_vlm_q += len(reqs)
        P = [p_yes(o) for o in llm.generate(reqs, VSP, use_tqdm=False)] if reqs else []
        best = collections.defaultdict(list)
        for (i, j), p in zip(meta, P): best[i].append((p, j))
        for i, lst in sorted(best.items(), key=lambda kv: -max(p for p, _ in kv[1])):
            lst.sort(reverse=True)
            p1, j1 = lst[0]; p2 = lst[1][0] if len(lst) > 1 else 0
            if p1 >= a.min_p and p1 - p2 >= 0.3 and j1 not in matched_c and i not in matched_r:
                take(i, j1); matched_r.add(i); matched_c.add(j1); n_vlm_ok += 1
    for j, d in enumerate(ds):
        if j in matched_c or d['conf'] < a.min_conf_start or d['box'][3] - d['box'][1] < a.min_h_start: continue   # chưa rõ -> chưa gán ID
        tracks[nid] = {'id': nid, 't': [t], 'boxes': [d['box']], 'cls_hist': [d['cls']], 'conf': [d['conf']], 'v': (0.0, 0.0),
                       'emb': E[j].copy(), 'hist': H[j].copy(), 'anchor': E[j].copy(), 'anchor_h': d['box'][3] - d['box'][1]}
        nid += 1
    active = [k for k, tr in tracks.items() if t - tr['t'][-1] < a.max_miss + 1 and (tr['t'][-1] == t or k in active)]
    prev_img = img

out = []
for tr in tracks.values():
    cls = collections.Counter(tr['cls_hist']).most_common(1)[0][0]   # chốt loại theo đa số trong cả vòng đời
    out.append({'id': tr['id'], 'cls': cls, 't': tr['t'], 'boxes': tr['boxes'], 'conf': tr['conf']})
json.dump(out, open(a.out, 'w'))
L = np.array([len(o['t']) for o in out])
print(f'{len(out)} ID | dài >=5s: {(L >= 5).sum()} | 1s: {(L == 1).sum()} | trung vị {np.median(L):.0f}s | VLM hỏi {n_vlm_q}, ghép thêm {n_vlm_ok}')
