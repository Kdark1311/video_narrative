# Bước 2b: dùng VLM làm "trọng tài" cho từng lần ghép ID giữa 2 giây liên tiếp.
# Mỗi cặp (ảnh giây t, ảnh giây t+k) của cùng 1 ID -> hỏi "cùng một đối tượng không?". KHÔNG -> cắt ID tại đó.
import argparse, glob, json, os
import cv2
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument('video'); ap.add_argument('trk'); ap.add_argument('out')
ap.add_argument('--model', default='Qwen/Qwen3-VL-32B-Instruct-FP8')
ap.add_argument('--min_p', type=float, default=0.85, help='xác suất CÓ tối thiểu để giữ lần ghép')
a = ap.parse_args()
TWO = {'motor', 'bicycle', 'tricycle', 'awning-tricycle'}
fs = sorted(glob.glob(f'{a.video}/*.webp')); cache = {}
T = json.load(open(a.trk))


def crop(t, box, two):
    # ảnh cắt có thêm ngữ cảnh + khung đỏ đánh dấu đúng đối tượng cần so (trong đám đông ảnh cắt chứa nhiều xe)
    im = cache.setdefault(t, cv2.imread(fs[t])).copy()
    x1, y1, x2, y2 = box; h = y2 - y1; w = x2 - x1
    top = y1 - int(0.9 * h) if two else y1
    cv2.rectangle(im, (x1 - 2, top - 2), (x2 + 2, y2 + 2), (0, 0, 255), 2)
    p = int(0.6 * max(w, h)) + 6
    c = im[max(0, top - p):y2 + p, max(0, x1 - p):x2 + p]
    s = max(1.0, 224 / min(c.shape[:2]))
    return Image.fromarray(cv2.cvtColor(cv2.resize(c, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB))


from vllm import LLM, SamplingParams  # noqa: E402
llm = LLM(model=a.model, max_model_len=8192, gpu_memory_utilization=0.92, limit_mm_per_prompt={'image': 2, 'video': 0}, max_num_seqs=256)
tok = llm.get_tokenizer()
Q = ('Hai ảnh trên được cắt từ camera giao thông ở 2 thời điểm cách nhau khoảng 1 giây. '
     'Chỉ xét đối tượng nằm trong KHUNG ĐỎ ở mỗi ảnh. Hai đối tượng trong khung đỏ có phải là CÙNG MỘT phương tiện (cùng người lái) hoặc cùng một người không? '
     'So sánh loại xe, màu xe, số người, màu áo, màu mũ, hàng hoá. Góc nhìn và kích thước có thể thay đổi. '
     'Chỉ trả lời một từ: CÓ hoặc KHÔNG.')
prompt = tok.apply_chat_template([{'role': 'user', 'content': [{'type': 'image'}, {'type': 'image'}, {'type': 'text', 'text': Q}]}],
                                 add_generation_prompt=True, tokenize=False)
reqs, links = [], []
for ti, tr in enumerate(T):
    two = tr['cls'] in TWO
    for k in range(1, len(tr['t'])):
        reqs.append({'prompt': prompt, 'multi_modal_data': {'image': [crop(tr['t'][k - 1], tr['boxes'][k - 1], two), crop(tr['t'][k], tr['boxes'][k], two)]}})
        links.append((ti, k))
import math
outs = llm.generate(reqs, SamplingParams(temperature=0, max_tokens=1, logprobs=10))
def p_yes(o):
    lp = o.outputs[0].logprobs[0]
    py = sum(math.exp(x.logprob) for x in lp.values() if x.decoded_token and x.decoded_token.strip().upper().startswith('C'))
    pn = sum(math.exp(x.logprob) for x in lp.values() if x.decoded_token and x.decoded_token.strip().upper().startswith('KH'))
    return py / (py + pn + 1e-9)
P = [p_yes(o) for o in outs]
bad = {l for l, p in zip(links, P) if p < a.min_p}
json.dump([[T[ti]['id'], k, round(p, 3)] for (ti, k), p in zip(links, P)], open(a.out + '.links_dump.json', 'w'))

out, nid = [], 1
for ti, tr in enumerate(T):
    cuts = [0] + [k for k in range(1, len(tr['t'])) if (ti, k) in bad] + [len(tr['t'])]
    for s, e in zip(cuts, cuts[1:]):
        out.append({'id': nid, 'cls': tr['cls'], 't': tr['t'][s:e], 'boxes': tr['boxes'][s:e], 'conf': tr['conf'][s:e], 'from_track': tr['id']}); nid += 1
json.dump(out, open(a.out, 'w'))
print(f'{len(links)} lần ghép được kiểm tra, VLM bác {len(bad)} ({100 * len(bad) / max(1, len(links)):.0f}%) -> {len(out)} ID sau khi cắt')
