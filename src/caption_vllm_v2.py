# v2: mỗi đoạn = WIN ảnh (1 ảnh/giây) đưa vào model dưới dạng ảnh có nhãn [giây t] để mốc thời gian chính xác từng giây.
# Output dạng văn bản nhiều dòng (dễ đọc để kiểm tra) + jsonl có tách sẵn các mục.
# Mỗi lần chạy lưu vào outputs/runs/<thời gian>_<model>_v2/: meta.json, <video>.md, <video>.jsonl
import argparse, glob, json, os, re, subprocess, time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument('videos', nargs='+', help='thư mục video, vd /content/traffic/data/N098/N098-V001')
ap.add_argument('--model', default='Qwen/Qwen3-VL-32B-Instruct-FP8')
ap.add_argument('--prompt', default=f'{ROOT}/prompts/traffic_caption_v2_vi.txt')
ap.add_argument('--runs_dir', default=f'{ROOT}/outputs/runs')
ap.add_argument('--run_name', default=None)
ap.add_argument('--win', type=int, default=10, help='số giây (ảnh) mỗi đoạn')
ap.add_argument('--overlap', type=int, default=1, help='số giây chồng lấn giữa 2 đoạn liên tiếp')
ap.add_argument('--max_seconds', type=int, default=None, help='chỉ chạy N giây đầu (để test)')
ap.add_argument('--fps', type=float, default=1.0)
ap.add_argument('--w', type=int, default=1280)
ap.add_argument('--h', type=int, default=720)
ap.add_argument('--crop_top', type=int, default=30)
ap.add_argument('--max_tokens', type=int, default=1536)
ap.add_argument('--gpu_util', type=float, default=0.92)
a = ap.parse_args()

prompt_tpl = open(a.prompt).read()
run_name = a.run_name or f"{datetime.now():%Y%m%d-%H%M%S}_{a.model.split('/')[-1]}_v2"
run_dir = f'{a.runs_dir}/{run_name}'
os.makedirs(run_dir, exist_ok=True)

from vllm import LLM, SamplingParams  # noqa: E402

t_load = time.time()
llm = LLM(model=a.model, max_model_len=16384, gpu_memory_utilization=a.gpu_util,
          limit_mm_per_prompt={'image': a.win, 'video': 0}, max_num_seqs=128, enable_prefix_caching=True)
t_load = time.time() - t_load
sp = SamplingParams(temperature=0, max_tokens=a.max_tokens, repetition_penalty=1.05)
tok = llm.get_tokenizer()


def load(f):
    im = Image.open(f).convert('RGB')
    return im.crop((0, a.crop_top, im.width, im.height)).resize((a.w, a.h))


def build_prompt(secs):
    content = []
    for t in secs:
        content += [{'type': 'text', 'text': f'[giây {t:g}]'}, {'type': 'image'}]
    content.append({'type': 'text', 'text': prompt_tpl.replace('{n}', str(len(secs)))})
    return tok.apply_chat_template([{'role': 'user', 'content': content}], add_generation_prompt=True, tokenize=False)


SECTIONS = ['BỐI CẢNH', 'ĐÈN GIAO THÔNG', 'CHUỖI HÀNH ĐỘNG', 'BẤT THƯỜNG', 'TÓM TẮT']


def parse(txt):
    out, cur = {}, None
    for line in txt.strip().splitlines():
        head = line.strip().rstrip(':').upper()
        if head in SECTIONS:
            cur = head; out[cur] = []
        elif cur and line.strip():
            out[cur].append(line.strip().lstrip('- ').strip())
    return {k: out.get(k, []) for k in SECTIONS}


meta = {'run': run_name, 'model': a.model, 'args': vars(a), 'prompt': prompt_tpl, 'mode': 'images (1 ảnh/giây, có nhãn giây)',
        'gpu': subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip(),
        'model_load_s': round(t_load, 1), 'videos': {}}
pool = ThreadPoolExecutor(16)
step = a.win - a.overlap
for vdir in a.videos:
    name = os.path.basename(vdir.rstrip('/'))
    fs = sorted(glob.glob(f'{vdir}/*.webp') or glob.glob(f'{vdir}/*.jpg'))
    if a.max_seconds:
        fs = fs[:int(a.max_seconds * a.fps)]
    imgs = list(pool.map(load, fs))
    # bỏ đoạn cuối nếu ngắn hơn nửa cửa sổ (phần đó đã nằm trong đoạn trước)
    starts = [s for s in range(0, len(fs), step) if s == 0 or len(fs) - s > a.win // 2]
    reqs, spans = [], []
    for s in starts:
        idx = list(range(s, min(s + a.win, len(fs))))
        secs = [i / a.fps for i in idx]
        spans.append((idx, secs))
        reqs.append({'prompt': build_prompt(secs), 'multi_modal_data': {'image': [imgs[i] for i in idx]}})
    t0 = time.time()
    outs = llm.generate(reqs, sp)
    dt = time.time() - t0
    with open(f'{run_dir}/{name}.jsonl', 'w') as fj, open(f'{run_dir}/{name}.md', 'w') as fm:
        fm.write(f'# {name}\n\nModel: {a.model} · đoạn {a.win}s, chồng {a.overlap}s\n')
        for (idx, secs), o in zip(spans, outs):
            txt = o.outputs[0].text.strip()
            f0, f1 = os.path.basename(fs[idx[0]]).split('.')[0], os.path.basename(fs[idx[-1]]).split('.')[0]
            fj.write(json.dumps({'video': name, 'start_sec': secs[0], 'end_sec': secs[-1] + 1 / a.fps,
                                 'start_frame': f0, 'end_frame': f1, 'text': txt, **parse(txt)}, ensure_ascii=False) + '\n')
            fm.write(f'\n---\n\n## Giây {secs[0]:g}–{secs[-1] + 1 / a.fps:g}  (frame {f0}–{f1})\n\n{txt}\n')
    meta['videos'][name] = {'segments': len(starts), 'gen_s': round(dt, 1)}
    json.dump(meta, open(f'{run_dir}/meta.json', 'w'), ensure_ascii=False, indent=1)
    print(f'[{name}] {len(starts)} đoạn | sinh {dt:.1f}s -> {run_dir}/{name}.md', flush=True)
