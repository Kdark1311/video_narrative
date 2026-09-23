# Video caption cho camera giao thông cố định bằng vLLM + Qwen3-VL.
# Mỗi video (thư mục keyframe 1 fps) được cắt thành cửa sổ WIN frame, trượt STRIDE frame;
# mỗi cửa sổ đưa vào model ở chế độ video. Mỗi lần chạy lưu vào outputs/runs/<thời gian>_<model>/.
import argparse, glob, json, os, platform, re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument('videos', nargs='+', help='thư mục video (vd data/N091/N091-V001) hoặc thư mục cha chứa nhiều video')
ap.add_argument('--model', default='Qwen/Qwen3-VL-32B-Instruct-FP8')
ap.add_argument('--prompt', default=f'{ROOT}/prompts/traffic_caption_vi.txt')
ap.add_argument('--runs_dir', default=f'{ROOT}/outputs/runs')
ap.add_argument('--run_name', default=None, help='mặc định: <thời gian>_<tên model>')
ap.add_argument('--win', type=int, default=20)
ap.add_argument('--stride', type=int, default=10)
ap.add_argument('--fps', type=float, default=1.0, help='fps của keyframe (dữ liệu hiện tại: 1 frame/giây)')
ap.add_argument('--w', type=int, default=1280)
ap.add_argument('--h', type=int, default=720)
ap.add_argument('--crop_top', type=int, default=30, help='cắt dải chữ chèn phía trên (tên camera, giờ)')
ap.add_argument('--max_tokens', type=int, default=1536)
ap.add_argument('--chunk', type=int, default=4, help='số video gửi vào vLLM cùng lúc để GPU luôn đầy')
ap.add_argument('--max_model_len', type=int, default=12288)
ap.add_argument('--gpu_util', type=float, default=0.92)
ap.add_argument('--kv_dtype', default='auto')
a = ap.parse_args()


def expand(paths):
    out = []
    for p in paths:
        p = p.rstrip('/')
        if glob.glob(f'{p}/*.webp') or glob.glob(f'{p}/*.jpg'):
            out.append(p)
        else:  # thư mục cha: lấy mọi thư mục con có ảnh
            out += sorted(os.path.dirname(f) for f in glob.glob(f'{p}/**/000000.*', recursive=True))
    return out


videos = expand(a.videos)
prompt_tpl = open(a.prompt).read()
run_name = a.run_name or f"{datetime.now():%Y%m%d-%H%M%S}_{a.model.split('/')[-1]}"
run_dir = f'{a.runs_dir}/{run_name}'
os.makedirs(run_dir, exist_ok=True)

from vllm import LLM, SamplingParams  # noqa: E402

t_load = time.time()
llm = LLM(model=a.model, max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_util,
          kv_cache_dtype=a.kv_dtype, limit_mm_per_prompt={'video': 1, 'image': 0},
          max_num_seqs=128, enable_prefix_caching=True,
          mm_processor_kwargs={'do_sample_frames': False})
t_load = time.time() - t_load
sp = SamplingParams(temperature=0, max_tokens=a.max_tokens, repetition_penalty=1.05)
text = llm.get_tokenizer().apply_chat_template(
    [{'role': 'user', 'content': [{'type': 'video'}, {'type': 'text', 'text': prompt_tpl.replace('{dur}', str(round(a.win / a.fps)))}]}],
    add_generation_prompt=True, tokenize=False)
meta_video = {'fps': a.fps, 'frames_indices': list(range(a.win)), 'total_num_frames': a.win,
              'duration': a.win / a.fps, 'video_backend': 'pyav'}


def load(f):
    im = Image.open(f).convert('RGB')
    return np.asarray(im.crop((0, a.crop_top, im.width, im.height)).resize((a.w, a.h)))


def parse(txt):
    m = re.search(r'\{.*\}', txt, re.S)
    try:
        return json.loads(m.group(0)), True
    except Exception:
        return {'raw': txt}, False


meta = {'run': run_name, 'model': a.model, 'args': vars(a), 'prompt': prompt_tpl,
        'gpu': subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip(),
        'python': platform.python_version(), 'model_load_s': round(t_load, 1), 'videos': {}}
pool = ThreadPoolExecutor(16)
t_all = time.time()
for c in range(0, len(videos), a.chunk):
    batch, reqs = [], []
    t0 = time.time()
    for vdir in videos[c:c + a.chunk]:
        fs = sorted(glob.glob(f'{vdir}/*.webp') or glob.glob(f'{vdir}/*.jpg'))
        frames = np.stack(list(pool.map(load, fs)))  # mỗi frame chỉ đọc 1 lần
        starts = list(range(0, len(fs) - a.win + 1, a.stride))
        batch.append((vdir, fs, starts))
        reqs += [{'prompt': text, 'multi_modal_data': {'video': (frames[s:s + a.win], meta_video)}} for s in starts]
    t1 = time.time()
    outs = iter(llm.generate(reqs, sp))
    t2 = time.time()
    for vdir, fs, starts in batch:
        name = os.path.basename(vdir)
        ok = ntok = 0
        with open(f'{run_dir}/{name}.jsonl', 'w') as fo:
            for s in starts:
                o = next(outs).outputs[0]
                res, good = parse(o.text)
                ok += good; ntok += len(o.token_ids)
                fo.write(json.dumps({'video': name, 'start_idx': s,
                                     'start_frame': os.path.basename(fs[s]).split('.')[0],
                                     'end_frame': os.path.basename(fs[s + a.win - 1]).split('.')[0],
                                     'start_sec': s / a.fps, 'end_sec': (s + a.win) / a.fps, **res}, ensure_ascii=False) + '\n')
        meta['videos'][name] = {'windows': len(starts), 'json_ok': ok, 'gen_tokens': ntok}
    n = sum(len(x[2]) for x in batch)
    print(f'[{c + len(batch)}/{len(videos)}] {", ".join(os.path.basename(x[0]) for x in batch)} | {n} cửa sổ | '
          f'đọc ảnh {t1 - t0:.1f}s | sinh {t2 - t1:.1f}s', flush=True)
    meta['total_gen_s'] = round(time.time() - t_all, 1)
    json.dump(meta, open(f'{run_dir}/meta.json', 'w'), ensure_ascii=False, indent=1)  # ghi sau mỗi chunk, dừng giữa chừng vẫn giữ kết quả
print(f'Xong: {run_dir}  ({meta["total_gen_s"]}s, không tính {t_load:.0f}s nạp model)')
