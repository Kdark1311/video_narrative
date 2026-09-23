# video_narrative

Sinh mô tả (video caption) chi tiết cho từng đoạn nhỏ trong video camera giao thông cố định, phục vụ tìm kiếm bằng câu mô tả
(vd: *"ô tô màu xanh đi qua ngã tư khi đèn đang đỏ"*) và tìm sự kiện bất thường. Không OCR.

## Cách làm
- Dữ liệu: keyframe 1 frame/giây, 1920×1080 (`N091`–`N100`, mỗi camera 3 video ~600 frame ≈ 10 phút).
- Cắt mỗi video thành **cửa sổ 20 frame (20 giây), trượt 10 frame** → ~59 cửa sổ/video.
- Mỗi cửa sổ đưa vào **Qwen3-VL** ở chế độ video (có mốc thời gian từng giây), resize 1280×720, cắt dải chữ chèn phía trên.
- Prompt: [`prompts/traffic_caption_vi.txt`](prompts/traffic_caption_vi.txt) → JSON gồm `boi_canh`, `mat_do_giao_thong`,
  `den_giao_thong`, `su_kien[]` (theo giây), `bat_thuong[]`, `caption`.
- Chạy bằng **vLLM**, gom nhiều video vào một lần `generate` để GPU luôn đầy.

## Chạy trên Colab
```bash
bash scripts/setup_colab.sh "<link Google Drive file zip>"   # cài vLLM + nvcc 13, tải & giải nén vào /content/traffic/data
source scripts/env.sh
python src/caption_vllm.py /content/traffic/data --model Qwen/Qwen3-VL-32B-Instruct-FP8
```
Mỗi lần chạy lưu vào `outputs/runs/<thời gian>_<model>/`: `meta.json` (model, tham số, prompt, GPU, thời gian) + `<video>.jsonl` (mỗi dòng 1 cửa sổ).

## Benchmark (1 video N091-V001 = 59 cửa sổ, RTX PRO 6000 Blackwell 96GB)
| Model | Engine | Thời gian sinh | Ghi chú |
|---|---|---|---|
| Qwen3-VL-32B-Instruct (bf16) | transformers | ~45–70 s/cửa sổ (~55 phút/video) | ước tính từ 7 cửa sổ mẫu |
| Qwen3-VL-32B-Instruct (bf16) | vLLM | 422 s | KV cache chỉ ~20GB |
| Qwen3-VL-32B-Instruct-FP8 | vLLM | 226 s | chất lượng ≈ bf16, chi tiết nhất |
| Qwen3-VL-30B-A3B-Instruct-FP8 | vLLM | 65 s | nhanh nhất, mô tả ít chi tiết hơn (ít sự kiện, hay bỏ sót đổi màu đèn) |


## Pipeline v4: theo dõi từng đối tượng (`objtrack/`)
Mô tả **từng đối tượng** (mỗi xe/người một ID) trong suốt thời gian nó xuất hiện, rồi mới ghép thành mô tả đoạn 10 giây.
Ưu tiên: **nhận diện đúng và đủ các đối tượng rõ ràng → theo dõi đúng ID → mô tả đặc trưng + chuỗi hành động**, không để LLM tự bịa.

```bash
source scripts/env.sh
bash objtrack/run.sh /content/traffic/data/N098/N098-V001 30      # 30 giây đầu; bỏ số giây để chạy cả video
```

| Bước | File | Cách làm |
|---|---|---|
| 1. Phát hiện | `detect.py` | YOLO11m-VisDrone + SAHI (cắt ô 640) cho xe; YOLO11x-COCO cho người và đèn giao thông |
| 2. Theo dõi | `track.py` | Chỉ tạo ID khi đối tượng **đã hiện rõ** (khung ≥ 45px, conf ≥ 0.45). Ghép bằng DINOv2 + histogram màu + vị trí dự đoán + IoU; phải giống cả ảnh mẫu lúc rõ nhất. Xe 2 bánh không ghép qua khoảng mất dấu. Trường hợp mơ hồ → **Qwen3-VL phân xử** (ảnh có khung đỏ, xác suất ≥ 0.85) |
| 3. Kiểm tra ID | `verify_links.py` | Qwen3-VL kiểm tra **từng lần ghép** giữa 2 giây liên tiếp, xác suất "cùng đối tượng" < 0.85 → cắt ID |
| 4. Mô tả | `pipeline.py` | Chuỗi hành động từng ID tính bằng hình học (vùng, hướng, dừng/chạy, màu đèn); ngoại hình do VLM mô tả từ ảnh cắt (kèm cờ hợp lệ để loại cọc, chòi, vạch sơn); mô tả đoạn do VLM viết từ frame có vẽ ID + ảnh phóng to + bảng dữ kiện |

**Cấu hình camera (tuỳ chọn)** `configs/<camera>.json`: vị trí đèn + hướng đèn điều khiển, các vùng (làn tới, giao lộ, đảo).
Có cấu hình → đọc màu đèn chính xác và bắt **vượt đèn đỏ** bằng luật. Không có → tự tìm đèn bằng YOLO (chỉ để mô tả),
**đi ngược chiều** được phát hiện bằng trường hướng lưu thông học từ chính video (không cần khai báo).

Output mỗi video trong `outputs/runs/<run>/`: `<video>.md` (dễ đọc), `<video>_objects.jsonl` (mỗi dòng 1 đối tượng: ngoại hình,
chuỗi hành động, bất thường, frame, khung), `<video>_segments.jsonl` (mô tả từng đoạn 10s, chồng 1s).

**Giới hạn đã biết** (dữ liệu chỉ 1 frame/giây): xe máy chạy nhanh trong dòng đông chỉ giữ được ID ngắn 1–2 giây (bị cắt thay vì tráo ID);
không kết luận "không đội mũ bảo hiểm" (đã kiểm chứng VLM hay nhầm, vd mũ cam bị coi là không đội mũ) — cần model chuyên dụng + ảnh gốc độ phân giải cao hơn; cảnh đêm mưa còn một ít ID lẫn.

## Lỗi đã biết của prompt v1
- Hay báo nhầm **"đi ngược chiều"** trên đường hai chiều (nhất là cảnh đêm mưa).
- Báo "vượt đèn đỏ" khi đèn ở quá xa để xác định.
- Đèn nhỏ ở xa đôi khi bị bỏ sót ("không thấy đèn").

## Ghi chú môi trường (Colab, GPU SM 12.0)
- `pip install vllm` nâng torch lên 2.13+cu130 → cần torchaudio bản cu130.
- nvcc của Colab là 12.8; FlashInfer JIT cần ≥ 12.9 → cài `nvidia-cuda-nvcc` 13 qua pip (`scripts/env.sh` đặt `CUDA_HOME`).
- Tắt DeepGEMM và sampler FlashInfer (`VLLM_USE_DEEP_GEMM=0`, `VLLM_USE_FLASHINFER_SAMPLER=0`); KV cache FP8 chưa build được nên để `auto`.
