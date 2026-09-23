#!/usr/bin/env bash
# Chạy toàn bộ pipeline cho 1 video: phát hiện -> theo dõi (VLM phân xử) -> VLM kiểm tra từng lần ghép ID -> mô tả.
# Dùng: bash objtrack/run.sh /content/traffic/data/N098/N098-V001 [số_giây] [tên_run]
set -e
VID=${1%/}; SEC=${2:-30}; NAME=$(basename $VID); CAM=${NAME%%-*}
ROOT=$(cd "$(dirname "$0")/.." && pwd); WORK=${WORK:-/content/traffic/work}; mkdir -p $WORK
MODEL=${MODEL:-Qwen/Qwen3-VL-32B-Instruct-FP8}
RUN=${3:-$(date +%Y%m%d-%H%M%S)_objtrack}
source $ROOT/scripts/env.sh
cd $ROOT/objtrack
t0=$(date +%s)
python detect.py $VID $WORK/${NAME}_det.json --max_seconds $SEC;                                   t1=$(date +%s)
python track.py $VID $WORK/${NAME}_det.json $WORK/${NAME}_trk.json --vlm $MODEL 2>&1 | grep " ID |";  t2=$(date +%s)
python verify_links.py $VID $WORK/${NAME}_trk.json $WORK/${NAME}_trkv.json --model $MODEL 2>&1 | grep "lần ghép"; t3=$(date +%s)
python pipeline.py $VID $WORK/${NAME}_trkv.json $ROOT/configs/$CAM.json --model $MODEL --max_seconds $SEC --run_name $RUN 2>&1 | grep -E "đối tượng|VLM loại|tự tìm|Xong"; t4=$(date +%s)
echo "[$NAME] thời gian: phát hiện $((t1-t0))s | theo dõi $((t2-t1))s | kiểm tra ID $((t3-t2))s | mô tả $((t4-t3))s"
