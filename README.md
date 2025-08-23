# Smartphone-Tracking
problem in sample - 3,5.6 RGB distortion.
cd ~/Desktop/Smartphone-Tracking
source .venv311/bin/activate

for f in sample/*.mp4; do
  base=$(basename "$f" .mp4)
  python3 phone_usage_detector.py \
    --input "$f" \
    --output "result/${base}_annotated.mp4" \
    --log "usage/${base}_usage.csv" \
    --report "usage/${base}_report.json" \
    --model yolov8s.pt \
    --conf 0.25 \
    --imgsz 768 \
    --device mps \
    --loose
done


# for 1 vid
python3 phone_usage_detector.py \
  --input "sample/sample5.mp4" \
  --output "result/sample5_annotated.mp4" \
  --log "usage/sample1_usage.csv" \
  --report "usage/sample5_report.json" \
  --model yolov8s.pt \
  --conf 0.05 \
  --imgsz 768 \
  --device mps \
  --loose
