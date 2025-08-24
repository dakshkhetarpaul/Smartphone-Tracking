# Smartphone-Tracking
Demo:
https://drive.google.com/file/d/1hwEa6UqJXOwwfJX7DdBAUu0BiEpMh7Q6/view?usp=sharing

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
  --input "sample/sample1.mp4" \
  --output "result/sample1_annotated.mp4" \
  --log "usage/sample1_usage.csv" \
  --report "usage/sample1_report.json" \
  --model yolov8l.pt \
  --conf 0.25 \
  --imgsz 768 \
  --device mps \
  --loose 
