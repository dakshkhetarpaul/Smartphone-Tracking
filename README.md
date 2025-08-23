# Smartphone-Tracking

python3 phone_usage_detector.py \
  --input "sample1.mp4" \
  --output "sample1_annotated.mp4" \
  --log "sample1_usage.csv" \
  --report "sample1_report.json" \
  --model yolov8n.pt \
  --conf 0.25 \
  --imgsz 768 \
  --device mps
