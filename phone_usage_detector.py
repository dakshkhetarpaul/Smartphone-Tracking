import argparse, os, sys, json, shutil, subprocess, math, csv
from pathlib import Path
import cv2
import numpy as np

# --- Models ---
# YOLO for objects (person/cell phone)
from ultralytics import YOLO

# MediaPipe for hands + face
import mediapipe as mp
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection

# -----------------------
# Geometry / utils
# -----------------------
def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    return (x1, y1, x2 - x1, y2 - y1)

def bbox_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter == 0: return 0.0
    a_area = (ax2 - ax1) * (ay2 - ay1)
    b_area = (bx2 - bx1) * (by2 - by1)
    return inter / (a_area + b_area - inter + 1e-6)

def expand_box(box, scale, W, H):
    x1, y1, x2, y2 = box
    cx, cy = (x1+x2)/2, (y1+y2)/2
    w, h = (x2-x1)*scale, (y2-y1)*scale
    nx1, ny1 = max(0, cx - w/2), max(0, cy - h/2)
    nx2, ny2 = min(W-1, cx + w/2), min(H-1, cy + h/2)
    return (nx1, ny1, nx2, ny2)

def point_to_box_dist(px, py, box):
    x1, y1, x2, y2 = box
    cx = min(max(px, x1), x2)
    cy = min(max(py, y1), y2)
    return math.hypot(px - cx, py - cy)

def draw_box_label(img, box, label, color=(0, 255, 0), thickness=2):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    ((tw, th), _) = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(img, label, (x1 + 3, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

# -----------------------
# Handheld decision logic
# -----------------------
def phone_is_handheld(phone_box, phone_conf, hands_landmarks, face_boxes, person_boxes, frame_shape):
    """
    Returns (bool is_handheld, str reason)
    """
    H, W = frame_shape[:2]
    pcx, pcy = bbox_center(phone_box)
    diag = math.hypot(W, H)
    # dynamic distance thresholds scale with frame size
    near_hand_thresh = 0.06 * diag
    near_face_expand = 1.4  # expand face box to count "near face"
    near_lap_margin = 0.05 * H  # small margin below lower-third boundary
    near_hand_for_lap = 0.08 * diag

    # 1) Near any hand landmark?
    for lm in hands_landmarks:
        for (hx, hy) in lm:  # pixel coords
            if math.hypot(pcx - hx, pcy - hy) <= near_hand_thresh:
                return True, "near hand"

    # 2) Near face (overlap with expanded face box)
    for fb in face_boxes:
        expanded = expand_box(fb, near_face_expand, W, H)
        if iou(phone_box, expanded) > 0:
            return True, "near face"

    # 3) On lap: phone center inside lower third of any person bbox AND a hand is nearby
    for pb in person_boxes:
        x1, y1, x2, y2 = pb
        lower_third_y = y1 + (2.0/3.0) * (y2 - y1)
        in_lower_third = (pcx >= x1 and pcx <= x2 and pcy >= lower_third_y - near_lap_margin and pcy <= y2 + near_lap_margin)
        if in_lower_third:
            # is there a nearby hand as well?
            for lm in hands_landmarks:
                for (hx, hy) in lm:
                    if math.hypot(pcx - hx, pcy - hy) <= near_hand_for_lap:
                        return True, "on lap (near hand)"

    # Otherwise: treat as static/unused
    return False, "static/unused"

# -----------------------
# MediaPipe helpers
# -----------------------
def get_hands_and_faces(frame_bgr, hands_ctx, face_ctx):
    H, W = frame_bgr.shape[:2]
    # MediaPipe expects RGB
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    hands_landmarks = []
    hr = hands_ctx.process(img_rgb)
    if hr.multi_hand_landmarks:
        for handLms in hr.multi_hand_landmarks:
            pts = []
            for lm in handLms.landmark:
                px = int(lm.x * W)
                py = int(lm.y * H)
                pts.append((px, py))
            hands_landmarks.append(pts)

    face_boxes = []
    fr = face_ctx.process(img_rgb)
    if fr.detections:
        for det in fr.detections:
            # bounding box in relative coords
            r = det.location_data.relative_bounding_box
            x1 = int(r.xmin * W)
            y1 = int(r.ymin * H)
            x2 = int((r.xmin + r.width) * W)
            y2 = int((r.ymin + r.height) * H)
            # clamp
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W - 1, x2), min(H - 1, y2)
            if x2 > x1 and y2 > y1:
                face_boxes.append((x1, y1, x2, y2))

    return hands_landmarks, face_boxes

# -----------------------
# Audio preservation
# -----------------------
def attach_original_audio(src_path, silent_video_path, out_path):
    """
    Mux the original audio track into the annotated video using ffmpeg.
    If the source has no audio or ffmpeg is missing, falls back to the silent video.
    """
    if shutil.which("ffmpeg") is None:
        print("[WARN] ffmpeg not found on PATH; output will be silent.")
        shutil.copyfile(silent_video_path, out_path)
        return

    # Try copy codecs to avoid re-encode. If source has no audio, command fails; catch and fallback.
    cmd = [
        "ffmpeg", "-y",
        "-i", silent_video_path,
        "-i", src_path,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "copy",
        "-shortest",
        out_path
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        print("[WARN] Audio mux failed (no audio track or incompatible). Writing silent video.")
        shutil.copyfile(silent_video_path, out_path)

# -----------------------
# Main processing
# -----------------------
def process_video(args):
    inp = str(args.input)
    cap = cv2.VideoCapture(inp)
    if not cap.isOpened():
        print(f"[ERR] Could not open input: {inp}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or math.isnan(fps):
        fps = args.fps if args.fps else 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output writer (silent video first; we’ll remux audio later)
    tmp_silent = str(Path(args.output).with_suffix(".silent.mp4"))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_silent, fourcc, fps, (W, H))
    if not writer.isOpened():
        print("[ERR] Could not open writer.")
        sys.exit(1)

    # Load YOLO
    model = YOLO(args.model)
    device = args.device if args.device != "auto" else ( "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else ("mps" if (sys.platform=="darwin") else "cpu") )

    # MediaPipe contexts
    hands_ctx = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    face_ctx = mp_face.FaceDetection(min_detection_confidence=0.5, model_selection=0)

    # Logging structures
    usage_events = []   # (start_frame, end_frame, mean_conf)
    active = False
    event_start = None
    conf_accum = []

    frame_idx = 0
    phone_label = "cell phone"
    person_label = "person"

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Get hands & faces first (RGB inside helper)
            hands_landmarks, face_boxes = get_hands_and_faces(frame, hands_ctx, face_ctx)

            # YOLO inference
            # We keep native resolution; imgsz can downscale for speed (e.g., 640) if needed.
            results = model.predict(
                source=frame,
                conf=args.conf,
                iou=0.5,
                imgsz=args.imgsz,
                device=device,
                verbose=False
            )[0]

            # Collect phones and persons
            phone_dets = []
            person_boxes = []

            names = results.names
            for b in results.boxes:
                cls_id = int(b.cls[0].item())
                conf = float(b.conf[0].item())
                x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
                box = (max(0, x1), max(0, y1), min(W-1, x2), min(H-1, y2))
                cls_name = names.get(cls_id, str(cls_id))
                if cls_name == phone_label:
                    phone_dets.append((box, conf))
                elif cls_name == person_label:
                    person_boxes.append(box)

            # Decide handheld vs static; draw
            any_handheld = False
            for (pbox, pconf) in phone_dets:
                is_handheld, reason = phone_is_handheld(pbox, pconf, hands_landmarks, face_boxes, person_boxes, frame.shape)
                if is_handheld:
                    any_handheld = True
                    draw_box_label(frame, pbox, f"PHONE {pconf:.2f} (handheld: {reason})", (0, 255, 0), 3)
                    conf_accum.append(pconf)
                else:
                    # We purposely do not draw static/unused phones to avoid clutter.
                    pass

            # Update event timeline
            if any_handheld and not active:
                active = True
                event_start = frame_idx
            elif (not any_handheld) and active:
                active = False
                if event_start is not None:
                    # close event
                    if len(conf_accum) == 0:
                        mean_c = 0.0
                    else:
                        mean_c = float(np.mean(conf_accum))
                    usage_events.append((event_start, frame_idx - 1, mean_c))
                    conf_accum = []

            # Write annotated frame
            writer.write(frame)
            frame_idx += 1

        # close last open event
        if active and event_start is not None:
            mean_c = float(np.mean(conf_accum)) if len(conf_accum) else 0.0
            usage_events.append((event_start, frame_idx - 1, mean_c))

    finally:
        cap.release()
        writer.release()
        hands_ctx.close()
        face_ctx.close()

    # Remux original audio into annotated video
    attach_original_audio(inp, tmp_silent, str(args.output))
    try:
        os.remove(tmp_silent)
    except Exception:
        pass

    # Write logs/reports
    if args.log:
        with open(args.log, "w", newline="") as f:
            cw = csv.writer(f)
            cw.writerow(["start_frame", "end_frame", "start_time_sec", "end_time_sec", "duration_sec", "mean_confidence"])
            for s, e, mc in usage_events:
                st = s / fps
                et = e / fps
                cw.writerow([s, e, f"{st:.3f}", f"{et:.3f}", f"{(et-st):.3f}", f"{mc:.4f}"])

    if args.report:
        total_frames = max(1, frame_idx)
        total_sec = total_frames / fps
        used_sec = sum([(e - s) / fps for s, e, _ in usage_events])
        report = {
            "video_path": os.path.abspath(inp),
            "output_path": os.path.abspath(args.output),
            "fps": fps,
            "resolution": {"width": W, "height": H},
            "episodes": len(usage_events),
            "total_video_seconds": round(total_sec, 3),
            "total_phone_use_seconds": round(used_sec, 3),
            "phone_use_percent": round(100.0 * used_sec / total_sec, 2) if total_sec > 0 else 0.0
        }
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2)

    print(f"[DONE] Wrote annotated video to: {args.output}")
    if args.log: print(f"[DONE] Usage log: {args.log}")
    if args.report: print(f"[DONE] Summary report: {args.report}")

def valid_path(p):
    return Path(p)

def main():
    ap = argparse.ArgumentParser(description="Handheld Phone Usage Detector (Video)")
    ap.add_argument("--input", type=valid_path, required=True, help="Input video path (.mp4/.avi/.mov)")
    ap.add_argument("--output", type=valid_path, required=True, help="Output annotated video (.mp4)")
    ap.add_argument("--log", type=valid_path, default=None, help="Optional CSV log path")
    ap.add_argument("--report", type=valid_path, default=None, help="Optional JSON summary report path")
    ap.add_argument("--model", type=str, default="yolov8n.pt", help="Ultralytics YOLO model path or name")
    ap.add_argument("--conf", type=float, default=0.35, help="Detections confidence threshold")
    ap.add_argument("--imgsz", type=int, default=640, help="Inference image size (smaller = faster)")
    ap.add_argument("--device", type=str, default="auto", help="cuda | cpu | mps | auto")
    ap.add_argument("--fps", type=float, default=None, help="Override FPS if unreadable")
    args = ap.parse_args()
    process_video(args)

if __name__ == "__main__":
    main()
