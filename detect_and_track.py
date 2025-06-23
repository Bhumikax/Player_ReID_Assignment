import cv2
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import os

# Create results folder
os.makedirs("results", exist_ok=True)

# Load model
model = YOLO("weights/best.pt")

# Initialize DeepSORT
tracker = DeepSort(max_age=15)

# Video input/output
video_path = "15sec_input_720p.mp4"
cap = cv2.VideoCapture(video_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter("results/output_video_with_ids.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

# Frame limiting logic
max_frames = 100  # Limit to 100 frames
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)[0]

    detections = []

    for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        label = int(cls)
        # Only track players (label 1)
        if label == 1:
            x1, y1, x2, y2 = box.tolist()
            confidence = float(conf)
            detections.append(([x1, y1, x2 - x1, y2 - y1], confidence, 'player'))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        l, t, w, h = track.to_ltrb()
        l, t, r, b = int(l), int(t), int(l + w), int(t + h)
        cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    out.write(frame)

    frame_count += 1
    if frame_count >= max_frames:
        print(f"🛑 Reached max_frames ({max_frames}). Stopping...")
        break

cap.release()
out.release()
print("✅ Done! Output saved at: results/output_video_with_ids.mp4")
