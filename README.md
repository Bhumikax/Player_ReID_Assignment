# 🎯 Player Re-Identification in a Single Feed

This project implements real-time player detection and re-identification in a 15-second sports video. The objective is to assign consistent player IDs throughout the video — even if players briefly exit and re-enter the frame.

---

## 📁 Project Structure

Player_ReID_Assignment/
├── detect_and_track.py # Main script: Detection + Tracking
├── 15sec_input_720p.mp4 # Input video (provided)
├── weights/
│ └── best.pt # YOLOv8 model file (provided)
├── results/
│ └── output_video_with_ids.mp4 # Final output with tracked IDs
├── README.md
└── report.pdf


---

## 🧠 Methodology

- **YOLOv8**: Used for detecting players in each frame.
- **DeepSORT**: Used for tracking players and assigning unique IDs.
- Players who leave and reappear in the frame are tracked with consistent IDs using temporal and spatial cues.

---

## ⚙️ Setup Instructions

🔹 1. Install Dependencies

```bash
pip install ultralytics opencv-python deep_sort_realtime

🔹 2.Download the YOLOv8 Model

Download the model file from the link provided in the assignment, rename it to best.pt, and place it in the weights/ folder.

🔹 3. Run the Program

python detect_and_track.py

📌 Notes
The YOLOv8 model is pre-trained to detect both players and the ball.
DeepSORT’s max_age parameter is tuned to allow temporary disappearance before reassigning IDs.
Ensure the input video 15sec_input_720p.mp4 is in the root directory.

📤 Submission Includes
✅ detect_and_track.py – detection + tracking code
✅ weights/best.pt – model file (not opened manually)
✅ results/output_video_with_ids.mp4 – output with ID overlays
✅ report.pdf – explains methodology and challenges
✅ README.md – this file

🤝 Credits
Ultralytics YOLOv8
Deep SORT Real-time
