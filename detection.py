# detection.py
import cv2
import torch
import numpy as np
from model_manager import get_model_manager


def detect_falls_in_frame(frame, frame_id):
    with torch.no_grad():
        model_manager = get_model_manager()
        model = model_manager.get_model()
        detections = []

        # Auto-detect if we can use .track() (only for .pt) or fallback to .predict()
        # Smart way: Try .track() first — it fails gracefully on ONNX/Engine/OpenVINO
        try:
            results = model.track(
                frame,
                persist=True,
                classes=[0],
                device=model.device if hasattr(model, "device") else "cuda",
                verbose=False,
            )[0]
        except Exception as e:
            # Fallback to .predict() if tracking not supported (ONNX, Engine, OpenVINO)
            print(f"[DETECTION] Tracking not supported, using predict(): {e}")
            results = model.predict(
                frame,
                classes=[0],
                device=model.device if hasattr(model, "device") else "cuda",
                verbose=False,
            )[0]
        annotated_frame = results.plot()

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            track_id = int(box.id.item()) if box.id is not None else 0

            height = y2 - y1
            width = x2 - x1

            if height - width < 0:
                fall_msg = f"FALL DETECTED: Track {track_id} | Frame {frame_id}"
                detection = {
                    "track_id": float(track_id),
                    "frame_id": float(frame_id),
                    "fall_message": fall_msg,
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                }
                detections.append(detection)

                cv2.rectangle(
                    annotated_frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 0, 255),
                    3,
                )
                cv2.putText(
                    annotated_frame,
                    "FALL!",
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                )

        return annotated_frame, detections
