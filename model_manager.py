# model_manager.py
import os
import torch
from ultralytics import YOLO
import threading

# Base directory — change this to your project root if needed
BASE_DIR = os.path.dirname(
    os.path.dirname(__file__)
)  # goes one level up from this file

MODELS_DIR = os.path.join(
    BASE_DIR,
    "D:\mazen\EagleVision\projects\studing-projects\Robust-RTSP-handling-auto-reconnect-async-threaded-pipeline-using-multi-models\models",
)


class ModelManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.current_model = None
            cls._instance.current_name = "yolo"
            cls._instance.load_model("yolo")  # default
        return cls._instance

    def load_model(self, model_type: str):
        with self._lock:
            print(f"[MODEL] Loading {model_type.upper()}...")

            config = {
                "yolo": {
                    "path": os.path.join(MODELS_DIR, "yolo11n.pt"),  # .pt in models/
                    "device": "cuda" if torch.cuda.is_available() else "cpu",
                },
                "pi": {
                    "path": os.path.join(
                        MODELS_DIR, "yolov11s.onnx"
                    ),  # .onnx in models/
                    "device": "cpu",
                },
                "jetson": {
                    "path": os.path.join(
                        MODELS_DIR, "yolov11s.onnx"
                    ),  # .engine in models/
                    "device": 0,
                },
                "cpu": {
                    "path": "yolov11s_openvino_model",  # whole openvino/ folder
                    "device": "cpu",
                },
            }

            cfg = config.get(model_type, config["yolo"])
            path = cfg["path"]

            if not os.path.exists(path):
                print(f"[MODEL] WARNING: Path not found → {path}")
                print("[MODEL] Falling back to default yolo11n.pt")
                path = os.path.join(MODELS_DIR, "yolo11n.pt")

            print(f"[MODEL] Loading file/folder: {path}")
            self.current_model = YOLO(path)  # Ultralytics handles all formats perfectly

            if model_type != "jetson":
                self.current_model.to(cfg["device"])

            self.current_name = model_type
            print(f"[MODEL] {model_type.upper()} loaded successfully!")

    def get_model(self):
        return self.current_model

    def get_name(self):
        return self.current_name


def get_model_manager():
    return ModelManager()
