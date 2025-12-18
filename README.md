# Robust RTSP Handling Pipeline (Multi-Model)

An advanced RTSP/IP camera pipeline with auto-reconnect, async/threaded design, and support for multiple detection models.

## Features
- Auto-reconnect logic for unstable streams
- Async + multi-threaded architecture
- Supports multiple models (YOLO, pose, etc.)
- Database integration for tracking and analysis

## Tech Stack
- Python (asyncio, threading)
- OpenCV
- YOLO (Ultralytics)
- PostgreSQL / MongoDB

## Use Case
Designed for scalable surveillance systems and real-time video analytics.

## Installation & Launch

- if you have a good gpu and can handle the project you can do all this in you local machine environment else do it on anaconda or anytihng else like it

1. Clone the repo:
   ```bash
   git clone https://github.com/Mazen-Ahmed12/people-fall-detection-using-YOLOv11-Pose-Deepsort.git
   cd people-fall-detection-using-YOLOv11-Pose-Deepsort

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate # Linux/Mac
   venv\Scripts\activate # Windows
   pip install -r requirements.txt
3. Run the detection script:
   ```bash
   python detect_fall.py

## note 
  - dont forget to change the directories of the model and any other directory in the project
  - and dont forget to run mongodb database before running the project
  - and the models you can use whatever model you want even a custome one but dont forget the model extension
  - and dont forget to change the RTSP_URL to the cam url 
