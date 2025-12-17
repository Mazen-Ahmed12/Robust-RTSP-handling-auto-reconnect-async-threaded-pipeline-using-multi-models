from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
import uvicorn
from rtsp_live import rtsp_generator, shutdown_executor
from model_manager import get_model_manager
import atexit

app = FastAPI(title="robust Rtsp Live Fall Detection using multiple models")


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <h1>Robust RTSP Live Fall Detection with multi models</h1>
    
    <label for="model-select">Choose Detection Model: </label>
    <select id="model-select" onchange="switchModel(this.value)">
        <option value="yolo">YOLOv11 (Default)</option>
        <option value="pi">Raspberry Pi Optimized</option>
        <option value="jetson">NVIDIA Jetson (TensorRT)</option>
        <option value="cpu">CPU Lightweight</option>
    </select>
    <p><strong>Current Model:</strong> <span id="current-model">YOLOv11</span></p>

    <br><br>
    <img src="/live" width="960" id="video-feed">

    <script>
        function switchModel(modelName) {
            fetch('/set_model', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({model: modelName})
            }).then(() => {
                document.getElementById('current-model').textContent = 
                    modelName.toUpperCase();
            });
            // Optional: reload stream to apply instantly
            document.getElementById('video-feed').src = '/live?' + new Date().getTime();
        }
    </script>
    """


@app.post("/set_model")
async def set_model(request: Request):
    data = await request.json()
    model_type = data.get("model", "yolo")
    if model_type not in ["yolo", "pi", "jetson", "cpu"]:
        return {"error": "Invalid model"}

    get_model_manager().load_model(model_type)
    return {"status": "success", "model": model_type}


@app.get("/live")
def live():
    return StreamingResponse(
        rtsp_generator(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
    atexit.register(shutdown_executor)
