import cv2
import uvicorn
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse

from io import BytesIO
import PIL

from Human_seg import Human_Semantic

app = FastAPI()
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
templates = Jinja2Templates(directory="templates")

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.get('/')
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get('/video_feed')
def video_feed():
    return StreamingResponse(gen_frames(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.post('/process_image')
def process_image():
    success, frame = camera.read()
    if success:
        model=Human_Semantic(model_path="weights/SGHM-ResNet50.pth")
        image_pil = PIL.Image.fromarray(frame)
        result,alpha_np=model.infer(image_pil)

        # Create a BytesIO object to hold the modified image
        image_buffer = BytesIO()

        # Save the modified image to the BytesIO buffer in JPEG format
        result.save(image_buffer, format='JPEG')

        # Reset the buffer's position to the beginning
        image_buffer.seek(0)

        # Return the modified image as a response
        return StreamingResponse(image_buffer, media_type='image/jpeg')

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000, debug=True)