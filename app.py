import freenect
import cv2, PIL, uvicorn
from io import BytesIO
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from scan3d import DensePose
import numpy as np
import os, sys, math

app = FastAPI()
templates = Jinja2Templates(directory="templates")


# 1. Khởi tạo các mô hình
root_path = os.getcwd()

# 1.1. Khởi tạo Seg Model
seg_path = os.path.join(root_path, "Human_Segmantation")
sys.path.append(seg_path)
from Human_seg import Human_Semantic

seg_model = Human_Semantic("Human_Segmantation/weights/SGHM-ResNet50.pth")

# 1.2. Khởi tạo Pose Model
pose_path = os.path.join(root_path, "Human_Pose")
sys.path.append(pose_path)
from Human_pose import Human_Pose

pose_model = Human_Pose("Human_Pose/weights/checkpoint_iter_370000.pth")

# 1.3. Khởi tạo Scan Model
scan_model = DensePose(
    model_path="./weights/model_final_162be9.pkl",
    model_config_path="detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml",
)


# 2. Tạo các class chuyên biệt


# 2.1 Tạo class HeightMeasure để tính chiều cao
class HeightMeasure:
    def __init__(self, d=285, f=650.1):
        self.d = d
        self.f = f

    def calculate_box(self, boxes):
        for box in boxes:
            x1, y1, x2, y2 = box
            w = abs(y2 - y1)
            return w * self.d / self.f

    def calculate_point(self, point1, point2):
        w = math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
        return w * self.d / self.f


measure = HeightMeasure()


# 3. Tạo các function chuyên biệt


# 3.1. Tạo hàm tính khoảng cách 2 điểm
def distance2d(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


# 3.2. Tạo hàm lấy độ sâu của pixel từ cam
def get_depth(depth_image, x, y):
    depth_value = depth_image[y, x]
    z_meters = 1.0 / (depth_value * -0.0030711016 + 3.3309495161)
    return z_meters


# 3.3. Tạo các hàm tính độ rộng của vai
def compute_each_shoulder_length(img, coor):
    result = np.where(np.array(img) > 100, 255, 0)
    arr_sho = result[coor[1]]
    indexes = np.array(np.where(arr_sho == 255)).flatten()
    return indexes[-1] - indexes[0]


def get_max_shoulder_length(img, r_sho, l_sho):
    return max(
        compute_each_shoulder_length(img, r_sho),
        compute_each_shoulder_length(img, l_sho),
    )


# 4. Các biến toàn cục

# 4.1. Tạo result_dict để lưu các thông tin quan trọng
result_dict = {
    "height": 0,
    "ongchan": 0,
    "haidaugoi": 0,
    "chieudaithan": 0,
    "dolechvai": 0,
    "dosau": 0,
    "r_sho": 0,
    "l_show": 0,
    "keypoints": None,
    "hunchback_result": False,
}

# 4.2. Tạo các biến toàn cục
scale_on_web = (320, 320)


# 5. Tạo các hàm tương tác với FE


# 5.1. Định nghĩa việc tương tác với index.html
@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# 5.2. Hàm dùng để đẩy lần lượt các frames lên FE
def gen_frames():
    while True:
        frame = cv2.cvtColor(freenect.sync_get_video()[0], cv2.COLOR_BGR2RGB)
        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


# 5.3. Hàm xử lý Scan 3D
@app.post("/process_scan3d")
def process_scan3d():
    frame = cv2.cvtColor(cv2.imread("other/quang.jpg"), cv2.COLOR_BGR2RGB)
    # frame = cv2.cvtColor(freenect.sync_get_video()[0],cv2.COLOR_BGR2RGB)

    result, boxes = scan_model.inference(frame)
    ###
    # W = 183
    # d = 285
    # for box in boxes:
    #     x1, y1, x2, y2 = box
    #     w = abs(y2 - y1)
    #     print(w * d / W)
    ###

    # for box in boxes:
    #     x1, y1, x2, y2 = box
    #     mid_point=(int((x1+x2)/2),int((y1+y2)/2))
    #     depth = freenect.sync_get_depth(0,freenect.DEPTH_MM)[0]
    #     dosau = np.around(get_depth(depth, mid_point[0], mid_point[1]), 2)
    #     result_dict['dosau'] = dosau

    height = measure.calculate_box(boxes)
    result_dict["height"] = round(height, 2)
    print(type(result_dict["height"]))
    result = cv2.resize(result, scale_on_web)

    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    result = PIL.Image.fromarray(result)

    image_buffer = BytesIO()
    result.save(image_buffer, format="JPEG")
    image_buffer.seek(0)

    return StreamingResponse(image_buffer, media_type="image/jpeg")


# 5.4. Hàm xử lý Segmentation
@app.post("/process_seg")
def process_seg():
    frame = cv2.cvtColor(cv2.imread("other/quang.jpg"), cv2.COLOR_BGR2RGB)
    # frame = cv2.cvtColor(freenect.sync_get_video()[0],cv2.COLOR_BGR2RGB)
    frame = PIL.Image.fromarray(frame)
    result, _ = seg_model.infer(frame)  # (480, 640)
    result = np.array(result)

    # Đo chiều cao theo pixel:
    temp = np.where(result > 100, 255, 0)
    b = np.where(temp == 255)
    c1 = np.min(b)
    c2 = np.max(b)
    print("Chiều cao:", c2 - c1)

    dolechvai = get_max_shoulder_length(
        result, result_dict["r_sho"], result_dict["l_sho"]
    )
    result_dict["dolechvai"] = dolechvai

    result = cv2.resize(result, scale_on_web)

    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    result = PIL.Image.fromarray(result)

    image_buffer = BytesIO()
    result.save(image_buffer, format="JPEG")
    image_buffer.seek(0)

    return StreamingResponse(image_buffer, media_type="image/jpeg")


@app.post("/process_pose")
def process_pose():
    frame = cv2.cvtColor(cv2.imread("other/quang.jpg"), cv2.COLOR_BGR2RGB)
    # frame = cv2.cvtColor(freenect.sync_get_video()[0],cv2.COLOR_BGR2RGB)
    result, keypoints = pose_model.infer(frame)  # (480, 640)
    keypoints_list = keypoints.tolist()
    print(keypoints_list)
    result_dict["r_sho"] = list(map(int, keypoints_list[2]))
    result_dict["l_sho"] = list(map(int, keypoints_list[5]))
    """
    'nose', 'neck', 'r_sho', 'r_elb', 'r_wri', 
    'l_sho', 'l_elb', 'l_wri', 'r_hip', 'r_knee', 
    'r_ank', 'l_hip', 'l_knee', 'l_ank', 'r_eye', 
    'l_eye', 'r_ear', 'l_ear'
    """
    ######
    # Tính độ dài ống chân: trái - phải
    ongtrai = distance2d(keypoints[12], keypoints[13])
    ongphai = distance2d(keypoints[9], keypoints[10])
    ongchan = f"L: {ongtrai} - R: {ongphai}"
    result_dict["ongchan"] = ongchan
    print(ongchan)

    # Khoảng cách 2 đầu gối
    haidaugoi = distance2d(keypoints[9], keypoints[12])
    result_dict["haidaugoi"] = float(haidaugoi)
    print(haidaugoi)

    # Chiều dài thân
    midpoint2hip = np.array(
        [
            int((keypoints[8][0] + keypoints[11][0]) / 2),
            int((keypoints[8][1] + keypoints[11][1]) / 2),
        ]
    )
    chieudaithan = distance2d(midpoint2hip, keypoints[1])
    result_dict["chieudaithan"] = float(chieudaithan)
    ######

    # Có bị gù không?
    hunchback_threshold = 0.95
    midpoint2sho = np.array(
        [
            int((keypoints[2][0] + keypoints[5][0]) / 2),
            int((keypoints[2][1] + keypoints[5][1]) / 2),
        ]
    )
    midpoint2ear = np.array(
        [
            int((keypoints[16][0] + keypoints[17][0]) / 2),
            int((keypoints[16][1] + keypoints[17][1]) / 2),
        ]
    )
    hip_sho_vector = midpoint2sho - midpoint2hip
    sho_ear_vector = midpoint2ear - midpoint2sho
    hunchback_degree = np.dot(hip_sho_vector, sho_ear_vector) / (
        np.linalg.norm(hip_sho_vector) * np.linalg.norm(sho_ear_vector)
    )
    if hunchback_degree < hunchback_threshold:
        result_dict["hunchback_result"] = True
    print("hunchback_degree---------------------------------------", hunchback_degree)

    result = cv2.resize(result, scale_on_web)

    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    result = PIL.Image.fromarray(result)

    image_buffer = BytesIO()
    result.save(image_buffer, format="JPEG")
    image_buffer.seek(0)

    print("measure.d-----------------", measure.d)

    ######################################## Test type
    print(type(StreamingResponse(image_buffer, media_type="image/jpeg")))
    ########################################

    return StreamingResponse(image_buffer, media_type="image/jpeg")


# 5.6. Hàm đưa dữ liệu lên FE
@app.post("/process_data")
def process_data():
    return {
        "height": int(result_dict["height"]),
        "ongchan": result_dict["ongchan"],
        "haidaugoi": int(result_dict["haidaugoi"]),
        "chieudaithan": int(result_dict["chieudaithan"]),
        "dolechvai": int(result_dict["dolechvai"]),
    }


# 5.7. Hàm khởi tạo tham số d và f cho mô hình
class DFParam(BaseModel):
    d_param: str
    f_param: str


@app.post("/initMeasure")
def initMeasure(param: DFParam):
    d_param = float(param.d_param)
    f_param = float(param.f_param)
    measure.d = d_param
    measure.f = f_param
    print("d--------------", type(d_param))
    print("f--------------", type(f_param))


# 6. Hàm main
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, debug=True)
