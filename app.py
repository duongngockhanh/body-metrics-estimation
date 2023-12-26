# import freenect
import cv2, PIL, uvicorn
from io import BytesIO
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from scan3d import DensePose
import numpy as np
import os, sys, math
import pandas as pd
from general_utils import *
import time


app = FastAPI()
templates = Jinja2Templates(directory="templates")





# ---------------------------- e1. Khởi tạo các mô hình ----------------------------
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

measure = HeightMeasure()


# ---------------------------- f1. Các biến toàn cục ----------------------------

scale_on_web = (320, 320)

kpt_names = ['nose', 'neck', 'r_sho', 'r_elb', 'r_wri', 
             'l_sho', 'l_elb', 'l_wri', 'r_hip', 'r_knee', 
             'r_ank', 'l_hip', 'l_knee', 'l_ank', 'r_eye', 
             'l_eye', 'r_ear', 'l_ear']

result_dict = {
    "height": 0,
    "keypoints": None,
    "seg_list": [],
    "pose_list": [],
    "scan_list": [],
    "view_list": [],
    "kpt_names": kpt_names,
    "information": {}
}




# ---------------------------- g1. Stream Video ----------------------------
# 5.1. Định nghĩa việc tương tác với index.html
@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


path = 0
cap = cv2.VideoCapture(path)


# 5.2. Hàm dùng để đẩy lần lượt các frames lên FE
def gen_frames():
    while True:
        _, frame = cap.read()
        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame"
    )





# ---------------------------- h1. Scan 3D ----------------------------
@app.post("/process_scan3d")
def process_scan3d():
    _, frame = cap.read()
    result, boxes = scan_model.inference(frame)
    result_dict["scan_list"].append(result)

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





# ---------------------------- i1. Seg Truoc ----------------------------
@app.post("/process_seg")
def process_seg():
    ret, frame = cap.read()
    result_dict["view_list"].append(frame)
    frame = PIL.Image.fromarray(frame)
    result, _ = seg_model.infer(frame)  # (480, 640)
    result = np.array(result)
    mask = np.where(result > 100, 255, 0).astype(np.uint8)
    mask = np.stack((mask, mask, mask), axis=2)
    result_dict["seg_list"].append(mask)
    # ---------------------

    DAy, BAz, CAx, img_draw = tinh_mom_vai(result_dict["seg_list"][0], result_dict["information"])
    result_dict['DAy'] = DAy
    result_dict['BAz'] = BAz
    result_dict['CAx'] = CAx

    kc_2_knee, kc_2_ank = tinh_khoang_cach_chan(result_dict["seg_list"][0], result_dict["information"])
    result_dict['kc_2_knee'] = kc_2_knee
    result_dict['kc_2_ank'] = kc_2_ank

    # ---------------------
    result = cv2.resize(img_draw, scale_on_web)
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    result = PIL.Image.fromarray(result)
    image_buffer = BytesIO()
    result.save(image_buffer, format="JPEG")
    image_buffer.seek(0)
    return StreamingResponse(image_buffer, media_type="image/jpeg")





# ---------------------------- i2. Seg Nghieng ----------------------------
@app.post("/process_seg_nghieng")
def process_seg_nghieng():
    _, frame = cap.read()
    result_dict["view_list"].append(frame)
    frame = PIL.Image.fromarray(frame)
    result, _ = seg_model.infer(frame)  # (480, 640)
    result = np.array(result)
    mask = np.where(result > 100, 255, 0).astype(np.uint8)
    mask = np.stack((mask, mask, mask), axis=2)
    result_dict["seg_list"].append(mask)
    # ---------------------

    CAz, ABz, img_draw = tinh_cot_song(result_dict["seg_list"][1], result_dict["information"])
    result_dict['CAz'] = CAz
    result_dict['ABz'] = ABz
    print(result_dict)

    # ---------------------
    result = cv2.resize(img_draw, scale_on_web)
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    result = PIL.Image.fromarray(result)
    image_buffer = BytesIO()
    result.save(image_buffer, format="JPEG")
    image_buffer.seek(0)
    return StreamingResponse(image_buffer, media_type="image/jpeg")





# ---------------------------- i3. Seg Sau ----------------------------
@app.post("/process_seg_sau")
def process_seg_sau():
    _, frame = cap.read()
    result_dict["view_list"].append(frame)
    frame = PIL.Image.fromarray(frame)
    result, _ = seg_model.infer(frame)  # (480, 640)
    result = np.array(result)
    mask = np.where(result > 100, 255, 0).astype(np.uint8)
    mask = np.stack((mask, mask, mask), axis=2)
    result_dict["seg_list"].append(mask)
    # ---------------------

    angle_A_right, angle_A_left, angle_B_right, angle_B_left, img_draw = tinh_goc_chan(result_dict["seg_list"][2], result_dict["information"])
    result_dict['angle_A_right'] = angle_A_right
    result_dict['angle_A_left'] = angle_A_left
    result_dict['angle_B_right'] = angle_B_right
    result_dict['angle_B_left'] = angle_B_left
    print(result_dict)

    # ---------------------
    result = cv2.resize(img_draw, scale_on_web)
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    result = PIL.Image.fromarray(result)
    image_buffer = BytesIO()
    result.save(image_buffer, format="JPEG")
    image_buffer.seek(0)
    return StreamingResponse(image_buffer, media_type="image/jpeg")





# ---------------------------- j1. Pose ----------------------------
@app.post("/process_pose")
def process_pose():
    _, frame = cap.read()
    result, keypoints = pose_model.infer(frame)  # (480, 640)

    keypoints_list = keypoints
    result_dict["keypoints"] = keypoints_list
    for name,point in zip(result_dict["kpt_names"],result_dict["keypoints"]):
        result_dict["information"][name]=point.tolist()
        x,y=point.tolist()

    result = cv2.resize(result, scale_on_web)
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    result = PIL.Image.fromarray(result)
    image_buffer = BytesIO()
    result.save(image_buffer, format="JPEG")
    image_buffer.seek(0)
    return StreamingResponse(image_buffer, media_type="image/jpeg")





# ---------------------------- k1. Send data ----------------------------
@app.post("/process_data")
def process_data():
    return {
        "height": int(result_dict["height"]),

        'DAy': int(result_dict['DAy']),
        'BAz': int(result_dict['BAz']),
        'CAx': int(result_dict['CAx']),

        'kc_2_knee': int(result_dict['kc_2_knee']),
        'kc_2_ank': int(result_dict['kc_2_ank']),

        'CAz': int(result_dict['CAz']),
        'ABz': int(result_dict['ABz']),

        'angle_A_right': int(result_dict['angle_A_right']),
        'angle_A_left': int(result_dict['angle_A_left']),
        'angle_B_right': int(result_dict['angle_B_right']),
        'angle_B_left': int(result_dict['angle_B_left']),
    }





# ---------------------------- l1. Hàm khởi tạo tham số d và f cho mô hình ----------------------------
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





# ---------------------------- m1. Show table ----------------------------
@app.post("/cut_image_00")
def cut_image_00():
    keypoints = result_dict["keypoints"]
    frontal_pose = result_dict["pose_list"][0]
    image_00 = frontal_pose[keypoints[8][1] :]
    cv2.imwrite("images_result/image_00.jpg", image_00)

    result = image_00
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    result = PIL.Image.fromarray(result)

    image_buffer = BytesIO()
    result.save(image_buffer, format="JPEG")
    image_buffer.seek(0)

    return StreamingResponse(image_buffer, media_type="image/jpeg")


@app.post("/cut_image_01")
def cut_image_01():
    keypoints = result_dict["keypoints"]
    frontal_pose = result_dict["pose_list"][0]
    image_01 = frontal_pose[: keypoints[8][1]]
    cv2.imwrite("images_result/image_01.jpg", image_01)

    result = image_01
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    result = PIL.Image.fromarray(result)

    image_buffer = BytesIO()
    result.save(image_buffer, format="JPEG")
    image_buffer.seek(0)

    return StreamingResponse(image_buffer, media_type="image/jpeg")


@app.post("/cut_image_10")
def cut_image_10():
    keypoints = result_dict["keypoints"]
    frontal_pose = result_dict["pose_list"][0]
    image_10 = frontal_pose[keypoints[8][1] :]
    cv2.imwrite("images_result/image_10.jpg", image_10)

    result = image_10
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    result = PIL.Image.fromarray(result)

    image_buffer = BytesIO()
    result.save(image_buffer, format="JPEG")
    image_buffer.seek(0)

    return StreamingResponse(image_buffer, media_type="image/jpeg")


@app.post("/cut_image_11")
def cut_image_11():
    keypoints = result_dict["keypoints"]
    profile_pose = result_dict["pose_list"][1]
    image_11 = profile_pose[: keypoints[8][1]]
    cv2.imwrite("images_result/image_11.jpg", image_11)

    result = image_11
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    result = PIL.Image.fromarray(result)

    image_buffer = BytesIO()
    result.save(image_buffer, format="JPEG")
    image_buffer.seek(0)

    return StreamingResponse(image_buffer, media_type="image/jpeg")


@app.post("/show_table")
def show_table():
    dis_2knee = distance_bet_2knee(result_dict["seg_list"][0], result_dict["keypoints"])
    point_max_l, point_max_r, sho_asymmetry = angle_bet_hip_neck(
        result_dict["seg_list"][0], result_dict["keypoints"]
    )
    angle_hip_neck = "angle_hip_neck"

    cut_image_00()
    cut_image_01()
    cut_image_10()
    cut_image_11()

    return {
        "dis_2knee": int(dis_2knee),
        "sho_asymmetry": int(sho_asymmetry),
        "angle_hip_neck": angle_hip_neck,
    }





# ---------------------------- n1. Hàm main ----------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, debug=True)
