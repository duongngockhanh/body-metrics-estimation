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


measure = HeightMeasure()


# 4. Các biến toàn cục

# 4.1. Tạo các biến toàn cục
scale_on_web = (320, 320)

kpt_names = ['nose', 'neck',
                 'r_sho', 'r_elb', 'r_wri', 'l_sho', 'l_elb', 'l_wri',
                 'r_hip', 'r_knee', 'r_ank', 'l_hip', 'l_knee', 'l_ank',
                 'r_eye', 'l_eye',
                 'r_ear', 'l_ear']

# 4.2. Tạo result_dict để lưu các thông tin quan trọng
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
    "seg_list": [],
    "pose_list": [],
    "scan_list": [],
    "view_list": [],
    "kpt_names": kpt_names,
    "information": {}
}




# 5. Tạo các hàm tương tác với FE


# 5.1. Định nghĩa việc tương tác với index.html
@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


cap = cv2.VideoCapture(0)


# 5.2. Hàm dùng để đẩy lần lượt các frames lên FE
def gen_frames():
    while True:
        # frame = cv2.cvtColor(freenect.sync_get_video()[0],cv2.COLOR_BGR2RGB)
        _, frame = cap.read()
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
    # frame = cv2.cvtColor(cv2.imread("other/quang.jpg"), cv2.COLOR_BGR2RGB)
    _, frame = cap.read()
    # frame = cv2.cvtColor(freenect.sync_get_video()[0],cv2.COLOR_BGR2RGB)
    start_scan = time.time()
    result, boxes = scan_model.inference(frame)
    scan_time = time.time() - start_scan
    print(f"---------------------------------------------------------------------------------------------------------------------- Scan Time: {scan_time:.2f}s", )

    result_dict["scan_list"].append(result)
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
    # frame = cv2.cvtColor(cv2.imread("other/quang.jpg"), cv2.COLOR_BGR2RGB)
    ret, frame = cap.read()
    result_dict["view_list"].append(frame)
    # frame = cv2.cvtColor(freenect.sync_get_video()[0],cv2.COLOR_BGR2RGB)
    if not ret:
        print("Loi anh")
    frame = PIL.Image.fromarray(frame)
    start_seg = time.time()
    result, _ = seg_model.infer(frame)  # (480, 640)
    seg_time = time.time() - start_seg
    print(f"---------------------------------------------------------------------------------------------------------------------- Scan Time: {seg_time:.2f}s", )
    result = np.array(result)
    # result_dict["seg_list"].append(result)

    # Đo chiều cao theo pixel:
    temp = np.where(result > 100, 255, 0).astype(np.uint8)
    result_dict["seg_list"].append(temp)

    # ---------------------

    DAy, BAz, CAx, img_draw = tinh_mom_vai(result_dict["seg_list"][0], result_dict["information"])
    result_dict['DAy'] = DAy
    result_dict['BAz'] = BAz
    result_dict['CAx'] = CAx
    kc_2_knee, kc_2_ank = tinh_khoang_cach_chan(result_dict["seg_list"][0], result_dict["information"])
    result_dict['kc_2_knee'] = kc_2_knee
    result_dict['kc_2_ank'] = kc_2_ank
    print(result_dict)



    # ---------------------

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


# 5.4. Hàm xử lý Segmentation
@app.post("/process_seg_nghieng")
def process_seg_nghieng():
    # frame = cv2.cvtColor(cv2.imread("other/quang.jpg"), cv2.COLOR_BGR2RGB)
    _, frame = cap.read()
    result_dict["view_list"].append(frame)
    # frame = cv2.cvtColor(freenect.sync_get_video()[0],cv2.COLOR_BGR2RGB)
    frame = PIL.Image.fromarray(frame)
    start_seg = time.time()
    result, _ = seg_model.infer(frame)  # (480, 640)
    seg_time = time.time() - start_seg
    print(f"---------------------------------------------------------------------------------------------------------------------- Scan Time: {seg_time:.2f}s", )
    result = np.array(result)
    # result_dict["seg_list"].append(result)

    # Đo chiều cao theo pixel:
    temp = np.where(result > 100, 255, 0).astype(np.uint8)
    result_dict["seg_list"].append(temp)

    # ---------------------

    CAz, ABz, img_draw = tinh_cot_song(result_dict["seg_list"][1], result_dict["information"])
    result_dict['CAz'] = CAz
    result_dict['ABz'] = ABz
    print(result_dict)

    # ---------------------

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


# 5.4. Hàm xử lý Segmentation
@app.post("/process_seg_sau")
def process_seg_sau():
    # frame = cv2.cvtColor(cv2.imread("other/quang.jpg"), cv2.COLOR_BGR2RGB)
    _, frame = cap.read()
    result_dict["view_list"].append(frame)
    # frame = cv2.cvtColor(freenect.sync_get_video()[0],cv2.COLOR_BGR2RGB)
    frame = PIL.Image.fromarray(frame)
    start_seg = time.time()
    result, _ = seg_model.infer(frame)  # (480, 640)
    seg_time = time.time() - start_seg
    print(f"---------------------------------------------------------------------------------------------------------------------- Scan Time: {seg_time:.2f}s", )
    result = np.array(result)
    # result_dict["seg_list"].append(result)

    # Đo chiều cao theo pixel:
    temp = np.where(result > 100, 255, 0).astype(np.uint8)
    result_dict["seg_list"].append(temp)

    # ---------------------
    angle_A_right, angle_A_left, angle_B_right, angle_B_left, img_draw = tinh_goc_chan(result_dict["seg_list"][0], result_dict["information"])
    result_dict['angle_A_right'] = angle_A_right
    result_dict['angle_A_left'] = angle_A_left
    result_dict['angle_B_right'] = angle_B_right
    result_dict['angle_B_left'] = angle_B_left
    print(result_dict)



    # ---------------------

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
    # frame = cv2.cvtColor(cv2.imread("other/quang.jpg"), cv2.COLOR_BGR2RGB)
    _, frame = cap.read()
    # frame = cv2.cvtColor(freenect.sync_get_video()[0],cv2.COLOR_BGR2RGB)
    start_pose = time.time()
    result, keypoints = pose_model.infer(frame)  # (480, 640)
    pose_time = time.time() - start_pose
    print(f"---------------------------------------------------------------------------------------------------------------------- Scan Time: {pose_time:.2f}s", )

    result_dict["pose_list"].append(result)
    result_dict["keypoints"] = keypoints

    keypoints_list = keypoints
    print(keypoints_list)

    result_dict["keypoints"] = keypoints_list


    print(type(result_dict["kpt_names"]))
    print(type(result_dict["keypoints"]))
    for name,point in zip(result_dict["kpt_names"],result_dict["keypoints"]):
        result_dict["information"][name]=point.tolist()
        x,y=point.tolist()

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


@app.post("/process_pose_nghieng")
def process_pose_nghieng():
    # frame = cv2.cvtColor(cv2.imread("other/quang.jpg"), cv2.COLOR_BGR2RGB)
    _, frame = cap.read()
    # frame = cv2.cvtColor(freenect.sync_get_video()[0],cv2.COLOR_BGR2RGB)
    start_pose = time.time()
    result, keypoints = pose_model.infer(frame)  # (480, 640)
    pose_time = time.time() - start_pose
    print(f"---------------------------------------------------------------------------------------------------------------------- Scan Time: {pose_time:.2f}s", )

    result_dict["pose_list"].append(result)
    result_dict["keypoints"] = keypoints

    keypoints_list = keypoints.tolist()
    print(keypoints_list)

    result_dict["keypoints"] = keypoints_list

    for name,point in zip(result_dict["kpt_names"],result_dict["keypoints"]):
        result_dict["information"][name]=point.tolist()
        x,y=point.tolist()

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


@app.post("/process_pose_sau")
def process_pose_sau():
    # frame = cv2.cvtColor(cv2.imread("other/quang.jpg"), cv2.COLOR_BGR2RGB)
    _, frame = cap.read()
    # frame = cv2.cvtColor(freenect.sync_get_video()[0],cv2.COLOR_BGR2RGB)
    start_pose = time.time()
    result, keypoints = pose_model.infer(frame)  # (480, 640)
    pose_time = time.time() - start_pose
    print(f"---------------------------------------------------------------------------------------------------------------------- Scan Time: {pose_time:.2f}s", )

    result_dict["pose_list"].append(result)
    result_dict["keypoints"] = keypoints

    keypoints_list = keypoints.tolist()
    print(keypoints_list)

    result_dict["keypoints"] = keypoints_list

    for name,point in zip(result_dict["kpt_names"],result_dict["keypoints"]):
        result_dict["information"][name]=point.tolist()
        x,y=point.tolist()

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
    mydataset = {
        "information": list(result_dict.keys())[:5],
        "figures": list(result_dict.values())[:5],
        "assessment": [0] * 5,
    }
    myvar = pd.DataFrame(mydataset)
    myvar.to_csv(index=False)

    # Lưu dữ liệu thành tệp CSV
    csv_filename = "data.csv"
    myvar.to_csv(csv_filename, index=False)

    # Lưu dữ liệu thành tệp Excel
    excel_filename = "data.xlsx"
    myvar.to_excel(excel_filename, index=False)

    # Hiển thị số liệu lên web
    return {
        "height": int(result_dict["height"]),
        "ongchan": result_dict["ongchan"],
        "haidaugoi": int(result_dict["haidaugoi"]),
        "chieudaithan": int(result_dict["chieudaithan"]),
        "dolechvai": int(result_dict["dolechvai"]),

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


"""
'nose', 'neck', 'r_sho', 'r_elb', 'r_wri', 
'l_sho', 'l_elb', 'l_wri', 'r_hip', 'r_knee', 
'r_ank', 'l_hip', 'l_knee', 'l_ank', 'r_eye', 
'l_eye', 'r_ear', 'l_ear'
"""


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
    # angle_hip_neck = angle_neck_hip(profile_mask, keypoints)
    # dis_2knee = "dis_2knee"
    # point_max_l,point_max_r,sho_asymmetry = "point_max_l","point_max_r","sho_asymmetry"
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


# 6. Hàm main
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, debug=True)
