import numpy as np
import cv2
import math



# -------------------- Hàm tính góc trái so với phương ngang --------------------
def get_angle(point1,point2):
    if point1[0]==point2[0]:
        return 0
    return math.degrees(math.atan2((point2[1]-point1[1]),(point2[0]-point1[0])))

def get_angle_cosine(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    cos_theta = dot_product / (norm_v1 * norm_v2)
    angle_in_radians = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    angle_in_degrees = np.degrees(angle_in_radians)

    return angle_in_degrees

# -------------------- Load mask và tạo information dictionary --------------------
kpt_names = ['nose', 'neck',
                 'r_sho', 'r_elb', 'r_wri', 'l_sho', 'l_elb', 'l_wri',
                 'r_hip', 'r_knee', 'r_ank', 'l_hip', 'l_knee', 'l_ank',
                 'r_eye', 'l_eye',
                 'r_ear', 'l_ear']
img=cv2.imread("test_spine.jpg")
img=np.where(img>100,255,0).astype(np.uint8)
img_draw = img.copy()
keypoints=np.load("keypoint_spine.npy")
information={}
for name,point in zip(kpt_names,keypoints):
    information[name]=point.tolist()
    x,y=point.tolist()
    cv2.circle(img_draw,(x,y),3,(255,0,0),-1)



# -------------------- Tính CAz, ABz --------------------

def tinh_cot_song(img, information):
    '''
    C: Tai trái
    A: Cổ
    B: Hông trái
    '''
    img_draw = img.copy()

    CAz = get_angle_cosine(np.array(information['l_ear']) - np.array(information['neck']), np.array([-10, 0]))
    ABz = get_angle_cosine(np.array(information['neck']) - np.array(information['l_hip']), np.array([-10, 0]))

    # print(CAz)
    # print(ABz)



    # -------------------- Vẽ CAz, ABz --------------------
    line_width = 2

    cv2.line(img_draw,(information['neck'][0],information['neck'][1] + 100),(information['neck'][0],information['neck'][1] - 100),(255,0,0),line_width) # trục dọc đi qua neck
    cv2.line(img_draw,(information['neck'][0] + 100,information['neck'][1]),(information['neck'][0] - 100,information['neck'][1]),(255,0,0),line_width) # trục dọc đi qua neck
    cv2.line(img_draw,(information['l_hip'][0],information['l_hip'][1] + 100),(information['l_hip'][0],information['l_hip'][1] - 100),(255,0,0),line_width) # trục dọc đi qua l_hip
    cv2.line(img_draw,(information['l_hip'][0] + 100,information['l_hip'][1]),(information['l_hip'][0] - 100,information['l_hip'][1]),(255,0,0),line_width) # trục dọc đi qua l_hip
    cv2.line(img_draw,information['neck'],information['l_hip'],(0,0,255),line_width) 
    cv2.line(img_draw,information['neck'],information['l_ear'],(0,0,255),line_width)
    # cv2.imwrite("test2.jpg", img_draw)

    return CAz, ABz, img_draw