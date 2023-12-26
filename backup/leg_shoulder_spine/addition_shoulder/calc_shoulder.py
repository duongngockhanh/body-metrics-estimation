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
img=cv2.imread("test.jpg")
img=np.where(img>100,255,0).astype(np.uint8)
img_draw = img.copy()
keypoints=np.load("keypoint.npy")
information={}
for name,point in zip(kpt_names,keypoints):
    information[name]=point.tolist()
    x,y=point.tolist()
    cv2.circle(img_draw,(x,y),3,(255,0,0),-1)



# -------------------- Tính mỏm vai trái, mỏm vai phải --------------------
def tinh_mom_vai(img, information):
    temp = 10
    mask = img[:, :, 0]

    mom_right = (int(np.where(mask[information['r_sho'][1] - temp, :information['r_sho'][0]] == 0)[0][-1]), information['r_sho'][1] - temp)
    mom_left = (int(np.where(mask[information['l_sho'][1] - temp, :] == 255)[0][-1]), information['l_sho'][1] - temp)

    # cv2.circle(img_draw,mom_right,5,(0,0,255),-1)
    # cv2.circle(img_draw,mom_left,5,(0,0,255),-1)



    # -------------------- Tính các góc --------------------
    DAy = get_angle_cosine(np.array(information['nose']) - np.array(information['neck']), np.array([0, -10]))
    BAz = get_angle_cosine(np.array(mom_right) - np.array(information['neck']), np.array([-10, 0]))
    CAx = get_angle_cosine(np.array(mom_left) - np.array(information['neck']), np.array([10, 0]))

    
    # print(DAy2)
    # print(BAz2)
    # print(CAx2)



# -------------------- Vẽ các góc --------------------
    line_width = 3
    cv2.line(img_draw,(information['neck'][0] - 100,information['neck'][1]),(information['neck'][0] + 100,information['neck'][1]),(255,0,0),line_width) # trục ngang đi qua neck
    cv2.line(img_draw,information['neck'],(information['neck'][0],information['neck'][1] - 100),(255,0,0),line_width) # trục dọc đi qua neck
    cv2.line(img_draw,information['neck'],mom_right,(0,255,0),line_width) 
    cv2.line(img_draw,information['neck'],mom_left,(0,255,0),line_width)
    cv2.line(img_draw,information['neck'],information['nose'],(0,255,0),line_width)
    cv2.imwrite("test2.jpg", img_draw)

    return DAy, BAz, CAx