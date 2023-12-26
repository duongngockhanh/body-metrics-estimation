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



# -------------------- Tìm điểm gót chân (Viết hàm tìm lại, tìm 2 gót chân) --------------------
def get_2_got_chan(img, information):
    mask = img[:, :, 0]
    mid_anks = (int((information["l_ank"][0]+information["r_ank"][0])/2),int((information["l_ank"][1]+information["r_ank"][1])/2))

    mask_right = mask[:, :mid_anks[0]]
    index_y_max_right = np.where(mask_right==255)[0][-1]
    index_x_min_right = np.where(mask_right[index_y_max_right]==255)[0][0]
    got_chan_right = (index_x_min_right + 5, index_y_max_right)

    mask_left = mask[:, mid_anks[0]:]
    index_y_max_left = np.where(mask_left==255)[0][-1]
    index_x_max_left = np.where(mask[index_y_max_left]==255)[0][-1]
    got_chan_left = (index_x_max_left - 5, index_y_max_left)

    return got_chan_left, got_chan_right


def tinh_goc_chan(img, information):
    got_chan_left, got_chan_right = get_2_got_chan(img, information)

    img_draw = img.copy()


    # -------------------- Xác định trung điểm của đầu gối và mắt cá chân --------------------
    point_center_l=(int((information["l_knee"][0]+information["l_ank"][0])/2),int((information["l_knee"][1]+information["l_ank"][1])/2))
    point_center_r=(int((information["r_knee"][0]+information["r_ank"][0])/2),int((information["r_knee"][1]+information["r_ank"][1])/2))


    cv2.circle(img_draw,point_center_r,5,(0,123,255),-1)
    cv2.circle(img_draw,point_center_l,5,(0,123,255),-1)

    cv2.circle(img_draw,got_chan_left,5,(0,0,255),-1)
    cv2.circle(img_draw,got_chan_right,5,(0,0,255),-1)



    # -------------------- Tính góc angle_A_left và angle_A_right --------------------
    angle_A_right = get_angle_cosine(np.array(information["r_ank"]) - np.array(got_chan_right), np.array([-10, 0]))
    angle_A_left = get_angle_cosine(np.array(information["l_ank"]) - np.array(got_chan_left), np.array([10, 0]))
    print("angle_A_right", angle_A_right)
    print("angle_A_left", angle_A_left)



    # -------------------- Tính góc angle_B_left và angle_B_right --------------------
    angle_B_right = get_angle_cosine(np.array(point_center_r) - np.array(information["r_ank"]), np.array([-10, 0]))
    angle_B_left = get_angle_cosine(np.array(point_center_l) - np.array(information["l_ank"]), np.array([10, 0]))
    print("angle_B_right", angle_B_right)
    print("angle_B_left", angle_B_left)


    # -------------------- Nối các điểm A, B, C --------------------
    line_width = 3
    cv2.line(img_draw,information["r_ank"],got_chan_right,(0,0,255),line_width) # AB_right
    cv2.line(img_draw,information["r_ank"],point_center_r,(0,0,255),line_width) # BC_right
    cv2.line(img_draw,information["l_ank"],got_chan_left,(0,0,255),line_width) # AB_left
    cv2.line(img_draw,information["l_ank"],point_center_l,(0,0,255),line_width) # BC_left
    draw_axes_B(img_draw, point_name=information["r_ank"])
    draw_axes_B(img_draw, point_name=information["l_ank"])
    draw_axes_A(img_draw, got_chan_left)
    draw_axes_A(img_draw, got_chan_right)

    return angle_A_right, angle_A_left, angle_B_right, angle_B_left, img_draw



# -------------------- Khoảng cách 2 đầu gối --------------------
def distance_bet_2_knee_new(img, information):
    mask = img[:, :, 2]
    img_draw = img.copy()
    diem_ngoai_cung_r_knee = np.array((int(np.where(mask[information['r_knee'][1], information['r_knee'][0]:]==0)[0][0] + information['r_knee'][0]), information['r_knee'][1])) # Điểm ngoài cùng r_knee
    diem_ngoai_cung_l_knee = np.array((int(np.where(mask[information['r_knee'][1], :information['l_knee'][0]]==0)[0][-1]), information['r_knee'][1])) # Điểm ngoài cùng l_knee
    cv2.circle(img_draw,diem_ngoai_cung_r_knee,5,(0,0,255),-1)
    cv2.circle(img_draw,diem_ngoai_cung_l_knee,5,(0,0,255),-1)
    return np.linalg.norm(diem_ngoai_cung_l_knee - diem_ngoai_cung_r_knee)
    
print("distance_bet_2_knee_new", distance_bet_2_knee_new(img, information))



# -------------------- Khoảng cách 2 mắt cá --------------------
def distance_bet_2_ank_new(img, information):
    mask = img[:, :, 2]
    img_draw = img.copy()
    diem_ngoai_cung_r_ank = np.array((int(np.where(mask[information['r_ank'][1], information['r_ank'][0]:]==0)[0][0] + information['r_ank'][0]), information['r_ank'][1])) # Điểm ngoài cùng r_ank
    diem_ngoai_cung_l_ank = np.array((int(np.where(mask[information['r_ank'][1], :information['l_ank'][0]]==0)[0][-1]), information['r_ank'][1])) # Điểm ngoài cùng l_ank
    cv2.circle(img_draw,diem_ngoai_cung_r_ank,5,(0,0,255),-1)
    cv2.circle(img_draw,diem_ngoai_cung_l_ank,5,(0,0,255),-1)
    return np.linalg.norm(diem_ngoai_cung_l_ank - diem_ngoai_cung_r_ank)
    
print("distance_bet_2_ank_new", distance_bet_2_ank_new(img, information))


def tinh_khoang_cach_chan(img, information):
    return distance_bet_2_knee_new(img, information), distance_bet_2_ank_new(img, information)


# -------------------- Kẻ các trục --------------------
def draw_axes_B(img_draw, point_name, line_width=3):
    # Vẽ line trục ngang theo neck
    cv2.line(img_draw,point_name,(point_name[0]+70,point_name[1]),(255,0,0),line_width)
    cv2.line(img_draw,point_name,(point_name[0]-70,point_name[1]),(255,0,0),line_width)

    # Vẽ line trục dọc theo neck
    cv2.line(img_draw,point_name,(point_name[0],point_name[1]-70),(255,0,0),line_width)
    cv2.line(img_draw,point_name,(point_name[0],point_name[1]+70),(255,0,0),line_width)



def draw_axes_A(img_draw, point_got_chan, line_width=3):
    cv2.line(img_draw,point_got_chan,(point_got_chan[0]-100,point_got_chan[1]),(255,200,0),line_width)
    cv2.line(img_draw,point_got_chan,(point_got_chan[0],point_got_chan[1]-100),(255,200,0),line_width)



# -------------------- Save image --------------------
cv2.imwrite("leg.jpg",img_draw)



