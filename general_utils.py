import math
import numpy as np
import cv2


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


# 3.4. Tính khoảng cách giữa 2 đầu gối
def distance_bet_2knee(mask, keypoint):
    r_knee = keypoint[9]
    l_knee = keypoint[12]
    list_point_dark = np.where(mask[r_knee[1], r_knee[0] : l_knee[0]] < 50)[0]
    return list_point_dark[-1] - list_point_dark[0]


# 3.5. Góc giữa hông và cổ - Cường
def angle_bet_hip_neck(frontal_mask, keypoint):
    """
    tính điểm pixel trắng cao nhất từ cùi trỏ đi lên
    data[3] và data[6]
    """
    r_elb = keypoint[6]
    l_elb = keypoint[3]
    print(r_elb)
    print(l_elb)
    neck = keypoint[1]
    list_point_white_r = np.where(frontal_mask[neck[1] : r_elb[1], r_elb[0]] > 150)[0]
    list_point_white_l = np.where(frontal_mask[neck[1] : l_elb[1], l_elb[0]] > 150)[0]
    print(list_point_white_l.shape[0])
    point_max_r = [r_elb[0], -list_point_white_r.shape[0] + r_elb[1]]
    print(point_max_r)
    point_max_l = [l_elb[0], -list_point_white_l.shape[0] + l_elb[1]]
    print(point_max_l)
    # print(list_point_white_r.shape[0])
    """
    Tìm phương trình đường thẳng của 2 điểm đo được
    y=ax+b
    a=(y2-y1)/(x2-x1)
    b=y1-(y2-y1)*x1/(x2-x1)
    """
    x1, y1 = point_max_l[0], point_max_l[1]
    x2, y2 = point_max_r[0], point_max_r[1]
    a = (y2 - y1) / (x2 - x1)
    b = y1 - ((y2 - y1) * x1 / (x2 - x1))
    print(a, b)
    angle = math.atan(a) * 180 / 3.142
    return point_max_l, point_max_r, angle


# 3.6. Góc giữa hông và cổ - Khanh
def angle_neck_hip(profile_mask, keypoints):
    l_hip = np.array(keypoints[11])
    l_neck = np.array(keypoints[1])
    hip_neck_vector = (l_hip - l_neck)[np.array([1, 0])]
    Oy_vector = np.array([0, 100])
    angle = (hip_neck_vector.dot(Oy_vector)) / (
        np.linalg.norm(hip_neck_vector) * np.linalg.norm(Oy_vector)
    )

    # profile_mask = cv2.circle(profile_mask, l_hip, 5, (0, 0, 255), -1)
    # profile_mask = cv2.circle(profile_mask, l_neck, 5, (0, 0, 255), -1)

    return angle


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




# -------------------- Tính mỏm vai trái, mỏm vai phải --------------------

def tinh_mom_vai(img, information):
    img_draw = img.copy()
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
    # cv2.line(img_draw,(information['neck'][0] - 100,information['neck'][1]),(information['neck'][0] + 100,information['neck'][1]),(255,0,0),line_width) # trục ngang đi qua neck
    # cv2.line(img_draw,information['neck'],(information['neck'][0],information['neck'][1] - 100),(255,0,0),line_width) # trục dọc đi qua neck
    cv2.line(img_draw,information['neck'],mom_right,(0,255,0),line_width) 
    cv2.line(img_draw,information['neck'],mom_left,(0,255,0),line_width)
    cv2.line(img_draw,information['neck'],information['nose'],(0,255,0),line_width)

    return DAy, BAz, CAx, img_draw


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


# -------------------- Tìm điểm gót chân (Viết hàm tìm lại, tìm 2 gót chân) --------------------
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
    mask = img[:, :, 0]
    img_draw = img.copy()
    diem_ngoai_cung_r_knee = np.array((int(np.where(mask[information['r_knee'][1], information['r_knee'][0]:]==0)[0][0] + information['r_knee'][0]), information['r_knee'][1])) # Điểm ngoài cùng r_knee
    diem_ngoai_cung_l_knee = np.array((int(np.where(mask[information['r_knee'][1], :information['l_knee'][0]]==0)[0][-1]), information['r_knee'][1])) # Điểm ngoài cùng l_knee
    cv2.circle(img_draw,diem_ngoai_cung_r_knee,5,(0,0,255),-1)
    cv2.circle(img_draw,diem_ngoai_cung_l_knee,5,(0,0,255),-1)
    return np.linalg.norm(diem_ngoai_cung_l_knee - diem_ngoai_cung_r_knee)
    



# -------------------- Khoảng cách 2 mắt cá --------------------
def distance_bet_2_ank_new(img, information):
    mask = img[:, :, 0]
    img_draw = img.copy()
    diem_ngoai_cung_r_ank = np.array((int(np.where(mask[information['r_ank'][1], information['r_ank'][0]:]==0)[0][0] + information['r_ank'][0]), information['r_ank'][1])) # Điểm ngoài cùng r_ank
    diem_ngoai_cung_l_ank = np.array((int(np.where(mask[information['r_ank'][1], :information['l_ank'][0]]==0)[0][-1]), information['r_ank'][1])) # Điểm ngoài cùng l_ank
    cv2.circle(img_draw,diem_ngoai_cung_r_ank,5,(0,0,255),-1)
    cv2.circle(img_draw,diem_ngoai_cung_l_ank,5,(0,0,255),-1)
    return np.linalg.norm(diem_ngoai_cung_l_ank - diem_ngoai_cung_r_ank)
    


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