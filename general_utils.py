import math
import numpy as np


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
