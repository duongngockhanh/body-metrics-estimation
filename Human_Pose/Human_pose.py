import cv2
import numpy as np
import torch

import time
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose
from val import normalize, pad_width, draw_grid, get_distance


class Human_Pose:
    def __init__(self, model_path=None):
        self.model = PoseEstimationWithMobileNet()
        self.checkpoint = torch.load(model_path, map_location="cpu")
        load_state(self.model, self.checkpoint)

    def preprocess(self, img):
        height, width, _ = img.shape
        net_input_height_size = 256
        img_mean = np.array([128, 128, 128], np.float32)
        img_scale = np.float32(1 / 256)

        stride = 8
        pad_value = (0, 0, 0)
        scale = net_input_height_size / height

        scaled_img = cv2.resize(
            img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR
        )
        scaled_img = normalize(scaled_img, img_mean, img_scale)
        min_dims = [
            net_input_height_size,
            max(scaled_img.shape[1], net_input_height_size),
        ]
        padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

        tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
        return tensor_img, scale, pad

    def infer(self, img):
        orig_img = img.copy()
        results_img = img.copy()
        tensor_img, scale, pad = self.preprocess(img)
        tik = time.time()
        upsample_ratio = 4
        stride = 8
        upsample_ratio = 4
        num_keypoints = Pose.num_kpts
        previous_poses = []
        list_sort = []
        delay = 1
        # output ############
        stages_output = self.model(tensor_img)
        print("Time processing :", time.time() - tik)
        stage2_heatmaps = stages_output[-2]
        heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
        heatmaps = cv2.resize(
            heatmaps,
            (0, 0),
            fx=upsample_ratio,
            fy=upsample_ratio,
            interpolation=cv2.INTER_CUBIC,
        )

        stage2_pafs = stages_output[-1]
        pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
        pafs = cv2.resize(
            pafs,
            (0, 0),
            fx=upsample_ratio,
            fy=upsample_ratio,
            interpolation=cv2.INTER_CUBIC,
        )
        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(
                heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num
            )

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (
                all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]
            ) / scale
            all_keypoints[kpt_id, 1] = (
                all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]
            ) / scale
        current_poses = []
        current_poses_condinate = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(
                        all_keypoints[int(pose_entries[n][kpt_id]), 0]
                    )
                    pose_keypoints[kpt_id, 1] = int(
                        all_keypoints[int(pose_entries[n][kpt_id]), 1]
                    )
            pose = Pose(pose_keypoints, pose_entries[n][18])
            current_poses.append(pose)
            current_poses_condinate.append(pose.keypoints)

        # print(pose.keypoints)
        list_sort = pose.keypoints.copy()
        list_sort = sorted(list_sort, key=lambda x: x[1])
        distance = list_sort[-1][-1] - list_sort[0][-1]
        ###Độ dài ống xương phải###
        distance_knee2ank_r = get_distance(pose.keypoints[10], pose.keypoints[9])
        #####Độ dài đùi phải######
        distance_knee2hip_r = get_distance(pose.keypoints[9], pose.keypoints[8])
        #######Độ dài ống xương trái #######
        distance_knee2ank_l = get_distance(pose.keypoints[13], pose.keypoints[12])
        ####### Độ dài đùi trái#################
        distance_knee2hip_l = get_distance(pose.keypoints[12], pose.keypoints[11])
        ####### Độ dài thân trên ###############
        point_neck2 = np.array([pose.keypoints[1][0], pose.keypoints[12][1]])
        distance_neck2hip = get_distance(pose.keypoints[1], point_neck2)
        print("distance_neck2hip", distance_neck2hip)
        ##############################################
        print("Left", [distance_knee2ank_l, distance_knee2hip_l])
        print("Right", [distance_knee2ank_r, distance_knee2hip_r])
        for pose in current_poses:
            pose.draw(results_img)
        img_sss = cv2.addWeighted(orig_img, 0.6, results_img, 0.4, 0)
        draw_grid(img_sss, line_space=20)
        return img_sss, pose.keypoints


if __name__ == "__main__":
    img = cv2.imread("quang3.jpg")
    model = Human_Pose(model_path="weights/checkpoint_iter_370000.pth")
    img_results, keypoints = model.infer(img)
    print(keypoints)
    np.save("keypoint.npy", np.array(keypoints))
    cv2.imwrite("results.jpg", img_results)
