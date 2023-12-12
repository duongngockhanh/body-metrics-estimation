import sys, os, cv2, PIL, argparse
import numpy as np

root_path = os.getcwd()

seg_path = os.path.join(root_path, "Human_Segmantation")
sys.path.append(seg_path)
from Human_seg import Human_Semantic

pose_path = os.path.join(root_path, "Human_Pose")
sys.path.append(pose_path)
from Human_pose import Human_Pose


class Estimation:
    def __init__(self, task, weight):
        self.task = task
        self.weight = weight
        if task == "pose":
            self.model = Human_Pose(weight)
        elif task == "seg":
            self.model = Human_Semantic(weight)

    def infer(self, img):
        if self.task == "pose":
            result, _ = self.model.infer(img)
            return result
        elif self.task == "seg":
            img = PIL.Image.fromarray(img)
            result, _ = self.model.infer(img)
            return np.array(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="seg")
    parser.add_argument(
        "--weight", type=str, default="Human_Segmantation/weights/SGHM-ResNet50.pth"
    )
    parser.add_argument("--image", type=str, required=True)
    opt = parser.parse_args()

    estimator = Estimation(opt.task, opt.weight)
    img = cv2.imread(opt.image)

    result = estimator.infer(img)
    save_path = "result_" + opt.task + ".jpg"
    cv2.imwrite(save_path, result)
    print("SUCCESS. The result is saved at", save_path)
