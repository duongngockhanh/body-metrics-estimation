import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from model.model import HumanMatting


import utils
pil_to_tensor = transforms.Compose(
    [
        transforms.ToTensor()
    ]
)

infer_size = 1280
class Human_Semantic():
    def __init__(self,model_path=None):
        self.infer_size=1280
        self.pil_to_tensor=transforms.Compose([transforms.ToTensor()])
        self.model=HumanMatting(backbone="resnet50")
        self.model = nn.DataParallel(self.model).cpu().eval()
        self.model.load_state_dict(torch.load(model_path,map_location=torch.device("cpu")))
        
    def preprocess(self,img):
        h = img.height
        w = img.width
        if w >= h:
            rh = infer_size
            rw = int(w / h * infer_size)
        else:
            rw = infer_size
            rh = int(h / w * infer_size)
        rh = rh - rh % 64
        rw = rw - rw % 64    

        img = pil_to_tensor(img)
        img = img[None, :, :, :]

        input_tensor = F.interpolate(img, size=(rh, rw), mode='bilinear')
        return input_tensor
    def infer(self,img):
        h = img.height
        w = img.width
        input_tensor=self.preprocess(img)
        with torch.no_grad():
            pred=self.model(input_tensor)
            # progressive refine alpha
        alpha_pred_os1, alpha_pred_os4, alpha_pred_os8 = pred['alpha_os1'], pred['alpha_os4'], pred['alpha_os8']
        pred_alpha = alpha_pred_os8.clone().detach()
        weight_os4 = utils.get_unknown_tensor_from_pred(pred_alpha, rand_width=30, train_mode=False)
        pred_alpha[weight_os4>0] = alpha_pred_os4[weight_os4>0]
        weight_os1 = utils.get_unknown_tensor_from_pred(pred_alpha, rand_width=15, train_mode=False)
        pred_alpha[weight_os1>0] = alpha_pred_os1[weight_os1>0]

        pred_alpha = pred_alpha.repeat(1, 3, 1, 1)
        pred_alpha = F.interpolate(pred_alpha, size=(h, w), mode='bilinear')
        alpha_np = pred_alpha[0].data.cpu().numpy().transpose(1, 2, 0)
        alpha_np = alpha_np[:, :, 0]

        # output segment
        pred_segment = pred['segment']
        pred_segment = F.interpolate(pred_segment, size=(h, w), mode='bilinear')
        segment_np = pred_segment[0].data.cpu().numpy().transpose(1, 2, 0)

        return  Image.fromarray(((alpha_np * 255).astype('uint8')), mode='L'),alpha_np
        

if __name__=="__main__":
    img=Image.open("quang3.jpg")
    model=Human_Semantic(model_path="weights/SGHM-ResNet50.pth")
    result,alpha_np=model.infer(img)
    alpha=alpha_np.astype(np.uint8)*255
    print(np.where(alpha==255))
    result.save("seg_res.jpg")
    
    
    
        
        
        