import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch
from PIL import Image
import numpy as np
import cv2
CHAR_IMG_SIZE = 64


def char_img_transform(input_img, threshold=False):
    img = input_img.copy()
    # if threshold:
        # h, w = img.shape[:2]
        # win = min(h, w) // 2
        # if win % 2 == 0:
        #     win += 1
        # if win < 3:
        #     win = 3
        # img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,win, 2) 
        #img = cv2.threshold(np.array(img, dtype=np.uint8), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        # img = cv2.equalizeHist(img)
        # img = cv2.normalize( img, None, alpha=0,beta=255, norm_type=cv2.NORM_MINMAX )
    if(isinstance(img, np.ndarray)):
        img = Image.fromarray(img, mode='L')
    transform =  transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((CHAR_IMG_SIZE, CHAR_IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
    
    return transform(img)