#first we get an image of cars
#now, we use vehilc yolo model and plate yolo model to segment license plates and cars, and match license plates to cars
#now, we use character yolo model to box the characters in the license plate
#We use the trained CNN model to recognize characters from the segmented license plate characters

from matplotlib import transforms
import torch
from ultralytics import YOLO
import cv2
import string
from my_config import CHAR_IMG_SIZE
from my_config import char_img_transform
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from statistics import median
import os

import logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)

LOG_ENABLED = True  

def print_log(args, **kwargs):
    if LOG_ENABLED:
        print(args, **kwargs)


class CharCNN(nn.Module):
    def __init__(self, num_classes=36):
        super(CharCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)   
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)  
        with torch.no_grad():
            sample_input = torch.zeros(1, 1, CHAR_IMG_SIZE, CHAR_IMG_SIZE)
            sample_output = self.pool3(F.relu(self.conv3(self.pool2(F.relu(self.conv2(self.pool1(F.relu(self.conv1(sample_input))))))))) #check , if you change above, change this too
            flattened_size = sample_output.numel()

        self.fc1 = nn.Linear(flattened_size, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        if(x.dim() == 3):
            x = x.unsqueeze(0)
        elif(x.dim() == 2):
            x = x.unsqueeze(0).unsqueeze(0)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

INT_TO_CHAR_MAP = {idx: ch for idx, ch in enumerate('0123456789' + string.ascii_uppercase)}


def get_character_recogniser(model):

    def character_recogniser(char_img, show_character = False):
        
        char_img_copy = char_img.copy()
        char_img_copy = cv2.cvtColor(char_img_copy, cv2.COLOR_BGR2GRAY) if len(char_img_copy.shape)==3 else char_img_copy
        if(np.max(char_img_copy)<1):
            char_img_copy = (char_img_copy*255).astype(np.uint8)
        boundary_sum= (np.sum(char_img_copy[0:2])+np.sum(char_img_copy[:,0:2])+np.sum(char_img_copy[-2:])+np.sum(char_img_copy[:,-2:])) #if the boundary is mostly white, then invert
        if((boundary_sum/(4*(char_img_copy.shape[0]+char_img_copy.shape[1]))) >50):
            char_img_copy = 255-char_img_copy

        char_img_copy = char_img_transform(char_img_copy)
        if(show_character):
            plt.imshow(char_img_copy.squeeze(), cmap='gray')
            plt.show()
        prediction = torch.argmax(model(char_img_copy), dim=-1)
        character = INT_TO_CHAR_MAP.get(prediction.item(), '')
        return character
    
    return character_recogniser

def get_text_from_plate(plate_img, character_detector, character_recogniser, show_individual_characters=False):
    detections = character_detector(plate_img)[0]
    # raise Exception("Debugging - remove this later")
    output=[]
    midpoints_x = set()
    for box in detections.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        exists=False
        for mp in midpoints_x:
            if(mp>x1 and mp<x2):
                exists=True
                break
        if(exists):
            continue
        midpoints_x.add((x1+x2)//2)
        c= character_recogniser(plate_img[y1:y2, x1:x2], show_character = show_individual_characters) #check
        output.append(((x1+x2)//2, c, (x2 - x1) * (y2 - y1)))
    if(len(output)==0):
        return None
    output=sorted(output, key=lambda x:x[0])
    box_areas_median = median([x[2] for x in output])
    output = [x for x in output if x[2]>=0.25*box_areas_median] #filtering out small boxes
    plate_text = ''.join(map(lambda x: x[1], output))
    return plate_text



def get_corresponding_car(license_plate, vehicle_detections):
    xplate1,yplate1,xplate2,yplate2,_,_ = license_plate
    for vehicle in vehicle_detections:
        xcar1,ycar1,xcar2,ycar2,_,_ = vehicle
        if (xplate1 >= xcar1 and xplate2 <= xcar2 and yplate1 >= ycar1 and yplate2 <= ycar2):
            return (xcar1, ycar1, xcar2, ycar2, True)
    return (-1, -1, -1, -1, False)


def number_plates_reader(img ,vehicle_detector, plate_detector, character_detector, character_recogniser, read_even_if_no_car=False, just_number_plates=False, show_captured_plates=False, show_individual_characters=False): #outputs the list of number plates(text) along with the car bounding box and number plate bounding box
    vehicle_detections = vehicle_detector(img)[0].boxes.data.tolist()
    vehicle_detections = [d for d in vehicle_detections if int(d[5]) in [2,3,5,7]] #filtering only vehicles
    license_plate_detections = plate_detector(img)[0].boxes.data.tolist()
    outputs = []
    def preprocess_plate(x1,y1,x2,y2):
        processed_plate = img[int(y1):int(y2), int(x1): int(x2), :]
        # processed_plate = cv2.adaptiveThreshold(cv2.cvtColor(processed_plate, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        return processed_plate
    number_plates_text = []
    for license_plate in license_plate_detections:
        x1, y1, x2, y2, _,_ = license_plate
        xcar1, ycar1, xcar2, ycar2, car_found = get_corresponding_car(license_plate, vehicle_detections)
        processed_plate = preprocess_plate(x1,y1,x2,y2)
        if(show_captured_plates):
            plt.imshow(processed_plate)
            plt.show()
        if (not car_found ) and (not read_even_if_no_car):
            continue

        license_plate_text = get_text_from_plate(processed_plate, character_detector, character_recogniser, show_individual_characters= show_individual_characters)
        if license_plate_text is not None:
            number_plates_text.append(license_plate_text)
            outputs.append({'car': {'box': [xcar1, ycar1, xcar2, ycar2]},'plate': {'box': [x1, y1, x2, y2],'plate_text': license_plate_text}})
    if(len(outputs)==0):
        for license_plate in license_plate_detections:
            x1, y1, x2, y2, _,_ = license_plate
            xcar1, ycar1, xcar2, ycar2, car_found = get_corresponding_car(license_plate, vehicle_detections)
            processed_plate = preprocess_plate(x1,y1,x2,y2)
            license_plate_text = get_text_from_plate(processed_plate, character_detector, character_recogniser)
            if license_plate_text is not None:
                number_plates_text.append(license_plate_text)
                outputs.append({'car': {'box': [xcar1, ycar1, xcar2, ycar2]},'plate': {'box': [x1, y1, x2, y2],'plate_text': license_plate_text}})
    
    if(just_number_plates):
        return number_plates_text
    
    return outputs


if __name__ == "__main__":
    vehicle_detector = YOLO('yolov8n.pt', verbose=False)
    plate_detector = YOLO('license_plate_detector.pt', verbose=False)
    character_detector = YOLO('Charcter-LP.pt', verbose=False)

    character_recogniser_model = CharCNN(num_classes=36)
    character_recogniser_model.load_state_dict(torch.load('char_cnn.pth'))
    character_recogniser_model.eval()
    character_recogniser = get_character_recogniser(character_recogniser_model)

    # count=0
    # for dirpath, dirnames, files in os.walk('./images/'):
    #     for file in files:
    #         if(count>=15):
    #             break
    #         if file.endswith('.jpg') or file.endswith('.png'):
    #             count+=1
    #             print(f"Results for {file}:")
    #             img = cv2.imread(os.path.join(dirpath, file))
    #             outputs = number_plates_reader(img, vehicle_detector, plate_detector, character_detector, character_recogniser,just_number_plates=True)
    #             for output in outputs:
    #                 print(output)
    file_name = 'car0.jpg'
    img = cv2.imread(file_name)
    outputs = number_plates_reader(img, vehicle_detector, plate_detector, character_detector, character_recogniser,just_number_plates=True)
    print("Result for", file_name, ":", outputs)