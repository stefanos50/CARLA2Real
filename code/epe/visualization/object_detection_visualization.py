import csv
import cv2
from PIL import Image
import numpy as np
import argparse
import os
import random
import glob
import xmltodict

parser = argparse.ArgumentParser(description='This script can be used to visualize the saved object detection annotations.')
parser.add_argument('--img_path', action='store', help='The path where the CARLA frames (enhanced or rgb) are stored.')
parser.add_argument('--annotations_path', action='store', help='The path where the object detection annotations are stored.')
parser.add_argument('--save_path', action='store', help='The path where the result images will be stored. If None then the result of each frame will be displayed in a open-cv window.')

args = parser.parse_args()

print(args.img_path)
if (args.img_path is None) or not os.path.isdir(args.img_path):
    print('--img_path argument is not set. Please provide a valid path in the disk where the CARLA frames are stored.')
    exit(1)

if (args.annotations_path is None) or not os.path.isdir(args.annotations_path):
    print('--annotations_path argument is not set. Please provide a valid path in the disk where the object detection annotations are stored.')
    exit(1)
save_result = False
if (args.save_path is not None):
    save_result = True
    if not os.path.isdir(args.save_path):
        print('--save_path argument is not valid. Please provide a valid path in the disk.')
        exit(1)

def generate_random_color():
    red = random.randint(0, 255)
    green = random.randint(0, 255)
    blue = random.randint(0, 255)

    return (red, green, blue)

image_dir = args.img_path
coco_xml_dir = args.annotations_path

class_color = {"vehicle":generate_random_color(),"truck":generate_random_color(),"traffic_light":generate_random_color(),"traffic_sign":generate_random_color(),"bus":generate_random_color(),"person":generate_random_color(),"motorcycle":generate_random_color(),"bicycle":generate_random_color(),"rider":generate_random_color()}
files = list(filter(os.path.isfile, glob.glob(coco_xml_dir + "\*")))
files.sort(key=os.path.getctime)
files.reverse()
for filename in files:
    filename = os.path.basename(filename)
    if filename.endswith(".xml"):
        with open(os.path.join(coco_xml_dir, filename), 'r') as file:
            data = xmltodict.parse(file.read())
    try:
        image = cv2.imread(image_dir+filename.split(".")[0]+".png")
        for bb in data['annotation']['object']:
            start_point = (int(float(bb['bndbox']['xmax']))), (int(float(bb['bndbox']['ymax'])))
            end_point = (int(float(bb['bndbox']['xmin'])), int(float(bb['bndbox']['ymin'])))
            color = class_color[bb['name']]
            thickness = 2
            x_min = int(float(bb['bndbox']['xmin']))
            y_min = int(float(bb['bndbox']['ymin']))
            x_max = int(float(bb['bndbox']['xmax']))
            y_max = int(float(bb['bndbox']['ymax']))

            image = cv2.rectangle(image, start_point, end_point, color, thickness)
            cv2.putText(image, bb['name'], (end_point[0], end_point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        print(filename)
        if save_result == False:
            cv2.imshow("image", image)
            cv2.waitKey(0)
        else:
            cv2.imwrite(os.path.join(args.save_path,filename.split(".")[0]+".png"), image)
    except Exception as err:
        print(err)
        pass