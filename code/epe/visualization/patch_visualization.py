import csv
import cv2
from PIL import Image
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description='This script can be used to visualize the patches that are generated after the matching procedure.')
parser.add_argument('--csv_path', action='store', help='The path where the csv file containing the patches is stored.')

args = parser.parse_args()

if (args.csv_path is None) or not os.path.isfile(args.csv_path):
    print('--csv_path argument is not set. Please provide a valid path in the disk where csv file is stored.')
    exit(1)


with open(args.csv_path, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        content = row[0].split(",")
        if content[0] == 'src_path':
            continue


        image_fake = cv2.imread(content[0])
        window_name = 'Patches Visualization'
        start_point = (int(content[3]), int(content[1]))
        end_point = (int(content[4]), int(content[2]))
        color = (255, 0, 0)
        thickness = 3
        image_fake = cv2.rectangle(image_fake, start_point, end_point, color, thickness)

        image_real = cv2.imread(content[5])
        start_point_point = (int(content[8]), int(content[6]))
        end_point = (int(content[9]), int(content[7]))
        image_real = cv2.rectangle(image_real, start_point, end_point, color, thickness)

        if not (image_real.shape == image_fake.shape):
            cv2.imshow(window_name, image_real)
            cv2.waitKey(0)
            cv2.imshow(window_name, image_fake)
            cv2.waitKey(0)
        else:
            merged_images = np.concatenate((image_real, image_fake), axis=1)
            cv2.imshow(window_name, merged_images)
            cv2.waitKey(0)