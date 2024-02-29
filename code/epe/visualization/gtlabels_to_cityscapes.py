import cv2
import numpy as np
import os
import argparse

CITYSCAPES_PALETTE_MAP = [
[0, 0, 0],
[128, 64, 128],
[244, 35, 232],
[70, 70, 70],
[102, 102, 156],
[190, 153, 153],
[153, 153, 153],
[250, 170, 30],
[220, 220, 0],
[107, 142, 35],
[152, 251, 152],
[70, 130, 180],
[220, 20, 60],
[255, 0, 0],
[0, 0, 142],
[0, 0, 70],
[0, 60, 100],
[0, 80, 100],
[0, 0, 230],
[119, 11, 32],
[110, 190, 160],
[170, 120, 50],
[55, 90, 80],
[45, 60, 150],
[157, 234, 50],
[81, 0, 81],
[150, 100, 100],
[230, 150, 140],
[180, 165, 180]
]

parser = argparse.ArgumentParser(description='This script converts a set of generated ground truth labels from carla to cityscapes palette colors.')
parser.add_argument('--labels_directory', action='store', help='The path where the ground truth labels are stored.')
parser.add_argument('--out_path', action='store', help='The path where the colorized gt labels will be stored.')

args = parser.parse_args()

if (args.labels_directory is None) or not os.path.isdir(args.labels_directory):
    print('--labels_directory argument is not set. Please provide a valid path in the disk where the gt labels are stored.')
    exit(1)
if (args.out_path is None) or not os.path.isdir(args.out_path):
    print('--out_path argument is not set. Please provide a valid path.')
    exit(1)

if args.labels_directory[-1] != '/' and args.labels_directory[-1] != '\\':
    args.labels_directory += '/'
if args.out_path[-1] != '/' and args.out_path[-1] != '\\':
    args.out_path += '/'


for root, dirs, files in os.walk(os.path.abspath(args.labels_directory)):
    for file in files:
        image_data = cv2.imread(os.path.join(root, file))
        semantic = np.zeros((image_data.shape[0], image_data.shape[1], 3))
        label_map = image_data[:, :, 2][:, :, np.newaxis].astype(np.float32)
        print(label_map.shape)
        for i in range(len(label_map)):
            for j in range(len(label_map[i])):
                label = int(label_map[i][j][0])
                semantic[i][j][0] = CITYSCAPES_PALETTE_MAP[label][0]
                semantic[i][j][1] = CITYSCAPES_PALETTE_MAP[label][1]
                semantic[i][j][2] = CITYSCAPES_PALETTE_MAP[label][2]
        cv2.imwrite(args.out_path+str(file), semantic)