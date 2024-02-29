import cv2
import numpy as np
import os
import imageio
import argparse

num_channels = 29
height = 540
width = 960
specific_classes = [11,1,2,24,25,27,14,15,16,17,18,19,10,9,12,13,6,7,8,21,23,20,3,4,5,26,28,0,22]
multi_channel_array = np.zeros((num_channels, height, width))
for channel_index, value in enumerate(specific_classes):
    multi_channel_array[channel_index, :, :] = value
multi_classes_array = np.transpose(multi_channel_array, axes=(1, 2, 0))

def material_from_gt_label(gt_labelmap):
	r = (multi_classes_array == gt_labelmap[:, :, 2][:, :, np.newaxis].astype(np.float32))

	class_sky = r[:, :, 0][:, :, np.newaxis]
	class_road = np.any(r[:, :, [1, 2, 3, 4, 5]], axis=2)[:, :, np.newaxis]
	class_vehicle = np.any(r[:, :, [6, 7, 8, 9, 10, 11]], axis=2)[:, :, np.newaxis]
	class_terrain = r[:, :, 12][:, :, np.newaxis]
	class_vegetation = r[:, :, 13][:, :, np.newaxis]
	class_person = np.any(r[:, :, [14, 15]], axis=2)[:, :, np.newaxis]
	class_infa = r[:, :, 16][:, :, np.newaxis]
	class_traffic_light = r[:, :, 17][:, :, np.newaxis]
	class_traffic_sign = r[:, :, 18][:, :, np.newaxis]
	class_ego = np.any(r[:, :, [19, 20]], axis=2)[:, :, np.newaxis]
	class_building = np.any(r[:, :, [21, 22, 23, 24, 25, 26]], axis=2)[:, :, np.newaxis]
	class_unlabeled = np.any(r[:, :, [27, 28]], axis=2)[:, :, np.newaxis]

	concatenated_array = np.concatenate((class_sky, class_road, class_vehicle, class_terrain, class_vegetation,
										 class_person, class_infa, class_traffic_light, class_traffic_sign, class_ego,
										 class_building, class_unlabeled), axis=2)
	return concatenated_array.astype(np.float32)

parser = argparse.ArgumentParser()
parser.add_argument('--label_map_directory', action='store', help='The path where the semantic segmentation images are stored.SemanticSegmentation directory that is generated from the generate_dataset.py script is a valid path.')
parser.add_argument('--save_path', action='store', help='The path where the preprocessed semantic segmentation label masks will be stored.')

args = parser.parse_args()

if (args.save_path is None) or not os.path.isdir(args.save_path):
    print('--save_path argument is not set. Please provide a valid path in the disk where the preprocessed gbuffers will be stored.')
    exit(1)
if (args.label_map_directory is None) or not os.path.isdir(args.label_map_directory):
    print('--label_map_directory argument is not set. Please provide a valid path.')
    exit(1)

label_map_directory = args.label_map_directory
save_directory = args.save_path

for root, dirs, files in os.walk(os.path.abspath(label_map_directory)):
	for file in files:
		gt_labels = material_from_gt_label(cv2.imread(os.path.join(root, file)))
		print("Saved label map masks " + file.split(".")[0] + " with shape " + str(gt_labels.shape))
		np.savez_compressed(save_directory + "/" + file.split(".")[0] + ".npz", gt_labels)