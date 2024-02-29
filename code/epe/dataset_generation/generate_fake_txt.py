import os
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_directory', action='store', help='The path where the synthetic (fake) dataset is stored.')
parser.add_argument('--out_path', action='store', help='The path where the txt files will be saved.')
parser.add_argument('--probability', action='store',type=float, help='The probability where an image will be used as validation. Higher probability means a bigger validation set')

args = parser.parse_args()

if (args.dataset_directory is None) or not os.path.isdir(args.dataset_directory):
    print('--dataset_directory argument is not set. Please provide a valid path in the disk where the synthetic dataset is stored.')
    exit(1)
if (args.out_path is None) or not os.path.isdir(args.out_path):
    print('--out_path argument is not set. Please provide a valid path.')
    exit(1)
if (args.probability is None):
    print('--probability argument is not set. Please provide a valid probability.')
    exit(1)

if args.dataset_directory[-1] != '/' and args.dataset_directory[-1] != '\\':
    args.dataset_directory += '/'
if args.out_path[-1] != '/' and args.out_path[-1] != '\\':
    args.out_path += '/'

paths_list_train = []
paths_list_val_test = []
for root, dirs, files in os.walk(os.path.abspath(args.dataset_directory+"Frames")):
    for file in files:
        frame_id = file.split("-")[1].split(".")[0]
        if random.random() < args.probability:
            if file.startswith("__"):
                paths_list_val_test.append(os.path.join(args.dataset_directory+'Frames', file) + "," + os.path.join(args.dataset_directory+'RobustImages',file) + "," + os.path.join(args.dataset_directory+'GBuffers', "__GBuffer-" + str(frame_id) + ".npz") + "," + os.path.join(args.dataset_directory+'SemanticSegmentation', "__SemanticSegmentation-" + str(frame_id) + ".npz"))
            else:
                paths_list_val_test.append(os.path.join(args.dataset_directory+'Frames', file) + "," + os.path.join(args.dataset_directory+'RobustImages',file) + "," + os.path.join(args.dataset_directory+'GBuffers', "GBuffer-" + str(frame_id) + ".npz") + "," + os.path.join(args.dataset_directory+'SemanticSegmentation', "SemanticSegmentation-" + str(frame_id) + ".npz"))
        else:
            if file.startswith("__"):
                paths_list_train.append(os.path.join(args.dataset_directory+'Frames', file) + "," + os.path.join(args.dataset_directory+'RobustImages',file) + "," + os.path.join(args.dataset_directory+'GBuffers', "__GBuffer-" + str(frame_id) + ".npz") + "," + os.path.join(args.dataset_directory+'SemanticSegmentation', "__SemanticSegmentation-" + str(frame_id) + ".npz"))
            else:
                paths_list_train.append(os.path.join(args.dataset_directory+'Frames', file) + "," + os.path.join(args.dataset_directory+'RobustImages',file) + "," + os.path.join(args.dataset_directory+'GBuffers', "GBuffer-" + str(frame_id) + ".npz") + "," + os.path.join(args.dataset_directory+'SemanticSegmentation', "SemanticSegmentation-" + str(frame_id) + ".npz"))



with open(args.out_path+'train.txt', 'w') as file:
    for element in paths_list_train:
        file.write(element + '\n')

with open(args.out_path+'val.txt', 'w') as file:
    for element in paths_list_val_test:
        file.write(element + '\n')