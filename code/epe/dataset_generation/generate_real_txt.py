import os
import argparse

parser = argparse.ArgumentParser(description='Warning: The image and robust directories must contain images with the same name and file type (extension). In any other case this script needs modification.')
parser.add_argument('--images_directory', action='store', help='The path where the real images are stored.')
parser.add_argument('--robust_directory', action='store', help='The path where the robust images are stored.')
parser.add_argument('--out_path', action='store', help='The path where the txt files will be saved.')

args = parser.parse_args()

if (args.images_directory is None) or not os.path.isdir(args.images_directory):
    print('--images_directory argument is not set. Please provide a valid path in the disk where the real images are stored.')
    exit(1)
if (args.robust_directory is None) or not os.path.isdir(args.robust_directory):
    print('--robust_directory argument is not set. Please provide a valid path in the disk where the robust labels are stored.')
    exit(1)
if (args.out_path is None) or not os.path.isdir(args.out_path):
    print('--out_path argument is not set. Please provide a valid path.')
    exit(1)

if args.images_directory[-1] != '/' and args.images_directory[-1] != '\\':
    args.images_directory += '/'
if args.out_path[-1] != '/' and args.out_path[-1] != '\\':
    args.out_path += '/'
if args.robust_directory[-1] != '/' and args.robust_directory[-1] != '\\':
    args.robust_directory += '/'

paths_list = []

for root, dirs, files in os.walk(os.path.abspath(args.images_directory)):
    for file in files:
        paths_list.append(os.path.join(args.images_directory, file)+","+os.path.join(args.robust_directory, file))

# Open the file in write mode and write each element to a new line
with open(args.out_path+"files.txt", 'w') as file:
    for element in paths_list:
        file.write(element + '\n')