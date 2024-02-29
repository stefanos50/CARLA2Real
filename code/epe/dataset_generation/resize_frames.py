import os
from PIL import Image
import argparse

def resize_images(input_dir, output_dir, target_resolution=(800, 600)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg')):
                input_path = os.path.join(root, file)
                output_path = os.path.join(output_dir, file)

                try:
                    img = Image.open(input_path)
                    img = img.resize(target_resolution, Image.BICUBIC)
                    img.save(output_path)
                    print(f"Resized and saved {input_path} to {output_path}")
                except Exception as e:
                    print(f"Error while processing {input_path}: {str(e)}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='This script converts a set of images to a lower resolution. This can be useful as the enhanced model works with frames roughly 960x540 (width x height).')
    parser.add_argument('--images_directory', action='store', help='The path where the frames are stored.')
    parser.add_argument('--out_path', action='store', help='The path where the resized frames will be stored.')
    parser.add_argument('--resolution', action='store', help='The new resolution of the frames (must be lower than the current resolution).')

    args = parser.parse_args()

    if (args.images_directory is None) or not os.path.isdir(args.images_directory):
        print(
            '--images_directory argument is not set. Please provide a valid path in the disk where the frames are stored.')
        exit(1)
    if (args.out_path is None) or not os.path.isdir(args.out_path):
        print('--out_path argument is not set. Please provide a valid path.')
        exit(1)

    if (args.resolution is None) or not isinstance(eval(args.resolution), tuple):
        print('--resolution argument is not set. Please provide a valid resolution as a python tuple (width,height).')
        exit(1)
    args.resolution = eval(args.resolution)
    resize_images(args.images_directory, args.out_path, args.resolution)