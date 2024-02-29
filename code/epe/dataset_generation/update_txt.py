import os
import argparse

def update_drive(path,letter):
    drive, rest_of_path = os.path.splitdrive(path)
    new_path = letter+":"+ rest_of_path
    return new_path

def update(input_dir, output_dir, disk_letter='C'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.txt')):
                input_path = os.path.join(root, file)
                output_path = os.path.join(output_dir, file)
                dataset_paths = []
                with open(input_path, 'r') as file:
                    for line in file:
                        line = line.strip()
                        frames,robust,gbuffers,ground_truth = line.split(",")
                        dataset_paths.append(update_drive(frames,disk_letter)+","+update_drive(robust,disk_letter)+","+update_drive(gbuffers,disk_letter)+","+update_drive(ground_truth,disk_letter))

                with open(output_path, 'w') as file:
                    for pth in dataset_paths:
                        file.write(f"{pth}\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='This script converts a txt file with dataset paths to the specific disk letter that the dataset is stored.')
    parser.add_argument('--txt_directory', action='store', help='The path where the txt files are stored.')
    parser.add_argument('--out_path', action='store', help='The path where the updated txt files will be stored.')
    parser.add_argument('--disk_name', action='store', help='The letter of the disk.')

    args = parser.parse_args()

    if (args.txt_directory is None) or not os.path.isdir(args.txt_directory):
        print(
            '--txt_directory argument is not set. Please provide a valid path in the disk where the txt are stored.')
        exit(1)
    if (args.out_path is None) or not os.path.isdir(args.out_path):
        print('--out_path argument is not set. Please provide a valid path.')
        exit(1)

    if (args.disk_name is None):
        print('--disk_name argument is not set. Please provide a valid disk letter.')
        exit(1)
    update(args.txt_directory, args.out_path, args.disk_name)