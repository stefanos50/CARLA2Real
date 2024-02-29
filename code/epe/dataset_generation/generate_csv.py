import os
import csv
import argparse

extra_data_paths = []

def create_csv(input_dir, output_dir):
    paths_list = []
    if not os.path.exists(input_dir):
        print("Directory not found...")
        return


    for root, dirs, files in os.walk(os.path.abspath(input_dir)):
        for file in files:
            paths_list.append(os.path.join(root, file))

    with open(output_dir+"images_paths.csv", 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        for item in paths_list:
            csv_writer.writerow([item])



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_directory', action='store',help='The path where the synthetic (fake) dataset is stored.')
    parser.add_argument('--out_path', action='store', help='The path where the txt files will be saved.')

    args = parser.parse_args()

    if (args.dataset_directory is None) or not os.path.isdir(args.dataset_directory):
        print('--dataset_directory argument is not set. Please provide a valid path in the disk where the dataset is stored.')
        exit(1)
    if (args.out_path is None) or not os.path.isdir(args.out_path):
        print('--out_path argument is not set. Please provide a valid path.')
        exit(1)

    if args.dataset_directory[-1] != '/' and args.dataset_directory[-1] != '\\':
        args.dataset_directory += '/'
    if args.out_path[-1] != '/' and args.out_path[-1] != '\\':
        args.out_path += '/'

    create_csv(args.dataset_directory , args.out_path)