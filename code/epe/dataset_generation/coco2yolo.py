import os
import xmltodict
import cv2
import argparse

#The code is based on this solution https://medium.com/@WamiqRaza/convert-coco-format-annotations-to-yolo-format-4380880d9b3b
def is_single(dict):
    try:
        x = dict['name']
        return True
    except:
        return False


def convert_coco_to_yolo(coco_xml_dir, yolo_txt_dir, class_mapping):
    for filename in os.listdir(coco_xml_dir):
        if filename.endswith(".xml"):
            with open(os.path.join(coco_xml_dir, filename), 'r') as file:
                data = xmltodict.parse(file.read())

            img_width = int(data['annotation']['size']['width'])
            img_height = int(data['annotation']['size']['height'])

            yolo_txt_path = os.path.join(yolo_txt_dir, filename.replace(".xml", ".txt"))

            with open(yolo_txt_path, 'w') as file:
                try:
                    if is_single(data['annotation']['object']):
                        obj = data['annotation']['object']
                        class_name = obj['name']
                        print(class_name)
                        class_id = class_mapping.get(class_name)

                        if class_id is not None:
                            x_min = int(float(obj['bndbox']['xmin']))
                            y_min = int(float(obj['bndbox']['ymin']))
                            x_max = int(float(obj['bndbox']['xmax']))
                            y_max = int(float(obj['bndbox']['ymax']))

                            x_center = (x_min + x_max) / 2 / img_width
                            y_center = (y_min + y_max) / 2 / img_height
                            width = (x_max - x_min) / img_width
                            height = (y_max - y_min) / img_height

                            file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                    else:
                        for obj in data['annotation']['object']:
                            class_name = obj['name']
                            class_id = class_mapping.get(class_name)

                            if class_id is not None:
                                x_min = int(float(obj['bndbox']['xmin']))
                                y_min = int(float(obj['bndbox']['ymin']))
                                x_max = int(float(obj['bndbox']['xmax']))
                                y_max = int(float(obj['bndbox']['ymax']))

                                x_center = (x_min + x_max) / 2 / img_width
                                y_center = (y_min + y_max) / 2 / img_height
                                width = (x_max - x_min) / img_width
                                height = (y_max - y_min) / img_height

                                file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                except:
                    pass


parser = argparse.ArgumentParser(description='This script can be used to preprocess object detection annotations to yolo compatible format.')
parser.add_argument('--annotations_path', action='store', help='The path where the object detection annotations are stored.')
parser.add_argument('--save_path', action='store', help='The path where the preprocessed annotations will be stored.')

args = parser.parse_args()


if (args.annotations_path is None) or not os.path.isdir(args.annotations_path):
    print('--annotations_path argument is not set. Please provide a valid path in the disk where the annotations are stored.')
    exit(1)
if (args.save_path is None) or not os.path.isdir(args.save_path):
    print('--save_path argument is not valid. Please provide a valid path in the disk.')
    exit(1)

class_mapping = {'person': 0,'rider': 1,'vehicle': 2,'bicycle': 3,'motorcycle': 4,'bus': 5,'truck': 6}

coco_xml_dir = args.annotations_path
yolo_txt_dir = args.save_path

convert_coco_to_yolo(coco_xml_dir, yolo_txt_dir, class_mapping)
