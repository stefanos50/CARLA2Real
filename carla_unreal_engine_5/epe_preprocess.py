import argparse
import ast
import sys

import cv2
import numpy as np
import os

multi_gt_labels = np.array([])

def split_gt_label(gt_labels):
    r = (multi_gt_labels == gt_labels[:, :, 2][:, :, np.newaxis].astype(np.float32))


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
    return concatenated_array

def initialize_gt_labels(width=960,height=540,num_channels=29):
    global multi_gt_labels
    specific_classes = [11,1,2,24,25,27,14,15,16,17,18,19,10,9,12,13,6,7,8,21,23,20,3,4,5,26,28,0,22]
    multi_channel_array = np.zeros((num_channels, height, width))
    for channel_index, value in enumerate(specific_classes):
        multi_channel_array[channel_index, :, :] = value
    multi_gt_labels = np.transpose(multi_channel_array, axes=(1, 2, 0))


def get_all_files_in_path(directory_path):
    files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
    return files

def create_out_dir(path):
    if not os.path.exists(path):
        os.makedirs(os.path.join(path, "CarlaUE5-EPE"))
    data_path = os.path.join(path, "CarlaUE5-EPE")
    fpath = os.path.join(data_path, "Frames")
    gbuffpath = os.path.join(data_path, "GBuffers")
    labelspath = os.path.join(data_path, "SemanticSegmentation")
    if not os.path.exists(fpath):
        os.makedirs(fpath)
    if not os.path.exists(gbuffpath):
        os.makedirs(gbuffpath)
    if not os.path.exists(labelspath):
        os.makedirs(labelspath)
    return fpath,gbuffpath,labelspath

def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--input_path", type=str, default="A:/UE5Dataset", help="Path to the input directory")
    parser.add_argument("--output_path", type=str, default="A:/", help="Path to the output directory")
    parser.add_argument("--gbuffers", type=str, default="['SceneColor','SceneDepth','WorldNormal','Metallic','Specular','Roughness','BaseColor','SubsurfaceColor']", help="The GBuffers names that define the images that will be used to construct the GBuffer matrix.")
    parser.add_argument("--gbuffers_grayscale", type=str, default="['SceneDepth','Metallic','Specular','Roughness']", help="The GBuffers names that define images that are grayscale (Depth,Metallic,etc.).")
    args = parser.parse_args()

    if not os.path.isdir(args.input_path):
        print("Error: Input path does not exist.")
        sys.exit(1)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    return args



args = parse_args()

input_dataset_path = args.input_path
out_dataset_path = args.output_path
out_frame_path,out_gbuffer_path,out_label_path = create_out_dir(out_dataset_path)
image_idx = 0

buffers = ast.literal_eval(args.gbuffers)
grayscale_gbuffers = ast.literal_eval(args.gbuffers_grayscale)
frames_path = os.path.join(input_dataset_path, "Frames")
gt_map_path = os.path.join(input_dataset_path, "Semantic")


while True:
    image_idx += 1
    try:
        image = cv2.imread(os.path.join(frames_path, str(image_idx)+".png"))
        height, width = image.shape[:2]

        if multi_gt_labels.size == 0:
            initialize_gt_labels(width=width,height=height,num_channels=29)

        gbuff = []
        for i in range(len(buffers)):
            buffer_image = cv2.imread(os.path.join(frames_path, str(image_idx)+"_"+buffers[i]+".png"))
            if buffers[i] in grayscale_gbuffers:
                buffer_image = buffer_image[:,:,0]
                buffer_image = np.expand_dims(buffer_image, axis=-1)
            gbuff.append(buffer_image)

        gt_map = cv2.imread(os.path.join(gt_map_path, "segmentation_"+str(image_idx)+".png"))
        label_map = split_gt_label(gt_map)

        ssao = random_array = np.random.rand(height, width, 1)
        gbuff.append(ssao)
        gbuff.append(gt_map[:, :, 2][:, :, np.newaxis])


        gbuffers = np.concatenate(gbuff, axis=2)
        print("Processed Image: " + str(np.array(image).shape))
        print("Processed Gbuffers: "+str(gbuffers.shape))
        print("Processed Masks: " + str(label_map.shape))
        cv2.imwrite(os.path.join(out_frame_path, "FinalColor-"+str(image_idx))+".png",np.array(image))
        np.savez_compressed(os.path.join(out_gbuffer_path, "GBuffer-"+str(image_idx)+".npz"),gbuffers)
        np.savez_compressed(os.path.join(out_label_path, "SemanticSegmentation-"+str(image_idx)+".npz"),label_map)
    except Exception as e:
            print(e)
            break


