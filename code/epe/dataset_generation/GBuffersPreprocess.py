import numpy as np
import os
import argparse
def is_grayscale(image_array):
    return (image_array.ndim == 2) or np.all(image_array[:,:,0] == image_array[:,:,1]) and np.all(image_array[:,:,0] == image_array[:,:,2])

def save_gbuffer(file_path,save_path,name):
    gbuff = np.load(file_path)
    gbuffers_list= [gbuff['SceneColor'], gbuff['SceneDepth'][:, :, 0][:, :, np.newaxis], gbuff['GBufferA'], gbuff['GBufferB'], gbuff['GBufferC'], gbuff['GBufferD'],gbuff['GBufferSSAO'][:, :, 0][:, :, np.newaxis], gbuff['CustomStencil'][:, :, 0][:, :, np.newaxis]]
    stacked_image = np.concatenate(gbuffers_list, axis=2)
    print("Saved gbuffer "+name+" with shape "+str(stacked_image.shape))
    np.savez_compressed(save_path+"/"+name+".npz",stacked_image)

parser = argparse.ArgumentParser()
parser.add_argument('--g_buffer_directory', action='store', help='The path where the gbuffers are stored.GBuffersCompressed directory that is generated from the generate_dataset.py script is a valid path.')
parser.add_argument('--save_path', action='store', help='The path where the preprocessed gbuffers will be stored.')

args = parser.parse_args()

if (args.save_path is None) or not os.path.isdir(args.save_path):
    print('--save_path argument is not set. Please provide a valid path in the disk where the preprocessed gbuffers will be stored.')
    exit(1)
if (args.g_buffer_directory is None) or not os.path.isdir(args.g_buffer_directory):
    print('--g_buffer_directory argument is not set. Please provide a valid path.')
    exit(1)

gbuffer_directory = args.g_buffer_directory
save_directory = args.save_path

for root, dirs, files in os.walk(os.path.abspath(gbuffer_directory)):
    for file in files:
        save_gbuffer(os.path.join(root, file),save_directory,file.split(".")[0])