import argparse
import os
import time
import cv2
import numpy as np
import torch
from PIL import Image
import onnxruntime

def test(io_binding,frame_path,gbuffers_path,label_maps_path,file_name,out_path):

    img = Image.open(frame_path).convert('RGB')
    gbuffers = np.load(gbuffers_path)['arr_0']
    label_map = np.load(label_maps_path)['arr_0']
    img = np.array(img)

    img = np.expand_dims(img, axis=0)
    gbuffers = np.expand_dims(gbuffers, axis=0)
    label_map = np.expand_dims(label_map, axis=0)

    img = np.transpose(img, (0, 3, 1, 2)) / 255
    gbuffers = np.transpose(gbuffers, (0, 3, 1, 2))
    label_map = np.transpose(label_map, (0, 3, 1, 2))

    img = img.astype(np.float32)
    gbuffers = gbuffers.astype(np.float32)
    label_map = label_map.astype(np.float32)

    img = onnxruntime.OrtValue.ortvalue_from_numpy(img, 'cuda', 0)
    gbuffers = onnxruntime.OrtValue.ortvalue_from_numpy(gbuffers, 'cuda', 0)
    label_map = onnxruntime.OrtValue.ortvalue_from_numpy(label_map, 'cuda', 0)

    tdtype = np.float32
    io_binding.bind_input(name='input', device_type=img.device_name(), device_id=0,
                          element_type=tdtype,
                          shape=img.shape(), buffer_ptr=img.data_ptr())
    io_binding.bind_input(name='gbuffers', device_type=gbuffers.device_name(), device_id=0,
                          element_type=tdtype,
                          shape=gbuffers.shape(), buffer_ptr=gbuffers.data_ptr())
    io_binding.bind_input(name='onnx::Gather_2', device_type=label_map.device_name(), device_id=0,
                          element_type=tdtype,
                          shape=label_map.shape(), buffer_ptr=label_map.data_ptr())
    io_binding.bind_output('output', device_type='cuda', device_id=0, element_type=tdtype)

    infer_timer = time.time()
    session.run_with_iobinding(io_binding)
    print("Inference time: " + str(time.time() - infer_timer))
    new_img = torch.from_numpy(io_binding.copy_outputs_to_cpu()[0])
    enhanced_frame = (new_img[0, ...].clamp(min=0, max=1).permute(1, 2, 0) * 255.0).detach().cpu().numpy().astype(np.uint8)

    image = Image.fromarray(enhanced_frame)
    image.save(out_path+file_name)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_onnx', action='store',help='The path where the onnx file (trained model) is stored.')
    parser.add_argument('--dataset_directory', action='store',help='The path where the synthetic (fake) dataset is stored.')
    parser.add_argument('--out_path', action='store', help='The path where the enhanced images will be saved.')

    args = parser.parse_args()

    if (args.dataset_directory is None) or not os.path.isdir(args.dataset_directory):
        print('--dataset_directory argument is not set. Please provide a valid path in the disk where the dataset is stored.')
        exit(1)

    if (args.out_path is None) or not os.path.isdir(args.out_path):
        print('--out_path argument is not set. Please provide a valid path.')
        exit(1)

    if (args.model_onnx is None) or not os.path.isfile(args.model_onnx):
        print('--model_onnx argument is not set. Please provide a valid path of the trained model.')
        exit(1)

    if args.dataset_directory[-1] != '/' and args.dataset_directory[-1] != '\\':
        args.dataset_directory += '/'
    if args.out_path[-1] != '/' and args.out_path[-1] != '\\':
        args.out_path += '/'

    subdirectories = [d for d in os.listdir(args.dataset_directory) if os.path.isdir(os.path.join(args.dataset_directory, d))]

    if not ('Frames' in subdirectories and 'GBuffers' in subdirectories and 'SemanticSegmentation' in subdirectories):
        print("The dataset should be structured as CarlaDataset directory that contains Frames,GBuffers, and SemanticSegmentation subdirectories.")

    frames_dir = os.path.join(args.dataset_directory, 'Frames')
    gbuffers_dir = os.path.join(args.dataset_directory, 'GBuffers')
    semseg_dir = os.path.join(args.dataset_directory, 'SemanticSegmentation')

    print("Clearing older ONNX profiles...")
    file_list = os.listdir(os.getcwd())
    for filename in file_list:
        if "onnxruntime_profile__" in filename:
            file_path = os.path.join(os.getcwd(), filename)
            os.remove(file_path)
            print(f"Deleted: {file_path}")

    opts = onnxruntime.SessionOptions()
    opts.enable_profiling = True
    session = onnxruntime.InferenceSession(
        args.model_onnx,
        opts,
        providers=[
            (
                "CUDAExecutionProvider",
                {
                    "device_id": 0,
                    "cudnn_conv_use_max_workspace": "1",
                    "cudnn_conv_algo_search": "DEFAULT",
                },
            ),
            "CPUExecutionProvider",
        ],
    )
    io_binding = session.io_binding()

    files = [f for f in os.listdir(frames_dir) if os.path.isfile(os.path.join(frames_dir, f))]

    for file in files:
        frame_id = file.split("-")[1].split(".")[0]
        if file.startswith("__"):
            test(io_binding=io_binding,frame_path=os.path.join(frames_dir, file),gbuffers_path=os.path.join(gbuffers_dir, "__GBuffer-" + str(frame_id) + ".npz"),label_maps_path=os.path.join(semseg_dir, "__SemanticSegmentation-" + str(frame_id) + ".npz"),file_name=file,out_path=args.out_path)
        else:
            test(io_binding=io_binding, frame_path=os.path.join(frames_dir, file),
                 gbuffers_path=os.path.join(gbuffers_dir, "GBuffer-" + str(frame_id) + ".npz"),
                 label_maps_path=os.path.join(semseg_dir, "SemanticSegmentation-" + str(frame_id) + ".npz"),
                 file_name=file, out_path=args.out_path)


