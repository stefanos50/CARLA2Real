
# CARLA2Real: a tool for reducing the sim2real gap in CARLA simulator

This project originated as part of a Master's Thesis within the "Digital Media - Computational Intelligence" program at the Computer Science Department of Aristotle University of Thessaloniki.

![Project Pipeline](https://drive.google.com/uc?export=view&id=1nnBIp1JD9QGzsk42XoZjoBKBVeH74b8- )

### Objective
The primary aim of this project is to enhance the photorealism of the CARLA simulator output in real-time. Leveraging the [**Enhancing Photorealism Enhancement**](https://github.com/isl-org/PhotorealismEnhancement) project developed by Intel Labs, our goal is to reduce the SIM2REAL appearance gap. This gap is persistent when training Deep Learning (DL) algorithms in the simulator and deploying them in real-world scenarios.

<div align="center">
  <img src="https://drive.google.com/uc?export=view&id=1MhYfYPcL1MrK67tgpVj78Xd5Fr-T0T69" alt="Image" width="900px" height="auto">
</div>

### Features

* Out-of-the-box compatibility, and code for training an enhancement model from CARLA to any real or synthetic dataset.
* Code for extracting a synthetic dataset (including G-Buffers) in synchronous mode.
* Parameterization from a yaml config file rather than directly interacting with the code.
* Dataset generation of RGB and their corresponding enhanced frames with annotations for Semantic Segmentation, Object Detection, and Autonomous Driving.
* A simplified version of the scenario runner for creating specific scenarios.
* Support for implementing and evaluating autonomous driving algorithms or tasks (such as semantic segmentation, object detection, etc.) through simplified subscripts for code simplicity and easier integration into a unified structure.
* Support for different compilers (Pytorch, ONNX Runtime, TensorRT) based on ONNX structure to accommodate various data types (FP32, FP16, TF16, and INT8), enhancing performance and expanding compatibility with a wide range of supported hardware.
* Support for synchronous and asynchronous modes.

### Data
This section provides a short summary of all the available data that are provided and are available for download through Google Drive (Original Frames, Enhanced Frames, and Ground Truth Annotations).

1. [**CARLA Enhanced Synthetic Dataset (Cityscapes, Mapillary Vistas, KITTI, and GTA - 80.000 Frames)**](https://drive.google.com/file/d/1iNuOP_5xCSBdM-jgs1iXflaq0AUTykbR/view?usp=sharing)
2. [**CARLA2Cityscapes**](https://drive.google.com/file/d/1Fj8WsWWzPVoNBez8Glgw65GGXAg9eIzM/view?usp=drive_link)
3. [**CARLA2KITTI**](https://drive.google.com/file/d/18KQW_KeA9HhyF2A6jwCRnhVvbW3CA3qf/view?usp=drive_link)
4. [**CARLA2GTA**](https://drive.google.com/file/d/1zT7iZFHeTlDYSXEHIM6EbMtHDCsF3UcX/view?usp=drive_link)

### BibTeX Citation

If you used the CARLA2Real tool or any dataset from this repository in a scientific publication, we would appreciate using the following citations:

```
The paper associated with this code has not yet been published. Please cite this repository if you use the code.
% Placeholder citation
@article{author_year,
  title = {CARLA2Real: a tool for reducing the sim2real gap in CARLA simulator},
  author = {Stefanos Pasios, Nikos Nikolaidis},
  journal = {Unpublished},
  year = {Year},
}

@Article{Richter_2021,
                    title = {Enhancing Photorealism Enhancement},
                    author = {Stephan R. Richter and Hassan Abu AlHaija and Vladlen Koltun},
                    journal= {arXiv:2105.04619},
                    year = {2021},
                }
```

# Installitation/Requirements

### Hardware Requirements
This project was developed on a system equipped with an RTX 4090 24GB GPU, an Intel i7 13700KF CPU, and 32GB of system memory, achieving an inference delay of 0.3 seconds with pure PyTorch in FP32 or 0.12 seconds with TensorRT in FP16 precision. Depending on the compiler choice and precision type in the configuration, GPU memory requirements can range from 9.5GB to 22.3GB, necessitating a GPU with a minimum of 10GB of VRAM. While the model was built to translate images roughly at 960x540, reducing the resolution of the camera will result in better performance without significantly altering the final translation result.

To attain optimal performance in asynchronous mode, where the GPU has to render CARLA at a high refresh rate while performing inference on the enhanced model, multi-GPUs are recommended. Exploring parameters that are most suitable for your hardware is advisable, as other options, such as asynchronous data transfer, may result in high requirements for other components like the CPU.

### Operating System
The operating system used and tested for this project was Windows 11. However, it should be compatible with all the other [CARLA simulator-supported operating systems](https://carla.readthedocs.io/en/latest/).

### Installing CARLA
This project is based on the `listen_to_gbuffer()` API method that extracts and sends the G-Buffers that are utilized by the enhancement model to the Python client. This was introduced in CARLA 0.9.14, so this project won't work in older versions. You can download CARLA 0.9.14 from [here](https://carla.org/2022/12/23/release-0.9.14/).

> ⚠️ **Warning**: CARLA released version 0.9.15 in November. While this version supports the `listen_to_buffer` method, it appears to have bugs that can crash the engine from the server side when attempting to assign a listener for a specific G-Buffer. Therefore, if this issue is not resolved, it is recommended to stick with CARLA 0.9.14.

### Setup
First, download the files of this repository as a `Zip archive` and extract the `code` directory to a disk. Alternatively, you can clone the repository and follow the same steps by running the command:

```javascript
git clone http:/...........
```

To run the code, you need to download and set up [Anaconda](https://www.anaconda.com/download) on your computer. After that, create and set up a Conda environment. For detailed instructions, you can refer to the `Setup` section in the [Enhancing Photorealism Enhancement repository](https://github.com/isl-org/PhotorealismEnhancement/tree/main/code).

To install all the other dependencies after creating the conda environment, we include a `requirement.txt` file.

```javascript
conda create --name epe python=3.8
y
conda activate epe
cd /d <path-to>/code/
pip install -r requirements.txt
pip install -e ./
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
y
conda install -c conda-forge faiss-gpu
cd /d epe
```

If you encounter an error when selecting ONNX Runtime as the selected compiler through the `compiler` parameter inside `\code\config\carla_config.yaml` after installing the requirements then use the following commands:
```javascript
pip uninstall onnxruntime-gpu
y
pip install onnxruntime-gpu
```

> ⚠️ **Warning**: The project has been tested with the latest versions of [PyTorch](https://pytorch.org/) compiled with CUDA 11.7 and 11.8. For Ada Lovelace/Hopper architectures, using CUDA version <11.8 may result in significant performance loss. Building PyTorch with CUDA is mandatory, as CPU inference is not supported. ONNX Runtime and TensorRT may require different version based on the installed CUDA development kit version.

### Installing TensorRT

Nvidia's TensorRT is utilized to optimize and improve inference performance for handling a model of this size. While installing PyTorch and ONNX Runtime is straightforward using `pip install`, TensorRT requires additional steps for installation. Begin by checking if your hardware supports TensorRT and identifying any other dependencies required based on your hardware configuration. For instance, RTX 4090 works only with CUDA 11.8 (or newer) for TensorRT. Refer to the [**TensorRT support matrix**](https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html) for compatibility information.

If your hardware supports TensorRT, follow the [**installation guide**](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html) provided in the NVIDIA documentation. After installation, it is crucial to specify the path to the `common.py` script, which is bundled with TensorRT, by configuring the `tensorrt_common_path` parameter inside the CARLA config YAML file.

> 📝 **Note**: TensorRT is recommended for achieving the best inference performance, but it is not mandatory to run the code. PyTorch and ONNX Runtime can be selected through the `compiler` parameter inside the CARLA config file, and they do not require the installation of TensorRT.

> ⚠️ **Warning**: This project was developed and tested using TensorRT `version 8.6.1.6` exclusively.

# Training & Dataset

For training CARLA on another target dataset, we provide a dataset generation script that can export frames, G-buffers, and ground truth label maps in synchronous mode, along with various functionalities that introduce diversity. To create your own synthetic dataset, execute the following commands after running CARLA:

```javascript
cd /d K:\code\epe\dataset_generation
python generate_dataset.py --save_path E:\ --ticks_per_frame 10 --perspective 0 --randomize_lights True --town Town10HD
```
The data are extracted as PNG images to preserve their utilization on a wider variety of models. To transform them to a compatible format directly for the Enhancing Photorealism Enhancement model, we provide a set of preprocessing scripts with the following commands:

```javascript
python LabelMapPreprocess.py --label_map_directory E:\CarlaDataset\SemanticSegmentation --save_path E:\CarlaDataset\SemanticEPE
python GBuffersPreprocess.py --g_buffer_directory E:\CarlaDataset\GBuffersCompressed\ --save_path E:\CarlaDataset\GBuffersEPE\
```

We recommend utilizing the dataset in the specific structure shown below. This is because the `\dataset_generation\generate_fake_txt.py` script, responsible for generating the required TXT files containing all the dataset paths, expects the data to be organized in this particular format. In any other scenario, you will need to modify or create your own script for this task. The images in the `RobustImages` directory are generated by [**MSEG**](https://github.com/mseg-dataset/mseg-semantic) by employing the original `FinalColor` frames.

    .
    ├── Disk
    │   ├── CarlaDataset
    │   │   ├── Frames
    │   │   │   ├── FinalColor-000062.png
    │   │   │   └── ...
    │   │   ├── GBuffers
    │   │   │   ├── GBuffer-000062.npz
    │   │   │   └── ...
    │   │   ├── RobustImages
    │   │   │   ├── FinalColor-000062.png
    │   │   │   └── ...
    │   │   ├── SemanticSegmentation
    │   │   │   ├── SemanticSegmentation-000062.npz
    │   │   │   └── ...
    │   │   └── ...
    │   └── ...
    └── ...

For a detailed understanding of how to train with your own dataset and generate all the required files that the model expects, we recommend visiting the [**Enhancing Photorealism Enhancement repository**](https://github.com/isl-org/PhotorealismEnhancement/tree/main/code) and reading their [**paper**](https://arxiv.org/abs/2105.04619).

> 💡 **Hint**: Before running any of the scripts inside `\code\epe\dataset_generation`, use the `--help` argument to check the available parameters and obtain information about how to use them.

![GBUFFERS](https://drive.google.com/uc?export=view&id=1KOgYtS9h2_bao1nedSRH-oHdF5y66UpV )

After setting up the dataset, the txt files with the data paths and the patch-matching-generated files modify the `\code\config\train_pfd2cs.yaml` accordingly and run the following command to start the training procedure:

```javascript
python EPEExperiment.py train <path-to>\code\config\train_pfd2cs.yaml --log=info
```

# Testing

To translate an existing CARLA dataset outside the simulator after setting the desired model in `\code\config\test_pfd2cs.yaml` execute the following command:

```javascript
python EPEExperiment.py test <path-to>\code\config\test_pfd2cs.yaml --log=info
```

The input frames should be in text format. To generate such a txt file, you can use the `/code/epe/dataset_generation/generate_fake_txt.py` script. For further details, as in the case of training, we recommend referring to the [**Enhancing Photorealism Enhancement repository**](https://github.com/isl-org/PhotorealismEnhancement/tree/main/cod). The translated frames are saved in `\code\epe\out\pfd2cs\`.

> ⚠️ **Warning**: Translating a set of CARLA frames requires the corresponding G-Buffers and GT Labels as input; it is not possible to translate frames on their own.

# Real-Time Inference

To enhance the output of the CARLA simulator in real-time, ensure that CARLA is running either after building from source inside Unreal Engine 4 or by executing the CARLA executable. Then, after setting the desired model in `\code\config\infer_pfd2cs.yaml` (`weight_dir` and `name_load` parameters), execute the following command:

```javascript
python EPEExperiment.py infer <path-to>\code\config\infer_pfd2cs.yaml --log=info --carla_config <path-to>/code/config/carla_config.yaml
```
The `\code\config\carla_config.yaml` file contains all the available parameters, and it is mandatory to provide it as an argument when inferencing in real-time, but not in all other cases.

# Experimenting
For experimenting with our code, we provide a `carla_config` file containing most of the parameters, along with a number of Python samples. In the next chapters, we will present some of the basic functionalities. More in-depth information for each parameter or function can be found via comments in the `code` or next to each parameter inside the `yaml` files.

## Autonomous Driving Tasks
To experiment with autonomous driving algorithms, read the samples located in the `\code\epe\autonomous_driving\` directory. These scripts are called based on the specified parameters within the `carla_config.yaml` file. For instance, configuring `pygame_output` to `ad_task` will trigger the execution of `\code\epe\autonomous_driving\ad_task.py` to conduct a forward pass of a model responsible for semantic segmentation or object detection. It will then render the resulting output in the PyGame window. Experiments can be done with both the original simulator output and the enhanced counterpart. Moreover, the enhancement model can be entirely deactivated within the configuration file if it's not used to improve performance.

## Exporting a Synthetic Dataset

We provide functionality for exporting a dataset with CARLA frames and their corresponding translated ones. This can be useful for comparing algorithms between CARLA and photorealistic CARLA or training algorithms only on the enhanced frames. Models trained on enhanced frames are expected to perform better on real-life data compared to models trained on original frames due to a smaller sim2real gap. Currently, this feature is available only in synchronous mode because it is crucial for all data to be synchronized from the same frame ID to avoid an accuracy decrease or other issues in DL models.

Various data can be exported, including frames, information about the world (weather, etc.), vehicle information (speed, steer, brake, etc.), ground truth labels for semantic segmentation, object detection annotations in PASCAL VOC format, and depth information. All this information can be individually enabled or disabled to save disk space if some of it is not needed for a specific task. Use the parameters inside `\code\config\carla_config.yaml` to control these export options.

When exporting datasets, a common approach is to save data for every number of frames. This can be done by changing the parameter `skip_frames`. Exporting frames when the car is stuck at a traffic light or in any other location can be problematic because the dataset will end up with multiple frames containing the same information. For such cases, we provide the parameters `capture_when_static` and `speed_threshold`. If the parameter is set to false, then if the ego vehicle is static, data will not be saved. Also, by thresholding the speed of the vehicle, you can avoid exporting data if the vehicle is moving at a very slow speed to prevent having data that are very close. Adjust these parameters in the `\code\config\carla_config.yaml` file.

![Annotations](https://drive.google.com/uc?export=view&id=1NvzD1kOVAEG4j6MA9mcW2MV2WpoGay75 )

### Semantic Segmentation

CARLA generates ground truth labels based on the UNREAL ENGINE 4 `custom stencil G-Buffer`. Currently, in `version 0.9.14`, CARLA follows the `Cityscapes scheme`. The ground truth labels are saved in the default scheme of CARLA and can be modified through the engine or by preprocessing the images with another script for compatibility with other datasets. To modify the ground truth labels via the engine after building from the source, refer to the [documentation](https://carla.readthedocs.io/en/latest/tuto_D_create_semantic_tags/). To preprocess the labels, you can find the classes and their corresponding IDs inside the [Cityscapes palette source code](https://github.com/carla-simulator/carla/blob/master/LibCarla/source/carla/image/CityScapesPalette.h).

### Object Detection

For object detection, we offer nine distinct classes: `person, vehicle, truck, bus, motorcycle, bicycle, rider, traffic light, and traffic sign`. CARLA, however, lacks functionality to reference static meshes, such as bikers, for bounding boxes. Furthermore, `version 0.9.14` still exhibits some bugs related to bounding box accuracy. Since the engine is aware of the position of all objects in the virtual world, it does not consider occlusion when drawing bounding boxes for objects. To mitigate these issues, we provide parameters in `/code/config/carla_config.yaml`, primarily based on ground truth label masks alongside a semantic lidar sensor for occlusion checks, to allow for parametrization based on the specific problem at hand. Depending on the desired annotations, additional check statements or code modifications may be necessary. These adjustments can be made within `/code/epe/experiment/BaseExperiment.py`, specifically in the function `save_object_detection_annotations()`. To generate annotations for a specific model, such as YOLOv5, you can either modify the latter function or utilize a script that converts PASCAL VOC format to the target model's format.

> 📝 **Note**: We include the `coco2yolo.py` script inside `\code\epe\dataset_generation` that preprocesses the object detection annotation to a YOLO-compatible format, as it is the most common model used for such a task. The code is based on [this solution](https://medium.com/@WamiqRaza/convert-coco-format-annotations-to-yolo-format-4380880d9b3b) with some modifications.

> ⚠️ **Warning**: To aquire adequate results from the Semantic Lidar sensor, since the data extraction method executes only in synchronous mode, its essential to achieve proper synchronization between the sensor rotation and the world steps (`fixed_time_step` and `semantic_lidar_rotation_frequency` parameters). For more detailed information, refer to the [CARLA sensors documentation](https://carla.readthedocs.io/en/latest/ref_sensors/).

> ⚠️ **Warning**: Parked vehicles are not considered actors, and their extraction of bounding boxes is impossible via the Python API. In that case, use the Opt version of the selected town (Town10HD_Opt, Town01_Opt, etc.), and our implementation will remove all the parked vehicles from the world. Moreover, the Cybertruck vehicle seems to have issues with the bounding boxes, so we recommend filtering it for object detection dataset generation.

> 💡 **Hint**: To overcome the occlusion challenges for bounding box annotations, an ideal approach is to utilize the instance segmentation sensor since each object has its own unique semantic mask. Although CARLA provides such a sensor, there is no direct matching between the instance IDs and the actor IDs without modifying the source code. You can find more information in the following GitHub discussion [#5047](https://github.com/carla-simulator/carla/discussions/5047).

### Scenarios

We offer the creation of simple scenarios through a configuration YAML file, which can be set in `/code/config/carla_config.yaml` under the `scenario` parameter. Samples of such scenarios and their expected structure can be found within `/code/scenarios`. The scenarios can range from spawning a vehicle at a random or fixed location to creating a static obstacle for the ego vehicle to avoid, setting up a distance-based trigger between another object (vehicle or person) as a moving obstacle, manipulating traffic lights, or following a leading vehicle. Many parameters require coordinates, so it is suggested to build CARLA from source to extract coordinates through the engine. Additionally, using scenarios that involve triggers is recommended to run in `synchronous mode`.

```javascript
cross_road_town10.yaml example

ego_vehicle_settings:
  init_spawn_point: [[-41,56,2],[0,-90,0]] #random or any point id or a transform [[10,10,10],[1,1,1]] ([location,rotation])
other_actor:
  actor_id: vehicle.tesla.model3 #any vehicle class or person class from the documentation
  init_spawn_point: [[-26,31,0.2],[0,180,0]] #[location,rotation]
  static: False #If the other vehicle should move or not
  init_controls: [0,0,0] #steer,throttle,brake values before the trigger (if zeros then the object will be static)
  distance_threshold: 23 #the distance between the ego vehicle to trigger the action
  threshold_critiria: less_equals #the threshold distance critiria to trigger the action (greater,less,equals,less_equals,greater_equals,not_equals)
  out_controls: [0,1,0] #new controls after triggering the event based of the distance threshold
general:
  traffic_lights: green #green or red
  traffic_lights_time: 100 #time duration for the traffic light state
  val_ticks: 100 #number of ticks in synchronous mode to run the scenario. If set to None, it will run infinitely.
```

https://github.com/stefanos50/thesis_test_repository/assets/36155283/79f8db70-829f-4a1c-bc1d-71122751cf58

### Compilers and Data Types

The implementation currently supports inference with three different compilers: PyTorch, Nvidia's TensorRT, and ONNX Runtime. Choosing a compiler and data type can have an impact on performance, quality, and system requirements. The choice is available through the `compiler` and `data_type` parameters in `carla_config.yaml`. Using a lower precision will reduce the model size and speed up the inference. Dropping to FP16 showed good results while keeping the quality at a high level, and that is the recommended option. For TensorRT, it also supports the INT8 datatype, which is even smaller. In that case, to build the engine, it is required to provide a dataset for calibration to avoid losing quality to a high degree. The dataset can be provided in a `txt` format, as in the case of training, via the `calibration_dataset` parameter. The code checks if all the required files are present based on the compiler selection; if not, it will try to generate them.

In the case of building an ONNX file, it can take up to more than 100GB of system memory. In that case, if the system memory is insufficient but there is enough memory available on disk, then it will succeed after some time. In the case of TensorRT, building the engine or calibrating can also take some time, but the memory requirements are lower. Based on our experiments, we recommend using `TensorRT` with `FP16 precision` as it gives the best performance, quality, and memory ratio.

> ⚠️ **Warning**: ONNX Runtime with FP16 precision introduces image artifacts. It is recommended to use ONNX Runtime only if you want to achieve faster inference in FP32 with the tradeoff of high GPU memory requirements.

> 📝 **Note**: For INT8 precision, we had to manually set a significant part of the network, particularly the section that processes the G-Buffers, to FP16 to maintain high quality. As a result, the improvement in performance is relatively small.

> ⚠️ **Warning**: NVIDIA's TensorRT can result in issues when exporting ONNX files. We recommend selecting PyTorch or ONNX Runtime from the `compiler` parameter in that particular case.
### Data Augmentation

In addition to enhancing CARLA with a real-life dataset, we experimented with translating the data to another game, such as GTA, using the [**Playing for Data dataset**](https://download.visinf.tu-darmstadt.de/data/from_games/). This experiment demonstrated that translating the world to another game or simulator could be a rapid method for generating new data to train CARLA models, leading to higher accuracy and better generalization. Similar to the real datasets, we provide a small [**sample dataset**](https://drive.google.com/file/d/1zT7iZFHeTlDYSXEHIM6EbMtHDCsF3UcX/view?usp=drive_link) with translated images from CARLA to GTA V (Grand Theft Auto V). 

![carla2gta](https://drive.google.com/uc?export=view&id=1wCrzvFQcNsWVI44A12yQOnR8CEmgoJ90)

### Adding a New Sensor

To add a new sensor that is currently not implemented through the parameters in the `carla_config.yaml` file, this can happen in the  `def evaluate_infer(self)` inside the `code\epe\Experiment\BaseExperiment.py` script. The callback function that will be assigned to the new sensor should store the final data in the `data_dict` dictionary, which is a global dictionary that stores the data from all the active sensors of the ego vehicle. After adding a new sensor, it is also crucial to increment the `data_length` variable since, during the synchronization loop, the algorithm waits for all the data from the active sensors of the ego vehicle to arrive in the client. If the variable is not set properly, then the client will get stuck inside the synchronization loop, and the server will permanently freeze.

### Spawning Traffic and other functionalities

Our tool works with most of the samples that CARLA already provides in the `\CarlaEXE\PythonAPI\examples` directory. If you want to spawn a variety of vehicles and NPCs in the world after executing our approach, you can easily execute the provided `generate_traffic.py` script. The same is also applicable for dynamic weather via `dynamic_weather.py` and any other functionality that is already provided by the CARLA team.

> ⚠️ **Warning**: The CARLA sample scripts should be executed after running our tool in the same synchronization mode as the selected parameter through the `carla_config.yaml` file of our implementation (synchronous or asynchronous).
