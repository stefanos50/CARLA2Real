# CARLA2Real: Unreal Engine 5 Update

![carla2kitti_ue5](https://github.com/user-attachments/assets/05b02ff9-f770-44fa-b188-4a281d182301)

![Screenshot 2025-01-24 015002](https://github.com/user-attachments/assets/9c2a3e76-0cf5-4606-9705-bbb564256b3e)



While our Unreal Engine 5 approach is easy to use and doesn’t require modifying the CARLA source code or project settings, you **must** build CARLA from source and run it through the Unreal Engine editor. The packaged version of CARLA locks the Unreal console, which is needed for this setup.
Building from source is simpler than before, and you can find detailed steps in the official [documentation](https://carla-ue5.readthedocs.io/en/latest/) and the [GitHub repository](https://github.com/carla-simulator/carla).

After building CARLA from source run the following command to run the Unreal editor:

```javascript
cmake --build Build --target launch
```

Download the packaged version from the [official github repository](https://github.com/carla-simulator/carla/releases) and install in an anaconda environment the carla 0.10.0 package (located in `\PythonAPI\carla`) based on your python version with the following command:

```javascript
pip install <WHEEL_FILE>
```

Run the simulator within Unreal Engine 5 as a standalone game. Then, press the `` ` `` key to open the console. Type the following command and press **Enter**:

```javascript
r.BufferVisualizationDumpFrames 1
```
This command enables high-resolution screenshots to expose and save the G-Buffers that contributed to rendering of the frame.

Execute the following command within your anaconda environment to start the data generation procedure:

```javascript
    python carla_epe_ue5.py --width 960 --height 540 --output_dir <path-to>/CarlaDataset --num_frames_export 50 --export_step 60
```

> ⚠️ **Warning**: The script will automatically input the high-res command and synchronize it with other data that can be exported via the CARLA Python API. After executing the script, simply focus the CARLA simulator window with your mouse to proceed.

Once the data generation is complete, run the following command to convert the data into a format compatible with the photorealism enhancement:

```javascript
    python epe_preprocess.py --input_path <path-to>/CarlaDataset/ --output_path <path-to>/ --gbuffers ['SceneColor','SceneDepth','WorldNormal','Metallic','Specular','Roughness','BaseColor','SubsurfaceColor'] --gbuffers_grayscale ['SceneDepth','Metallic','Specular','Roughness']
```

The final step is to enhance the data by running inference with the model. We provide an ONNX Runtime script, which can be executed using the following command:

```javascript
    python test.py --model_onnx <path-to>/carla2cityscapes-360000.onnx --dataset_directory <path-to>/CarlaUE5-EPE --out_path <path-to>/EPE
```
> ⚠️ **Warning**: ONNX Runtime library will require a large amount of VRAM. If you have a limited amount of GPU VRAM then the inference should be performed with the PyTorch code as described in the [CARLA2Real readme](https://github.com/stefanos50/CARLA2Real).
