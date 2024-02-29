import glob
import os
import sys
import time

import numpy as np
import pygame
import argparse
import torch
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

import carla
import random
import onnxruntime



image_w = 960
image_h = 540
perspective = 0
data_dict = {} #global dictionary that keep tracks of all the required inputs
multi_gt_labels = None

# Render object to keep and pass the PyGame surface
class RenderObject(object):
    def __init__(self, width, height):
        init_image = np.random.randint(0,255,(height,width,3),dtype='uint8')
        self.surface = pygame.surfarray.make_surface(init_image.swapaxes(0,1))

def convert_image_to_array(image):
    img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
    img = img[:, :, :3]
    img = img[:, :, ::-1]
    return img

#add frame to dictionary
def add_frame(image):
    img = convert_image_to_array(image)
    data_dict["color_frame"] = img


def add_sensor(data,sensor_name):
    data_dict[sensor_name] = data

#add the gbuffers in the dictionary. Semantic Segmentation is the same as the CustomStencil buffer.
def add_gbuffer(image, name):
    data_dict[name] = convert_image_to_array(image)
    if name == 'CustomStencil': #custom stencil buffer is the same as semantic segmentation camera
        data_dict["semantic_segmentation"] = data_dict[name]

def make_image():
    img = data_dict['color_frame']
    result = np.expand_dims(img, axis=0)
    result = np.transpose(result, (0, 3, 1, 2)) / 255.0
    result = result.astype(np.float32)
    result = onnxruntime.OrtValue.ortvalue_from_numpy(result, 'cuda', 0)
    return result

#map each available CARLA semantic class in a tensor (29 semantic classes = 29 channels)
def initialize_gt_labels(width=960,height=540,num_channels=29):
    global multi_gt_labels
    specific_classes = [11,1,2,24,25,27,14,15,16,17,18,19,10,9,12,13,6,7,8,21,23,20,3,4,5,26,28,0,22]
    multi_channel_array = np.zeros((num_channels, height, width))
    for channel_index, value in enumerate(specific_classes):
        multi_channel_array[channel_index, :, :] = value
    multi_gt_labels = np.transpose(multi_channel_array, axes=(1, 2, 0))

#create the 12 compressed (grouped) one hot encoded masks from the 29 total semantic classes of carla
def split_gt_label(gt_labels):
    r = (multi_gt_labels == gt_labels[:, :, 0][:, :, np.newaxis].astype(np.float32))


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

def make_gtlabels():
    label_map = data_dict['semantic_segmentation']
    result = np.expand_dims(split_gt_label(label_map), axis=0)
    result = np.transpose(result, (0, 3, 1, 2))
    result = result.astype(np.float32)
    result = onnxruntime.OrtValue.ortvalue_from_numpy(result, 'cuda', 0)
    return result

#the gbuffers should be stacked in that particular sequence otherwise the translation will have artifacts.
def make_gbuffer_matrix():
    global image_h
    global image_w
    try:
        stacked_image = np.concatenate(
            [data_dict['SceneColor'], data_dict['SceneDepth'][:, :, 0][:, :, np.newaxis], data_dict['GBufferA'],
                data_dict['GBufferB'], data_dict['GBufferC'], data_dict['GBufferD'],
                 data_dict['GBufferSSAO'][:, :, 0][:, :, np.newaxis],
                 data_dict['CustomStencil'][:, :, 0][:, :, np.newaxis][:, :, 0][:, :, np.newaxis]], axis=-1)
    except:
        return np.zeros((image_h, image_w, 18))

    return stacked_image

def make_gbuffers():
    result = np.expand_dims(make_gbuffer_matrix(), axis=0)
    result = np.transpose(result, (0, 3, 1, 2))
    result = result.astype(np.float32)
    result = onnxruntime.OrtValue.ortvalue_from_numpy(result, 'cuda', 0)
    return result

def preprocess_data():
    image = make_image()
    gbuffers = make_gbuffers()
    label_maps = make_gtlabels()

    return image,gbuffers,label_maps



def main(onnx_path,town):
    global data_dict

    actor_list = []

    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(10.0)

    world = client.get_world()
    print(client.get_available_maps())
    world = client.load_world(town)
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    world.set_weather(carla.WeatherParameters.ClearNoon)

    blueprint_library = world.get_blueprint_library()

    cars = ['vehicle.chevrolet.impala','vehicle.tesla.model3','vehicle.mercedes.coupe_2020','vehicle.mini.cooper_s_2021','vehicle.dodge.charger_2020','vehicle.lincoln.mkz_2017']
    car_cam_coord = [[1.2,-0.3,1.4],[0.8,-0.3,1.4],[1.0,-0.3,1.4],[1.0,-0.3,1.4],[1.0,-0.3,1.4],[1.0,-0.3,1.4]]
    selected_random_car  = random.randint(0, len(cars) - 1)
    bp = random.choice(blueprint_library.filter(cars[selected_random_car]))


    if bp.has_attribute('color'):
        color = random.choice(bp.get_attribute('color').recommended_values)
        bp.set_attribute('color', color)

    transform = world.get_map().get_spawn_points()[random.randint(0, len(world.get_map().get_spawn_points())-1)]

    vehicle = world.spawn_actor(bp, transform)

    actor_list.append(vehicle)
    print('created %s' % vehicle.type_id)

    vehicle.set_autopilot(True)

    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(image_w))
    camera_bp.set_attribute('image_size_y', str(image_h))

    if perspective == 0:
        camera_transform = carla.Transform(carla.Location(x=car_cam_coord[selected_random_car][0],y=car_cam_coord[selected_random_car][1], z=car_cam_coord[selected_random_car][2]))
    elif perspective == 1:
        camera_transform = carla.Transform(carla.Location(x=1.5, z=1.4))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    actor_list.append(camera)
    print('created %s' % camera.type_id)

    camera_bp2 = blueprint_library.find('sensor.camera.semantic_segmentation')
    camera_bp2.set_attribute('image_size_x', str(image_w))
    camera_bp2.set_attribute('image_size_y', str(image_h))
    if perspective == 0:
        camera_transform2 = carla.Transform(carla.Location(x=car_cam_coord[selected_random_car][0],y=car_cam_coord[selected_random_car][1], z=car_cam_coord[selected_random_car][2]))
    elif perspective == 1:
        camera_transform2 = carla.Transform(carla.Location(x=1.5, z=1.4))
    camera_semseg = world.spawn_actor(camera_bp2, camera_transform2, attach_to=vehicle)
    actor_list.append(camera_semseg)
    print('created %s' % camera_semseg.type_id)

    try:
        #initialize the rgb frame and g buffers listeners
        camera.listen(lambda image: add_frame(image))
        camera.listen_to_gbuffer(carla.GBufferTextureID.SceneColor,
                                 lambda image: add_gbuffer(image, "SceneColor"))
        camera.listen_to_gbuffer(carla.GBufferTextureID.SceneDepth,
                                 lambda image: add_gbuffer(image, "SceneDepth"))
        camera.listen_to_gbuffer(carla.GBufferTextureID.GBufferA,
                                 lambda image: add_gbuffer(image, "GBufferA"))
        camera.listen_to_gbuffer(carla.GBufferTextureID.GBufferB,
                                 lambda image: add_gbuffer(image, "GBufferB"))
        camera.listen_to_gbuffer(carla.GBufferTextureID.GBufferC,
                                 lambda image: add_gbuffer(image, "GBufferC"))
        camera.listen_to_gbuffer(carla.GBufferTextureID.GBufferD,
                                 lambda image: add_gbuffer(image, "GBufferD"))

        camera.listen_to_gbuffer(carla.GBufferTextureID.SSAO,
                                 lambda image: add_gbuffer(image, "GBufferSSAO"))

        camera.listen_to_gbuffer(carla.GBufferTextureID.CustomStencil,
                                 lambda image: add_gbuffer(image, "CustomStencil"))


        renderObject = RenderObject(image_w, image_h)
        pygame.init()
        gameDisplay = pygame.display.set_mode((image_w, image_h), pygame.HWSURFACE | pygame.DOUBLEBUF)
        gameDisplay.fill((0, 0, 0))
        gameDisplay.blit(renderObject.surface, (0, 0))
        pygame.display.flip()
        done = False
        num_skip_frames = 3
        initialize_gt_labels(image_w,image_h,29)

        opts = onnxruntime.SessionOptions()
        opts.enable_profiling = True
        session = onnxruntime.InferenceSession(
            onnx_path,
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

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

            gameDisplay.blit(renderObject.surface, (0, 0))
            pygame.display.flip()

            if num_skip_frames > 0: #skip a number of ticks to speed up performance
                for i in range(num_skip_frames):
                    data_dict = {} #each time the loop should wait for all the data to avoid desync issues
                    world.tick()
                    while True:
                        if len(data_dict) == 10:
                            break
                    data_dict = {}

            vehicle.set_autopilot(True)
            world.tick() #final tick
            world.get_spectator().set_transform(camera.get_transform())

            while True: #wait for all the data
                if len(data_dict) == 10:
                    break

            img,gbuffers,label_map = preprocess_data()

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
            new_img = torch.from_numpy(io_binding.copy_outputs_to_cpu()[0]) #ONNX Runtime inference
            enhanced_frame = (new_img[0, ...].clamp(min=0, max=1).permute(1, 2, 0) * 255.0).detach().cpu().numpy().astype(np.uint8)

            renderObject.surface = pygame.surfarray.make_surface(enhanced_frame[:, :, :3].swapaxes(0, 1)) #add enhanced frame to pygame window
            gameDisplay.fill((0, 0, 0))
            gameDisplay.blit(renderObject.surface, (0, 0))
            pygame.display.flip()

    except KeyboardInterrupt:

        print('destroying actors')
        camera.destroy()
        camera_semseg.destroy()
        vehicle.destroy()
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        print('done.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_onnx', action='store',help='The path where the onnx file (trained model) is stored.')
    parser.add_argument('--carla_town', action='store',help='The name of the CARLA town (Town01,Town10HD, etc.).')

    args = parser.parse_args()

    if (args.model_onnx is None) or not os.path.isfile(args.model_onnx):
        print('--model_onnx argument is not set. Please provide a valid path of the trained model.')
        exit(1)

    if args.carla_town == '' or args.carla_town == None:
        print('--carla_town should be a valid CARLA town name from https://carla.readthedocs.io/en/latest/core_map/.')
        exit(1)

    print("Clearing older ONNX profiles...")
    file_list = os.listdir(os.getcwd())
    for filename in file_list:
        if "onnxruntime_profile__" in filename:
            file_path = os.path.join(os.getcwd(), filename)
            os.remove(file_path)
            print(f"Deleted: {file_path}")

    main(args.model_onnx,args.carla_town)
