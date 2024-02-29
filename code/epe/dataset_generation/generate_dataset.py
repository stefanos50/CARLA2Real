#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys
import numpy as np
import pygame
from PIL import Image
import asyncio
import random
import math
from itertools import repeat
import argparse
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
import time


dataset_path = ""
ticks_per_frame = 0
image_w = 960
image_h = 540
perspective = 0
town = ""
randomize_lights = True

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

def is_vehicle_moving(vehicle):
    velocity = vehicle.get_velocity()
    speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
    return speed > 0.1

gbuffers_list = []
frames_list = []
semantic_list = []
gbuffers_list_id = []
frames_list_id = []
semantic_list_id = []
gbuffers_name = []
def save_image_semantic(image):
    semantic_list_id.append(image.frame)
    semantic_list.append(image)

def save_image(image,name,is_g_buffer,obj):
    if is_g_buffer:
        gbuffers_list.append(image)
        gbuffers_name.append(name)
        gbuffers_list_id.append(image.frame)
    else:
        frames_list_id.append(image.frame)
        frames_list.append(image)

    if is_g_buffer == False:
        img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
        img = img[:, :, :3]
        img = img[:, :, ::-1]
        obj.surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))

def save_g_buffers(buffers_list):
    images_array_list = []
    for buffer_idx in range(len(buffers_list)):
        id = "%06d" % buffers_list[buffer_idx].frame
        buffers_list[buffer_idx].save_to_disk(dataset_path+"CarlaDataset"+"/GBuffers/GBuffer-"+str(id)+"/"+gbuffers_name[buffer_idx]+"-"+str(id)+".png")
        images_array_list.append(convert_image_to_array(buffers_list[buffer_idx]))

    if len(images_array_list)>0:
        gbuffers_dict = dict(zip(gbuffers_name, images_array_list))
        np.savez_compressed(dataset_path+"CarlaDataset"+"/GBuffersCompressed/GBuffer-%07d.npz" % buffers_list[0].frame, **gbuffers_dict)
def exec_full(filepath):
    global_namespace = {
        "__file__": filepath,
        "__name__": "__main__",
    }
    with open(filepath, 'rb') as file:
        exec(compile(file.read(), filepath, 'exec'), global_namespace)


def check(lst):
    repeated = list(repeat(lst[0], len(lst)))
    return repeated == lst

def clear_lists():
    gbuffers_list.clear()
    gbuffers_name.clear()
    semantic_list.clear()
    frames_list.clear()
    frames_list_id.clear()
    gbuffers_list_id.clear()
    semantic_list_id.clear()

def main():

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

    weather_presets = [carla.WeatherParameters.ClearNoon, carla.WeatherParameters.CloudyNoon,
                       carla.WeatherParameters.WetNoon, carla.WeatherParameters.WetCloudyNoon,
                       carla.WeatherParameters.MidRainyNoon, carla.WeatherParameters.HardRainNoon,
                       carla.WeatherParameters.SoftRainNoon, carla.WeatherParameters.ClearSunset,
                       carla.WeatherParameters.CloudySunset, carla.WeatherParameters.WetSunset,
                       carla.WeatherParameters.WetCloudySunset, carla.WeatherParameters.MidRainSunset,
                       carla.WeatherParameters.HardRainSunset, carla.WeatherParameters.SoftRainSunset]


    # The world contains the list blueprints that we can use for adding new
    # actors into the simulation.
    blueprint_library = world.get_blueprint_library()

    # Now let's filter all the blueprints of type 'vehicle' and choose one
    # at random.
    cars = ['vehicle.chevrolet.impala','vehicle.tesla.model3','vehicle.mercedes.coupe_2020','vehicle.mini.cooper_s_2021','vehicle.dodge.charger_2020','vehicle.lincoln.mkz_2017']
    car_cam_coord = [[1.2,-0.3,1.4],[0.8,-0.3,1.4],[1.0,-0.3,1.4],[1.0,-0.3,1.4],[1.0,-0.3,1.4],[1.0,-0.3,1.4]]
    selected_random_car  = random.randint(0, len(cars) - 1)
    #bp = random.choice(blueprint_library.filter(cars[random.randint(0,len(cars)-1)]))
    bp = random.choice(blueprint_library.filter(cars[selected_random_car]))

    #actor_list_remaining = world.get_actors()

    #for actor in actor_list_remaining:
        #if 'vehicle' in actor.type_id or 'walker.pedestrian' in actor.type_id:
            #actor.destroy()

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


        camera.listen(lambda image: save_image(image,"FinalColor",False,renderObject))

        camera.listen_to_gbuffer(carla.GBufferTextureID.SceneColor,
                                  lambda image: save_image(image,"SceneColor",True,renderObject))
        camera.listen_to_gbuffer(carla.GBufferTextureID.SceneDepth,
                                  lambda image: save_image(image,"SceneDepth",True,renderObject))
        camera.listen_to_gbuffer(carla.GBufferTextureID.GBufferA,
                                  lambda image: save_image(image,"GBufferA",True,renderObject))
        camera.listen_to_gbuffer(carla.GBufferTextureID.GBufferB,
                                  lambda image: save_image(image,"GBufferB",True,renderObject))
        camera.listen_to_gbuffer(carla.GBufferTextureID.GBufferC,
                                  lambda image: save_image(image,"GBufferC",True,renderObject))
        camera.listen_to_gbuffer(carla.GBufferTextureID.GBufferD,
                                  lambda image: save_image(image,"GBufferD",True,renderObject))
        camera.listen_to_gbuffer(carla.GBufferTextureID.Velocity,
                                  lambda image: save_image(image,"Velocity",True,renderObject))
        camera.listen_to_gbuffer(carla.GBufferTextureID.SSAO,
                                  lambda image: save_image(image,"GBufferSSAO",True,renderObject))

        camera.listen_to_gbuffer(carla.GBufferTextureID.CustomStencil,
                                  lambda image: save_image(image,"CustomStencil",True,renderObject))

        camera_semseg.listen(lambda image: save_image_semantic(image))

        #camera dimensions

        # Instantiate objects for rendering and vehicle control
        renderObject = RenderObject(image_w, image_h)
        # Initialise the display
        pygame.init()
        gameDisplay = pygame.display.set_mode((image_w, image_h), pygame.HWSURFACE | pygame.DOUBLEBUF)
        # Draw black to the display
        gameDisplay.fill((0, 0, 0))
        gameDisplay.blit(renderObject.surface, (0, 0))
        pygame.display.flip()
        done = False

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

            if randomize_lights:
                actor_list = world.get_actors()
                for actor in actor_list:
                    if 'vehicle' in actor.type_id or 'motorcycle' in actor.type_id:
                        carla_light_states = [carla.VehicleLightState.NONE,carla.VehicleLightState.Position,carla.VehicleLightState.LowBeam,carla.VehicleLightState.HighBeam,carla.VehicleLightState.Fog,carla.VehicleLightState.All]
                        actor.set_light_state(carla_light_states[random.randint(0, len(carla_light_states))-1])

            # Update the display
            gameDisplay.blit(renderObject.surface, (0, 0))
            pygame.display.flip()

            vehicle.set_autopilot(True)
            for tick in range(ticks_per_frame):
                world.tick()
                while True:
                    world.get_spectator().set_transform(camera.get_transform())
                    if len(gbuffers_list) >= 9 and len(semantic_list) >= 1 and len(frames_list) >= 1:
                            clear_lists()
                            break
            world.tick()
            world.get_spectator().set_transform(camera.get_transform())

            while True:
                print(len(gbuffers_list))
                print(len(semantic_list))
                print(len(frames_list))
                if len(gbuffers_list) == 9 and len(semantic_list) == 1 and len(frames_list) == 1 and (gbuffers_list[0].frame == frames_list[0].frame == semantic_list[0].frame) and check([img.frame for img in gbuffers_list]):
                    print(len(gbuffers_list))
                    print(gbuffers_list_id)
                    print(frames_list_id[0])
                    print(semantic_list_id[0])

                    if is_vehicle_moving(vehicle):
                        save_g_buffers(gbuffers_list)
                        semantic_list[0].save_to_disk(dataset_path+"CarlaDataset"+"/SemanticSegmentation/SemanticSegmentation-%07d.png" % semantic_list[0].frame)
                        frames_list[0].save_to_disk(dataset_path+"CarlaDataset"+"/Frames/FinalColor-%07d.png" % frames_list[0].frame)

                    world.set_weather(weather_presets[random.randint(0, len(weather_presets))-1])
                    break
                elif len(gbuffers_list) > 9 and len(semantic_list) > 1 and len(frames_list) > 1:
                    break


            clear_lists()

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

def create_dataset_folders():
    global dataset_path
    if dataset_path[-1] == ":":
        dataset_path += "/"
    if not os.path.exists(dataset_path+"CarlaDataset"):
        os.makedirs(dataset_path+"CarlaDataset")
        print(f"Folder CarlaDataset created successfully.")
    if not os.path.exists(dataset_path+"CarlaDataset"+"/Frames"):
        os.makedirs(dataset_path+"CarlaDataset"+"/Frames")
        print(f"Folder Frames created successfully.")
    if not os.path.exists(dataset_path+"CarlaDataset"+"/GBuffers"):
        os.makedirs(dataset_path+"CarlaDataset"+"/GBuffers")
        print(f"Folder GBuffers created successfully.")
    if not os.path.exists(dataset_path+"CarlaDataset"+"/GBuffersCompressed"):
        os.makedirs(dataset_path+"CarlaDataset"+"/GBuffersCompressed")
        print(f"Folder GBuffersCompressed created successfully.")
    if not os.path.exists(dataset_path+"CarlaDataset"+"/SemanticSegmentation"):
        os.makedirs(dataset_path+"CarlaDataset"+"/SemanticSegmentation")
        print(f"Folder SemanticSegmentation created successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="To generate traffic use the generate_traffic.py file from /PythonAPI/examples directory of your Carla installation. An example of a valid command is  python generate_dataset.py --help --save_path E:\\ --ticks_per_frame 10 --perspective 0 --town Town10HD --randomize_lights False")

    parser.add_argument('--save_path', action='store', help='The path where the dataset will be stored.')
    parser.add_argument('--ticks_per_frame',type=int, action='store', help='The number of ticks that will be skipped until the next time a set of frames will be stored in the disk.')
    parser.add_argument('--perspective',type=int, action='store', help='The perspective of the camera.Set it 0 for Cityscapes compatibility or 1 for general use.')
    parser.add_argument('--town', action='store', help='The carla town that the ego vehicle will be spawned (Town10HD,Town01,... etc).')
    parser.add_argument('--randomize_lights', action='store',help='Enable/Disable random lights of the vehicles for each frame. Valid value is True or False.')

    args = parser.parse_args()

    if (args.save_path is None) or not os.path.isdir(args.save_path):
        print('--save_path argument is not set. Please provide a valid path in the disk where the dataset will be stored.')
        exit(1)

    if (args.ticks_per_frame is None) or not isinstance(args.ticks_per_frame, int):
        print('--ticks_per_frame argument is not set. It should be a positive integer.')
        exit(1)
    ticks_per_frame = int(args.ticks_per_frame)
    if (args.perspective is None) or not isinstance(args.perspective, int):
        print('--perspective argument is not set. It should be a positive integer.')
        exit(1)
    if int(args.perspective) < 0 or int(args.perspective) > 1:
        print('--perspective argument should be 0 or 1.')
        exit(1)
    if (args.town is None):
        print('--town argument is not set. It should be a valid carla town. Visit https://carla.readthedocs.io/en/latest/core_map/ .')
        exit(1)
    if not (args.randomize_lights is None):
        if args.randomize_lights == 'True':
            randomize_lights = True
        elif args.randomize_lights == 'False':
            randomize_lights = False
        else:
            randomize_lights = True
    perspective = int(args.perspective)
    dataset_path = args.save_path
    town = args.town
    create_dataset_folders()

    main()
