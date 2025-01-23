import carla
import time
import os
import pyautogui
import random
import argparse
#use the following command in the unreal engine console before running the script
#r.BufferVisualizationDumpFrames 1

##width = 960
#height = 540
#output_dir = 'K:/CarlaDataset/'
#num_frames_export = 50
#export_step = 60



parser = argparse.ArgumentParser(description="Capture frames using pyautogui with specified parameters.")
parser.add_argument("--width", type=int, default=960, help="Width of the exported frames.")
parser.add_argument("--height", type=int, default=540, help="Height of the exported frames.")
parser.add_argument("--output_dir", type=str, default="K:/CarlaDataset/", help="Directory to save the frames.")
parser.add_argument("--num_frames_export", type=int, default=50, help="Number of frames to export.")
parser.add_argument("--export_step", type=int, default=60, help="Step size between frame exports.")


args = parser.parse_args()

width = args.width
height = args.height
output_dir = args.output_dir
num_frames_export = args.num_frames_export
export_step = args.export_step

frame_counter = 1
data = {}
paused = False  # Flag to check if simulation is paused
world_tick_counter = 0





def remove_file_if_exists(file_path):
    if os.path.isfile(file_path):  # Check if the file exists and is a file
        os.remove(file_path)      # Remove the file
        print(f"{file_path} has been removed.")
    else:
        print(f"{file_path} does not exist.")

def main():
    global output_dir, frame_counter, width, height, paused, data, world_tick_counter, num_frames_export, export_step
    try:
        # Connect to the CARLA server
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)

        # Get the world and configure synchronous mode
        world = client.get_world()
        settings = world.get_settings()
        settings.synchronous_mode = True  # Enable synchronous mode
        settings.fixed_delta_seconds = 0.05  # Simulation step (20 FPS)
        world.apply_settings(settings)

        # Blueprint library
        blueprint_library = world.get_blueprint_library()

        # Spawn a vehicle
        spawn_points = world.get_map().get_spawn_points()
        vehicle_bp = blueprint_library.filter('vehicle.*')[0]

        if spawn_points:
            random_spawn_point = random.choice(spawn_points)
            
            vehicle = world.spawn_actor(vehicle_bp, random_spawn_point)
            print(f"Vehicle spawned at location: {random_spawn_point.location}")
        else:
            print("No spawn points available in the map.")

        vehicle.set_autopilot(True)

        # Set up semantic segmentation camera
        camera_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
        camera_bp.set_attribute('image_size_x', str(width))  # Resolution
        camera_bp.set_attribute('image_size_y', str(height))
        camera_bp.set_attribute('fov', '90')  # Field of view

        # Attach camera to vehicle
        camera_transform = carla.Transform(carla.Location(x=5.0, z=1.4))  # Adjust position
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir , "Semantic"), exist_ok=True)
        os.makedirs(os.path.join(output_dir , "Frames"), exist_ok=True)

        # Obtain the spectator camera
        spectator = world.get_spectator()

        # Callback to save images only when paused
        def save_segmentation_image(image):
            global data #the sensor data should be added to the data dict
            if not data:
                data["semantic_segmentation"] = image 


        camera.listen(save_segmentation_image)

        # Simulation loop
        frame_exported_counter = 0
        start_time = time.time()
        print("Starting simulation... Press Ctrl+C to stop.")
        time.sleep(2)
        while True:
            vehicle.set_autopilot(True)
            spectator.set_transform(camera.get_transform())
            world.tick()  # Ensures simulation step is processed, even if we're paused.

            #if you add more sensor data e.g. depth update the if with the additional data
            while True:
                if "semantic_segmentation" in data:
                    break

            world_tick_counter += 1
            # Continuously update the spectator camera to follow the vehicle's camera
            #spectator.set_transform(camera.get_transform())

            if world_tick_counter % export_step == 0:
                print("Pausing simulation...")
                print("Running console command...")
                pyautogui.press('`')


                pyautogui.write(
                    f'HighResShot filename={os.path.join(output_dir, "Frames", f"{frame_counter}.png")} {width}x{height}'
                )
                #pyautogui.write(f'HighResShot filename={output_dir}Frames/{frame_counter}.png {width}x{height}')
                pyautogui.press('enter')  # Press Enter to execute
                #pyautogui.press('`')
                print("Console command executed.")

                #save your data to the disk
                #data["semantic_segmentation"].convert(carla.ColorConverter.CityScapesPalette)
                data["semantic_segmentation"].save_to_disk(os.path.join(os.path.join(output_dir , "Semantic"), f"segmentation_{frame_counter-1}.png"))

               
                data = {}
                frame_counter += 1
                time.sleep(1)
                # Resume the simulation
                print("Resuming simulation...")
                remove_file_if_exists(os.path.join(os.path.join(output_dir , "Semantic"), f"segmentation_{0}.png"))

                frame_exported_counter += 1
                if frame_exported_counter == num_frames_export:
                    print("Finished data generation...")
                    exit(1)
        time.sleep(0.1)




    except KeyboardInterrupt:
        print("\nSimulation stopped.")
    finally:

        camera.destroy()
        vehicle.destroy()
        settings.synchronous_mode = False
        world.apply_settings(settings)

        time.sleep(1)
        print("Deleting the additional highreshot...")
        number_prefix = str(frame_counter-1)
        print(number_prefix)

        for filename in os.listdir(os.path.join(output_dir,"Frames")):
            if filename.startswith(number_prefix):  # Check if the filename starts with the specific number
                file_path = os.path.join(os.path.join(output_dir,"Frames"), filename)
                if os.path.isfile(file_path):  # Ensure it's a file, not a directory
                    os.remove(file_path)  # Delete the file
                    print(f"Deleted: {file_path}")

if __name__ == "__main__":
    main()
