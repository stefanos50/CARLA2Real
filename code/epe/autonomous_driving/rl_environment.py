import torch
import numpy as np
import math
import carla
from decimal import Decimal

class AutonomousDrivingEnvironment():
    def __init__(self):
        self.state = None
        self.collision_history = []
        self.ticks_per_episode_num = 20
        self.ticks_per_episode = self.ticks_per_episode_num
        self.spawn_location = 0
        #self.algorithm = algorithm

    def reset(self, vehicle):
        control = carla.VehicleControl(throttle=0, steer=0,brake=0)
        vehicle.apply_control(control)
        self.collision_history = []
        self.state = np.zeros((3, 224, 224), dtype=np.uint8)
        #self.state = np.zeros((1, 67), dtype=np.uint8)
        self.ticks_per_episode = self.ticks_per_episode_num
        return self.state

    def apply_action(self, action, vehicle):
        print("ACTION: "+str(action))
        if action == 0:
            vehicle.apply_control(carla.VehicleControl(throttle=0.6, steer=-1))
        elif action == 1:
            vehicle.apply_control(carla.VehicleControl(throttle=0.6, steer=0))
        elif action == 2:
            vehicle.apply_control(carla.VehicleControl(throttle=0.6, steer=1))
       #vehicle.apply_control(carla.VehicleControl(steer=np.clip(float(action[0]), -1, 1), throttle=0.5,brake=0.0))

    #other_data: dictionary with the data of all the active sensors of the ego vehicle
    #tick: the current tick of the world
    #action: the selected action
    def calculate_reward(self, action , vehicle, world, tick, next_state, other_data):
        reward = 0
        done = False
        out_of_road = False

        vehicle_location = vehicle.get_location()
        waypoint = world.get_map().get_waypoint(vehicle_location,project_to_road=True)

        distance_to_waypoint = vehicle_location.distance(waypoint.transform.location) #ego vehicle distance to closest waypoint

        rotation1 = vehicle.get_transform().rotation
        rotation2 = waypoint.transform.rotation
        yaw_diff = abs((rotation1.yaw - rotation2.yaw + 180) % 360 - 180) #closest to zero the ego vehicle is perfectly alinged with the road direction
        print("yaw_diff: "+str(np.clip((yaw_diff/180),0,1)))

        v = vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))
        print("kmh->"+str(kmh))

        if len(self.collision_history) != 0: #or yaw_diff > 20: #or distance_to_waypoint > 2: #out_of_road
             done = True
             reward = -100
             self.ticks_per_episode=self.ticks_per_episode_num
        elif yaw_diff > 3 and yaw_diff < 60:
             done = False
             reward = -np.clip((yaw_diff/180),0,1)
        elif yaw_diff >=0 and yaw_diff <= 3:
             done = False
             reward = 1 - np.clip((yaw_diff/180),0,1)
        elif yaw_diff >= 60:
            done = True
            reward = -100
        #if self.ticks_per_episode <= 0:
             #reward = 1
             #done=True
        self.ticks_per_episode -= 1

        return reward,done,next_state,{}