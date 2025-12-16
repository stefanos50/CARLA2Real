import logging
import os
import datetime
import random
from pathlib import Path
import sys
from PIL import Image
from collections import Counter
import numpy as np
from scipy.io import savemat
import torch
from torch import autograd
import yaml
import carla
import pygame
import time
import cv2
import epe.dataset as ds
from epe.dataset.utils import mat2tensor
from epe.dataset.batch_types import EPEBatch
from epe.autonomous_driving.ad_model import ADModel
from epe.autonomous_driving.rl_model import RLModel
from epe.autonomous_driving.rl_environment import AutonomousDrivingEnvironment
from epe.autonomous_driving.ad_task import ADTask
from epe.REGEN import regen_generator
from contextlib import contextmanager
import string
from torch import Tensor, ByteTensor
import torch.nn.functional as F
from torch.autograd import Variable
import json
import concurrent.futures
import threading
import math
import torchvision.transforms as trans
import queue
from skimage.measure import label, regionprops
import torch.onnx
import onnxconverter_common
import onnxruntime
import onnx


multi_gt_labels = None
data_dict = {}
names_dict = {}
result_container = {}
enh_height = 540
enh_width = 960
other_actor = None
class_color = {14: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
               15: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
               7: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
               8: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
               16: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
               12: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
               18: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
               19: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
               13: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))}

od_class_names = {
  12: "person",
  14: "vehicle",
  15: "truck",
  16: "bus",
  18: "motorcycle",
  19: "bicycle",
  13: "rider"
}

yolo_class_names = {
  12: 0,
  14: 1,
  15: 2,
  16: 3,
  18: 4,
  19: 5,
  13: 6
}


export_dataset_path = ""
carla_config_path = ""

weather_mapping = {
    "ClearNoon": carla.WeatherParameters.ClearNoon,
    "ClearSunset": carla.WeatherParameters.ClearSunset,
    "CloudyNoon": carla.WeatherParameters.CloudyNoon,
    "CloudySunset": carla.WeatherParameters.CloudySunset,
    "WetNoon": carla.WeatherParameters.WetNoon,
    "WetSunset": carla.WeatherParameters.WetSunset,
    "SoftRainNoon": carla.WeatherParameters.SoftRainNoon,
    "SoftRainSunset": carla.WeatherParameters.SoftRainSunset,
    "HardRainNoon": carla.WeatherParameters.HardRainNoon,
    "HardRainSunset": carla.WeatherParameters.HardRainSunset,
    "WetCloudyNoon": carla.WeatherParameters.WetCloudyNoon,
    "WetCloudySunset": carla.WeatherParameters.WetCloudySunset

}


def seed_worker(id):
    random.seed(torch.initial_seed() % np.iinfo(np.int32).max)
    np.random.seed(torch.initial_seed() % np.iinfo(np.int32).max)
    pass


def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)
        pass
    pass


_logstr2level = {
    'critical': logging.CRITICAL,
    'error': logging.ERROR,
    'warn': logging.WARNING,
    'warning': logging.WARNING,
    'info': logging.INFO,
    'debug': logging.DEBUG
}


def parse_loglevel(loglevel_arg):
    level = _logstr2level.get(loglevel_arg.lower())
    if level is None:
        raise ValueError(
            f"log level given: {loglevel_arg}"
            f" -- must be one of: {' | '.join(_logstr2level.keys())}")

    return level


def init_logging(args):
    now = datetime.datetime.now()
    log_path = args.log_dir / f'{args.config.stem}_{datetime.date.today().isoformat()}_{now.hour}-{now.minute}-{now.second}.log'
    level = parse_loglevel(args.log)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=level, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(log_path, mode='a'), logging.StreamHandler()])
    #logging.getLogger().setLevel(level)

class TinyModel(torch.nn.Module):

    def __init__(self):
        super(TinyModel, self).__init__()

        self.linear1 = torch.nn.Linear(960, 200)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, 10)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x

class NetworkState:
    """ Capture (training) state of a network.

	"""

    def __init__(self, network, cfg, name='network_state'):

        self._log = logging.getLogger(f'epe.experiment.{name}')
        self.network = network
        self.iterations = 0

        self._parse_config(cfg)
        pass

    def _parse_config(self, cfg):

        self._init_optimizer(dict(cfg.get('optimizer', {})))
        self._init_scheduler(dict(cfg.get('scheduler', {})))

        self.learning_rate = self.scheduler.get_last_lr()
        pass

    def _init_optimizer(self, cfg):

        self.learning_rate = float(cfg.get('learning_rate', 0.001))
        self.clip_gradient_norm = float(cfg.get('clip_gradient_norm', -1))
        self.clip_weights = float(cfg.get('clip_weights', -1))

        momentum = float(cfg.get('momentum', 0.0))
        weight_decay = float(cfg.get('weight_decay', 0.0001))
        adam_ams = bool(cfg.get('adam_ams', False))
        adam_beta = float(cfg.get('adam_beta', 0.9))
        adam_beta2 = float(cfg.get('adam_beta2', 0.999))
        optimizer = str(cfg.get('type', 'adam'))

        self._log.debug(f'  learning rate : {self.learning_rate}')
        self._log.debug(f'  clip grad norm: {self.clip_gradient_norm}')
        self._log.debug(f'  clip_weights  : {self.clip_weights}')

        self._log.debug(f'  optimizer     : {optimizer}')

        if optimizer == 'adam':
            self._log.debug(f'    ams         : {adam_ams}')
            self._log.debug(f'    beta        : {adam_beta}')
            self._log.debug(f'    beta2       : {adam_beta2}')
            self.optimizer = torch.optim.Adam(params=self.network.parameters(), lr=self.learning_rate,
                                              betas=(adam_beta, adam_beta2), weight_decay=weight_decay,
                                              amsgrad=adam_ams)

        elif optimizer == 'adamw':
            self._log.debug(f'    ams         : {adam_ams}')
            self._log.debug(f'    beta        : {adam_beta}')
            self._log.debug(f'    beta2       : {adam_beta2}')
            self.optimizer = torch.optim.AdamW(params=self.network.parameters(), lr=self.learning_rate,
                                               betas=(adam_beta, adam_beta2), weight_decay=weight_decay,
                                               amsgrad=adam_ams)

        elif optimizer == 'sgd':
            self._log.debug(f'    momentum      : {momentum}')
            self._log.debug(f'    weight_decay  : {weight_decay}')
            self.optimizer = torch.optim.SGD(params=self.network.parameters(), lr=self.learning_rate, momentum=momentum,
                                             weight_decay=weight_decay)

        else:
            raise NotImplementedError

        pass

    def _init_scheduler(self, cfg):

        scheduler = str(cfg.get('scheduler', 'step'))
        step = int(cfg.get('step', 1000000))
        step_gamma = float(cfg.get('step_gamma', 1))

        if scheduler == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step, gamma=step_gamma)

        elif scheduler == 'exp':
            # will produce  a learning rate of step_gamma at step
            gamma = step_gamma ** (1.0 / step)
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)

        elif scheduler == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=step,
                                                                        eta_min=self.learning_rate * step_gamma)
        pass

    def load_from_dict(self, d):
        """ Initialize the network state from a dictionary."""
        print("Loading a checkpoint...")
        self.network.load_state_dict(d['network'])
        self.optimizer.load_state_dict(d['optimizer'])
        self.scheduler.load_state_dict(d['scheduler'])
        self.iterations = d.get('iterations', 0)
        pass

    def save_to_dict(self):
        """ Save the network state to a disctionary."""

        return { \
                   'network': self.network.state_dict(),
                   'optimizer': self.optimizer.state_dict(),
                   'scheduler': self.scheduler.state_dict()}, {
                   'iterations': self.iterations}

    def prepare(self):
        self.optimizer.zero_grad(set_to_none=True)
        pass

    def update(self):
        if self.clip_gradient_norm > 0:
            # loss_infos['ggn'] =
            # n = torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.clip_gradient_norm, norm_type=2)
            n = torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.clip_gradient_norm, norm_type='inf')
            if self._log.isEnabledFor(logging.DEBUG):
                self._log.debug(f'gradient norm: {n}')
            pass

        self.optimizer.step()
        self.scheduler.step()
        lr = self.scheduler.get_last_lr()

        if lr != self.learning_rate:
            self._log.info(f'Learning rate set to {lr}.')
            self.learning_rate = lr
            pass

        if self.clip_weights > 0:
            for p in self.network.parameters():
                p.data.clamp_(-self.clip_weights, self.clip_weights)
                pass
            pass

        self.iterations += 1
        pass

    pass


class LogSync:
    def __init__(self, logger, log_interval):
        self.scalars = {}
        self._log = logger
        self._log_interval = log_interval
        self._scalar_queue = {}
        self._delay = 3

    def update(self, i, scalars):
        for k, v in scalars.items():
            if k not in self._scalar_queue:
                self._scalar_queue[k] = {}
                pass

            self._scalar_queue[k][i] = v.to('cpu', non_blocking=True)
            pass
        pass

    # def _update_gpu(self, scalars):
    # 	for k,v in scalars.items():
    # 		if k not in self.scalars:
    # 			self.scalars[k] = [torch.tensor([0.0], device=v.device), 0]
    # 			pass

    # 		self.scalars[k] = [self.scalars[k][0]+v, self.scalars[k][1]+1]
    # 		pass
    # 	pass

    def print(self, i):
        """ Print to screen. """

        if i % (20 * self._log_interval) == 0:
            line = [f'{i:d} ']
            for t in self._scalar_queue.keys():
                line.append('%-4.4s ' % t)
                pass
            self._log.info('')
            self._log.info(''.join(line))

        if i % self._log_interval == 0:
            line = [f'{i:d} ']

            # Loss infos
            new_queue = {}
            for k, v in self._scalar_queue.items():
                valid = {j: float(vj) for j, vj in v.items() if i - j >= self._delay}
                if valid:
                    vv = valid.values()
                    line.append(f'{sum(vv) / len(vv):.2f} ')
                    for vk in valid.keys():
                        del v[vk]
                        pass
                    pass
                else:
                    line.append('---- ')
                    pass
                pass
            self._log.info(''.join(line))
            pass


class RenderObject(object):
    def __init__(self, width, height):
        init_image = np.random.randint(0, 255, (height, width, 3), dtype='uint8')
        self.surface = pygame.surfarray.make_surface(init_image.swapaxes(0, 1))


class BaseExperiment:
    """ Provide scaffold for common operations in an experiment.

	The class provides a scaffold for common operations required in running an experiment.
	It provides a training, validation, and testing loop, methods for loading and storing weights,
	and storing debugging info. It does not specify network architectures, optimizers, or datasets
	as they may vary a lot depending on the specific experiment or task.

	"""

    actions = ['train', 'test', 'infer']
    networks = {}

    def __init__(self, args):
        """Common set up code for all actions."""
        self.action = args.action
        self._log = logging.getLogger('main')
        self.no_safe_exit = args.no_safe_exit
        self.collate_fn_train = None
        self.collate_fn_val = None

        self.device = torch.device(f'cuda:{args.gpu}' if args.gpu >= 0 else 'cpu')

        self._load_config(args.config)
        self._parse_config()

        self._log_sync = LogSync(self._log, self._log_interval)

        self._save_id = 0
        self._init_directories()
        self._init_dataset()
        self._init_network()
        self._init_network_state()

        if self.seed is not None:
            torch.manual_seed(self.seed)
            pass
        pass

    def _load_config(self, config_path):
        with open(config_path) as file:
            self.cfg = yaml.safe_load(file)
            pass
        pass

    def _parse_config(self):

        common_cfg = dict(self.cfg.get('common', {}))
        self.unpin = bool(common_cfg.get('unpin', False))
        self.seed = common_cfg.get('seed', None)
        self.batch_size = int(common_cfg.get('batch_size', 1))
        self.num_loaders = int(common_cfg.get('num_loaders', 10))
        self._log_interval = int(common_cfg.get('log_interval', 1))

        prof_cfg = dict(self.cfg.get('profile', {}))
        self._profile = bool(prof_cfg.get('enable', False))
        self._profile_gpu = bool(prof_cfg.get('gpu', False))
        self._profile_memory = bool(prof_cfg.get('memory', False))
        self._profile_stack = bool(prof_cfg.get('stack', False))
        self._profile_path = Path(prof_cfg.get('path', '.'))

        self._log.debug(f'  unpin        : {self.unpin}')
        self._log.debug(f'  seed         : {self.seed}')
        self._log.debug(f'  batch_size   : {self.batch_size}')
        self._log.debug(f'  num_loaders  : {self.num_loaders}')
        self._log.debug(f'  log_interval : {self._log_interval}')
        self._log.debug(f'  profile      : {self._profile}')

        self.shuffle_test = bool(self.cfg.get('shuffle_test', False))
        self.shuffle_train = bool(self.cfg.get('shuffle_train', True))

        self.weight_dir = Path(self.cfg.get('weight_dir', './savegames/'))
        self.weight_init = self.cfg.get('name_load', None)
        self.dbg_dir = Path(self.cfg.get('out_dir', './out/'))
        self.result_ext = '.jpg'

        self._log.debug(f'  weight_dir   : {self.weight_dir}')
        self._log.debug(
            f'  name_load    : {self.weight_init}{" (will not load anything)" if self.weight_init is None else ""}')
        self._log.debug(f'  out_dir      : {self.dbg_dir}')

        train_cfg = dict(self.cfg.get('train', {}))
        self.max_epochs = int(train_cfg.get('max_epochs', -1))
        self.max_iterations = int(train_cfg.get('max_iterations', -1))
        self.save_epochs = int(train_cfg.get('save_epochs', -1))
        self.save_iterations = int(train_cfg.get('save_iterations', 100000))
        self.weight_save = str(train_cfg.get('name_save', 'model'))
        self.no_validation = bool(train_cfg.get('no_validation', False))
        self.val_interval = int(train_cfg.get('val_interval', 20000))

        self._log.debug(f'  training config:')
        self._log.debug(f'    max_epochs      : {self.max_epochs}')
        self._log.debug(f'    max_iterations  : {self.max_iterations}')
        self._log.debug(f'    name_save       : {self.weight_save}')
        self._log.debug(f'    save_epochs     : {self.save_epochs}')
        self._log.debug(f'    save_iterations : {self.save_iterations}')
        self._log.debug(f'    validation      : {"off" if self.no_validation else f"every {self.val_interval}"}')
        pass

    @property
    def i(self):
        raise NotImplementedError

    # return self._iterations

    def _init_directories(self):
        self.dbg_dir.mkdir(parents=True, exist_ok=True)
        (self.dbg_dir / self.weight_save).mkdir(parents=True, exist_ok=True)
        self.weight_dir.mkdir(parents=True, exist_ok=True)
        pass

    def _init_network(self):
        pass

    def _init_dataset(self):
        pass

    def _init_network_state(self):
        """ Initialize optimizer and scheduler for the network. """
        pass

    def _train_network(self, batch):
        """ Run forward and backward pass of a network. """
        raise NotImplementedError

    def _should_stop(self, e, i):
        """ Check whether training stop criterion is reached. """

        if self.max_epochs > 0 and e >= self.max_epochs:
            return True

        if self.max_iterations > 0 and i >= self.max_iterations:
            return True

        return False

    def _should_save_epoch(self, e):
        return self.save_epochs > 0 and e % self.save_epochs == 0

    def _should_save_iteration(self, i):
        return self.save_iterations > 0 and i % self.save_iterations == 0

    def _dump(self, img_vars, other_vars={}, force=False):
        if force or ((self.i // 1000) % 5 == 0 and (self.i % 100 == 0)) or (self.i < 20000 and self.i % 100 == 0):
            d1 = {('i_%s' % k): v for k, v in img_vars.items() if v is not None}
            d2 = {('o_%s' % k): v for k, v in other_vars.items()}
            self.save_dbg({**d1, **d2}, '%d' % self.i)

    def evaluate_test(self, batch, batch_id):
        raise NotImplementedError
        pass

    # Render object to keep and pass the PyGame surface

    # -------------------------------------------------------------------------------------------------------------------------------------------------------

    def convert_image_to_array(self, image):
        img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
        img = img[:, :, :3]
        img = img[:, :, ::-1]
        return img

    def make_gbuffer_matrix(self):
        global enh_height
        global enh_width
        try:
            stacked_image = np.concatenate(
                [data_dict['SceneColor'], data_dict['SceneDepth'][:, :, 0][:, :, np.newaxis], data_dict['GBufferA'],
                 data_dict['GBufferB'], data_dict['GBufferC'], data_dict['GBufferD'],
                 data_dict['GBufferSSAO'][:, :, 0][:, :, np.newaxis],
                 data_dict['CustomStencil'][:, :, 0][:, :, np.newaxis][:, :, 0][:, :, np.newaxis]], axis=-1)
        except:
            return np.zeros((enh_height, enh_width, 18))
        return stacked_image

    def split_gt_label(self, gt_labels):
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

    def add_frame(self, image):
        #if "color_frame" not in data_dict:
        img = self.convert_image_to_array(image)
        data_dict["color_frame"] = img
        names_dict["color_frame"] = image.frame


    def add_sensor(self,data,sensor_name):

        if sensor_name == "instance_segmentation":
            array = np.frombuffer(data.raw_data, dtype=np.uint8)
            array = array.reshape((data.height, data.width, 4))[:, :, :3]
            array = array[:, :, ::-1]
            data_dict[sensor_name] = array
            names_dict[sensor_name] = data.frame
            return
        
        data_dict[sensor_name] = data
        names_dict[sensor_name] = data.frame

    def add_semantic(self, image):
        #if "semantic_segmentation" not in data_dict:
        label_map = self.convert_image_to_array(image)
        data_dict["semantic_segmentation"] = label_map
        names_dict["semantic_segmentation"] = image.frame

    def add_gbuffer(self, image, name):
        #if name == 'CustomStencil':
            #self.add_semantic(image)
        #if name not in data_dict:
        data_dict[name] = self.convert_image_to_array(image)
        names_dict[name] = image.frame

        if name == 'CustomStencil': #custom stencil buffer is the same as semantic segmentation camera
            data_dict["semantic_segmentation"] = data_dict[name]
            names_dict["semantic_segmentation"] = names_dict[name]

    def make_image(self, method):
        # img = self.convert_image_to_array(color_frame_list[0])
        img = data_dict['color_frame']

        # renderObject.surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))
        if method == "EPE":
            result = mat2tensor(img.astype(np.float32) / 255.0)
        else:
            img = np.ascontiguousarray(img)
            result = torch.from_numpy(img).permute(2, 0, 1).float() / 127.5 - 1
            result = result.unsqueeze(0)
        return result

    # Function 2 to be executed on a separate thread
    def make_gtlabels(self):
        # label_map = self.convert_image_to_array(semantic_sementation_list[0])
        label_map = data_dict['semantic_segmentation']

        result = mat2tensor(self.split_gt_label(label_map))
        return result

    # Function 3 to be executed on a separate thread
    def make_gbuffers(self):
        result = mat2tensor(self.make_gbuffer_matrix().astype(np.float32))
        return result

    def render_thread(self, queue, render_screen, method):
        while True:
            argument = queue.get()
            if argument is None:
                break
            img =  self.process_final_image(argument, method)
            render_screen.surface = pygame.surfarray.make_surface(img[:, :, :3].swapaxes(0, 1))
    
    def process_final_image(self, output, method):
        if method == "EPE":
            return (output[0, ...].clamp(min=0, max=1).permute(1, 2, 0) * 255.0).detach().cpu().numpy().astype(np.uint8)
        else:
            out_img = output[0].cpu().permute(1, 2, 0).numpy()
            out_img = ((out_img + 1) * 127.5).clip(0, 255).astype(np.uint8)
            #out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
            return out_img


    def preprocess_worker(self, name, comp, inputs, method):
        global result_container
        nameid = 0
        if name == "frame":
            result = self.make_image(method)
            nameid = 0
        elif name == "gt_labels":
            result = self.make_gtlabels()
            nameid = 2
        else:
            result = self.make_gbuffers()
            nameid = 1
        if comp == 'onnxruntime' or comp == 'tensorrt':
            result_container[name] = np.expand_dims(result.numpy(),axis=0)
            if comp == 'tensorrt':
                inputs[nameid].host = result_container[name]
            elif comp == 'onnxruntime':
                result_container[name] = onnxruntime.OrtValue.ortvalue_from_numpy(result_container[name], 'cuda', 0)
        else:
            result_container[name] = result.pin_memory().to(self.device)

    def save_vehicle_status(self, vehicle, frame_id):
        velocity = vehicle.get_velocity()
        speed = 3.6 * math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        control = vehicle.get_control()
        steering = control.steer
        throttle = control.throttle
        brake = control.brake
        location = vehicle.get_location()
        rotation = vehicle.get_transform().rotation

        vehicle_status = {"velocity": [velocity.x, velocity.y, velocity.z], "speed": speed, "steering": steering,
                          "throttle": throttle, "brake": brake, "location": [location.x, location.y, location.z],
                          "rotation": [rotation.pitch, rotation.yaw, rotation.roll]}

        with open(export_dataset_path + "ADDataset/VehicleStatus/" + str(frame_id) + ".json", "w") as outfile:
            json.dump(vehicle_status, outfile)

    def save_world_status(self, town_name, weather_name, vehicle_name, perspective, synchronization, frame_id,
                          real_dataset="cityscapes"):
        world_status = {"town": town_name, "weather": weather_name, "vehicle": vehicle_name, "perspective": perspective,
                        "synchronization_mode": synchronization, "real_dataset": real_dataset}

        with open(export_dataset_path + "ADDataset/WorldStatus/" + str(frame_id) + ".json", "w") as outfile:
            json.dump(world_status, outfile)

    def save_frames(self, frame_id, enhanced_frame, carla_config, method):
        enhanced_frame = self.process_final_image(enhanced_frame, method)

        im = Image.fromarray(enhanced_frame[:, :, :3])
        im.save(export_dataset_path + "ADDataset/EnhancedFrames/" + str(frame_id) + "." + str(carla_config['dataset_settings']['images_format']))

        im = Image.fromarray(data_dict["color_frame"])
        im.save(export_dataset_path + "ADDataset/RGBFrames/" + str(frame_id) + "." + str(carla_config['dataset_settings']['images_format']))

        if carla_config['dataset_settings']['export_semantic_gt']:
            im = Image.fromarray(data_dict["semantic_segmentation"])
            im.save(export_dataset_path + "ADDataset/SemanticSegmentation/" + str(frame_id) + "." + str(carla_config['dataset_settings']['images_format']))
        if carla_config['dataset_settings']['export_instance_gt']:
            im = Image.fromarray(data_dict["instance_segmentation"])
            im.save(export_dataset_path + "ADDataset/InstanceSegmentation/" + str(frame_id) + "." + str(carla_config['dataset_settings']['images_format']))
        if carla_config['dataset_settings']['export_depth']:
            im = Image.fromarray(data_dict["SceneDepth"])
            im.save(export_dataset_path + "ADDataset/Depth/" + str(frame_id) + "." + str(carla_config['dataset_settings']['images_format']))

    def random_id_generator(self, n=10, chars=string.ascii_uppercase + string.ascii_lowercase + string.digits):
        return ''.join(random.choice(chars) for _ in range(n))

    def save_rl_stats(self, actor_losses, critic_losses, steps, rewards, current_step,distances,episode,epsilon):
        rl_stats = {"losses": actor_losses, "critic_losses": critic_losses, "steps": steps, "rewards": rewards,"distances":distances,"episodes":episode,"epsilons":epsilon}

        with open("out/rl_stats/rl_stats_"+str(current_step)+".json", "w") as outfile:
            json.dump(rl_stats, outfile)

    def on_collision(self, event, environment):
        collision_actor = event.other_actor
        print(collision_actor)
        environment.collision_history.append(collision_actor.type_id)

    def is_vehicle_moving(self,vehicle,speed_threshold):
        velocity = vehicle.get_velocity()
        speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        return speed > speed_threshold

    def create_dataset_folders(self,carla_config):
        global export_dataset_path
        if not os.path.isdir(export_dataset_path):
            print('\033[91m'+"Export dataset path is not a valid path in the disk.")
            exit(1)
        if export_dataset_path[-1] == ":":
            export_dataset_path += "/"

        if not os.path.exists(export_dataset_path + "ADDataset" + "/SemanticSegmentation") and carla_config['dataset_settings']['export_semantic_gt']:
            os.makedirs(export_dataset_path + "ADDataset" + "/SemanticSegmentation")
            print(f"Folder SemanticSegmentation created successfully.")
        if not os.path.exists(export_dataset_path + "ADDataset" + "/InstanceSegmentation") and carla_config['dataset_settings']['export_instance_gt']:
            os.makedirs(export_dataset_path + "ADDataset" + "/InstanceSegmentation")
            print(f"Folder InstanceSegmentation created successfully.")
        if not os.path.exists(export_dataset_path + "ADDataset" + "/WorldStatus") and carla_config['dataset_settings']['export_status_json']:
            os.makedirs(export_dataset_path + "ADDataset" + "/WorldStatus")
            print(f"Folder WorldStatus created successfully.")
        if not os.path.exists(export_dataset_path + "ADDataset" + "/VehicleStatus") and carla_config['dataset_settings']['export_status_json']:
            os.makedirs(export_dataset_path + "ADDataset" + "/VehicleStatus")
            print(f"Folder VehicleStatus created successfully.")
        if not os.path.exists(export_dataset_path + "ADDataset" + "/Depth") and carla_config['dataset_settings']['export_depth']:
            os.makedirs(export_dataset_path + "ADDataset" + "/Depth")
            print(f"Folder Depth created successfully.")
        if not os.path.exists(export_dataset_path + "ADDataset" + "/EnhancedFrames"):
            os.makedirs(export_dataset_path + "ADDataset" + "/EnhancedFrames")
            print(f"Folder EnhancedFrames created successfully.")
        if not os.path.exists(export_dataset_path + "ADDataset" + "/RGBFrames"):
            os.makedirs(export_dataset_path + "ADDataset" + "/RGBFrames")
            print(f"Folder RGBFrames created successfully.")
        if not os.path.exists(export_dataset_path + "ADDataset" + "/ObjectDetection") and carla_config['dataset_settings']['export_object_annotations']:
            os.makedirs(export_dataset_path + "ADDataset" + "/ObjectDetection")
            print(f"Folder ObjectDetection created successfully.")

    def initialize_gt_labels(self,width=960,height=540,num_channels=29):
        global multi_gt_labels
        specific_classes = [11,1,2,24,25,27,14,15,16,17,18,19,10,9,12,13,6,7,8,21,23,20,3,4,5,26,28,0,22]
        multi_channel_array = np.zeros((num_channels, height, width))
        for channel_index, value in enumerate(specific_classes):
            multi_channel_array[channel_index, :, :] = value
        multi_gt_labels = np.transpose(multi_channel_array, axes=(1, 2, 0))

    def stabilize_vehicle(self,world,spectator_camera_mode,camera,num_ticks,data_length):
        global data_dict
        global names_dict

        for i in range(num_ticks):
            data_dict = {}
            names_dict = {}
            world.tick()
            if spectator_camera_mode == 'follow':
                world.get_spectator().set_transform(camera.get_transform())
            while True:
                if len(data_dict) == data_length:
                    break
            data_dict = {}
            names_dict = {}

    def get_transform_from_field(self,world,scenario_config,field_name):
        if scenario_config[field_name]['init_spawn_point'] == 'random':
            transform = world.get_map().get_spawn_points()[random.randint(0, len(world.get_map().get_spawn_points()) - 1)]
        elif isinstance(scenario_config[field_name]['init_spawn_point'], int):
            transform = world.get_map().get_spawn_points()[scenario_config[field_name]['init_spawn_point']]
        elif isinstance(scenario_config[field_name]['init_spawn_point'], list):
            coords = scenario_config[field_name]['init_spawn_point']
            transform = carla.Transform(carla.Location(x=coords[0][0], y=coords[0][1], z=coords[0][2]), carla.Rotation(coords[1][0],coords[1][1],coords[1][2]))
        return transform

    def initialize_movement(self,scenario_config,controls):
        global other_actor
        if scenario_config['other_actor']['static'] == False:
            if isinstance(other_actor, carla.Vehicle):
                init_controls = scenario_config['other_actor'][controls]
                other_actor.apply_control(carla.VehicleControl(throttle=init_controls[1], steer=init_controls[0], brake=init_controls[2]))
            elif isinstance(other_actor, carla.Walker):
                init_controls = scenario_config['other_actor'][controls]
                walker_control = carla.WalkerControl()
                walker_control.speed = init_controls[0]  # Set the desired speed in m/s
                walker_control.direction = carla.Vector3D(x=init_controls[1][0], y=init_controls[1][1], z=init_controls[1][2])  # Set the direction
                other_actor.apply_control(walker_control)
            else:
                print('\033[91m'+"Not compatible actor. Choose a vehicle or pederestrian actor instance.")
                exit(1)




    def save_object_detection_annotations(self,camera,world,vehicle,frame_id,carla_config):
            global class_colors, data_dict, od_class_names, export_dataset_path, yolo_class_names
            world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

            image_w = int(carla_config['ego_vehicle_settings']['camera_width'])
            image_h = int(carla_config['ego_vehicle_settings']['camera_height'])
            image_fov = int(carla_config['ego_vehicle_settings']['camera_fov'])
            listed_classes = list(carla_config['dataset_settings']['object_annotations_classes'])
            
            current_frame = np.ascontiguousarray(data_dict['color_frame'], dtype=np.uint8)
            current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)


            instance_img = data_dict['instance_segmentation']
            instance_img = np.array(instance_img, dtype=np.uint16)
            
            #instance_img = (instance_img[:, :, 1] << 8) + instance_img[:, :, 2]
            instance_img = instance_img[:, :, 0] + 256 * instance_img[:, :, 1] + 256**2 * instance_img[:, :, 2]

            instance_ids = np.unique(instance_img)
            instance_ids = instance_ids[instance_ids != 0]

            semantic_img = data_dict['semantic_segmentation'][:,:,2]

            boxes = []
            yolo_lines = []  # <-- ADDED

            for inst_id in instance_ids:
                mask = (instance_img == inst_id)
                ys, xs = np.where(mask)
                if ys.size == 0 or xs.size == 0:
                    continue

                x_min, x_max = xs.min(), xs.max()
                y_min, y_max = ys.min(), ys.max()
                width, height = x_max - x_min, y_max - y_min


                # Get class ID from semantic segmentation
                sem_values = semantic_img[mask]
                class_id = Counter(sem_values).most_common(1)[0][0]  # <-- UNCHANGED

                # Filter by allowed semantic classes
                if class_id not in listed_classes:
                    continue

                boxes.append((x_min, y_min, x_max, y_max, class_id))

                cv2.rectangle(current_frame, (int(x_max), int(y_max)), (int(x_min), int(y_min)), class_color[class_id], 2)
                cv2.putText(current_frame, od_class_names[class_id], (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, class_color[class_id], 2)

                # --- ADDED: Save to YOLO format ---
                x_center = ((x_min + x_max) / 2.0) / image_w
                y_center = ((y_min + y_max) / 2.0) / image_h
                w_norm = (x_max - x_min) / image_w
                h_norm = (y_max - y_min) / image_h
                yolo_lines.append(f"{yolo_class_names[class_id]} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
                # ---------------------------------

            # --- ADDED: Save YOLO txt file ---
            import os
            save_dir = os.path.join(export_dataset_path, "ADDataset", "ObjectDetection")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{frame_id}.txt")
            with open(save_path, "w") as f:
                f.write("\n".join(yolo_lines))
            # ---------------------------------

            if carla_config['dataset_settings']['object_visualize_annotations'] == True:
                cv2.imshow('Object Detection Annotations', current_frame)


    def initialize_scenario(self,world,scenario_config,vehicle):
        global other_actor

        vehicle.set_target_velocity(carla.Vector3D(x=0, y=0, z=0))
        ego_transform = self.get_transform_from_field(world,scenario_config,'ego_vehicle_settings')
        vehicle.set_transform(ego_transform)

        if other_actor == None and scenario_config['other_actor']['actor_id'] is not None:
            blueprint_library = world.get_blueprint_library()
            available_blueprints = blueprint_library.filter(scenario_config['other_actor']['actor_id'])
            vehicle_blueprint = available_blueprints[0]
            transform = self.get_transform_from_field(world,scenario_config,'other_actor')
            other_actor = world.spawn_actor(vehicle_blueprint, transform)
            self.initialize_movement(scenario_config,'init_controls')
        elif other_actor is not None:
            other_actor.set_target_velocity(carla.Vector3D(x=0, y=0, z=0))
            self.initialize_movement(scenario_config, 'init_controls')
            transform = self.get_transform_from_field(world,scenario_config,'other_actor')
            other_actor.set_transform(transform)

        if scenario_config['general']['traffic_lights'] == 'green':
            list_actor = world.get_actors()
            for actor_ in list_actor:
                if isinstance(actor_, carla.TrafficLight):
                    actor_.set_state(carla.TrafficLightState.Green)
                    actor_.set_green_time(scenario_config['general']['traffic_lights_time'])
        elif scenario_config['general']['traffic_lights'] == 'red':
            for actor_ in list_actor:
                if isinstance(actor_, carla.TrafficLight):
                    actor_.set_state(carla.TrafficLightState.Red)
                    actor_.set_red_time(scenario_config['general']['traffic_lights_time'])

        return ego_transform

    def compare_distance(self,comparison_critiria,dist,threshold):
        result = False
        comparison_critiria = str(comparison_critiria)
        if comparison_critiria == 'greater':
            result = dist > threshold
        elif comparison_critiria == 'less':
            result = dist < threshold
        elif comparison_critiria == 'equals':
            result = dist == threshold
        elif comparison_critiria == 'not equals':
            result = dist != threshold
        elif comparison_critiria == 'greater_equals':
            result = dist >= threshold
        elif comparison_critiria == 'less_equals':
            result = dist <= threshold
        return result

    def manual_controls_apply(self,vehicle,manual_controls):
        keys = pygame.key.get_pressed()

        throttle = 0.0
        steer = 0.0
        brake = 0.0

        if keys[pygame.K_UP] or keys[pygame.K_w]:
            throttle = manual_controls[0]
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            brake = manual_controls[2]
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            steer = -1 * manual_controls[1]
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            steer = manual_controls[1]

        vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))

    def trigger_scenario(self,world,scenario_config,vehicle):
        global other_actor

        if other_actor is not None and scenario_config['other_actor']['static'] is not None and scenario_config['other_actor']['static'] == False:
            ego_transform = vehicle.get_transform()
            other_transform = other_actor.get_transform()

            ego_location = ego_transform.location
            other_location = other_transform.location

            distance = math.sqrt((ego_location.x - other_location.x) ** 2 + (ego_location.y - other_location.y) ** 2 + (ego_location.z - other_location.z) ** 2)
            print("Distance to other actor: "+ str(round(distance, 2))+ "m")
            #print(self.compare_distance(scenario_config['other_actor']['threshold_critiria'],distance,scenario_config['other_actor']['distance_threshold']))
            if self.compare_distance(scenario_config['other_actor']['threshold_critiria'],distance,scenario_config['other_actor']['distance_threshold']):
                self.initialize_movement(scenario_config,'out_controls')

    def generate_onxx(self,path,input_width,input_height):
        img = torch.randn(1, 3, input_height, input_width)
        gb = torch.randn(1, 18, input_height, input_width)
        gt = torch.randn(1, 12, input_height, input_width)
        torch.onnx.export(self.network.generator.to("cpu"),
                              [img,gb,gt],
                              path,
                              export_params=True,
                              opset_version=11,
                              do_constant_folding=True,
                              input_names=['input'],
                              output_names=['output'])


    def get_engine(self,onnx_file_path, engine_file_path="",precision="fp16",calib_cache=None,input_width=960,input_height = 540,calibration_dataset=None):
        try:
            import tensorrt as trt
        except:
            print("Error importing tensorrt. Please check your tensorrt installation...")
            exit(1)
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        def build_engine():
            with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
                    (1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            ) as network, builder.create_builder_config() as config, trt.OnnxParser(
                network, TRT_LOGGER
            ) as parser, trt.Runtime(
                TRT_LOGGER
            ) as runtime:
                config.max_workspace_size = 1 << 32
                builder.max_batch_size = 1

                if precision in ["fp16"]:
                    config.set_flag(trt.BuilderFlag.FP16)
                if precision in ["tf32"]:
                    config.set_flag(trt.BuilderFlag.TF32)


                if not os.path.exists(onnx_file_path):
                    print(
                        "ONNX file {} not found, first to generate it.".format(
                            onnx_file_path)
                    )
                    exit(0)
                print("Loading ONNX file from path {}...".format(onnx_file_path))
                with open(onnx_file_path, "rb") as model:
                    print("Beginning ONNX file parsing")
                    if not parser.parse(model.read()):
                        print("ERROR: Failed to parse the ONNX file.")
                        for error in range(parser.num_errors):
                            print(parser.get_error(error))
                        return None
                network.get_input(0).shape = [1, 3, input_height, input_width]
                network.get_input(1).shape = [1, 18, input_height, input_width]
                network.get_input(2).shape = [1, 12, input_height, input_width]
                print("Completed parsing of ONNX file")
                print("Building an engine from file {}; this may take a while...".format(onnx_file_path))

                if precision in ["int8"]:
                    config.set_flag(trt.BuilderFlag.FP16)
                    config.set_flag(trt.BuilderFlag.STRICT_TYPES)
                    for i in range(network.num_layers):
                        layer = network.get_layer(i)
                        #print(layer.name)
                        if  layer.type == trt.LayerType.CONVOLUTION and any([ #layer.type == trt.LayerType.CONVOLUTION and
                            "/network/stages" in layer.name,
                            "/network/gbuffer_encoder/class_encoders" in layer.name,
                            "/network/layer1" in layer.name,
                            "/network/stages" in layer.name,
                            "Unnamed Layer* 2" in layer.name,
                            "Unnamed Layer* 4" in layer.name,
                            "Unnamed Layer* 172" in layer.name,
                            "Unnamed Layer* 175" in layer.name,
                            "Unnamed Layer* 177" in layer.name,
                            "Unnamed Layer* 346" in layer.name,
                            "Unnamed Layer* 515" in layer.name,
                            "Unnamed Layer* 666" in layer.name,
                            "Unnamed Layer* 817" in layer.name,
                            "Unnamed Layer* 968" in layer.name,
                            "Unnamed Layer* 1119" in layer.name,
                            "Unnamed Layer* 1270" in layer.name,
                            "Unnamed Layer* 1421" in layer.name,
                            "Unnamed Layer* 1572" in layer.name,
                            "Unnamed Layer* 1723" in layer.name,
                            "Unnamed Layer* 6046" in layer.name
                        ]):
                            print(layer.name)
                            network.get_layer(i).precision = trt.DataType.HALF
                    #exit(1)"""
                    from epe.general.calibrator import EngineCalibrator,ImageBatcher
                    config.set_flag(trt.BuilderFlag.INT8)
                    config.int8_calibrator = EngineCalibrator(calib_cache)
                    if calib_cache is None or not os.path.exists(calib_cache):
                        config.int8_calibrator.set_image_batcher(ImageBatcher(calibration_dataset, [input_height,input_width], trt.nptype(trt.float32)))

                    engine_bytes = None
                    try:
                        engine_bytes = builder.build_serialized_network(network, config)
                    except AttributeError:
                        engine = builder.build_engine(network, config)
                        engine_bytes = engine.serialize()
                        del engine
                    assert engine_bytes
                    with open(engine_file_path, "wb") as f:
                        f.write(engine_bytes)
                    print("Successfully generated cache and trt files...Please re-execute the script...")
                    exit(1)

                plan = builder.build_serialized_network(network, config)
                engine = runtime.deserialize_cuda_engine(plan)
                print("Completed creating Engine")
                with open(engine_file_path, "wb") as f:
                    f.write(plan)
                return engine

        if os.path.exists(engine_file_path):
            print("Reading engine from file {}".format(engine_file_path))
            with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                return runtime.deserialize_cuda_engine(f.read())
        else:
            return build_engine()

    def validate_parameters(self,carla_config):
        from colorama import Back, Style

        if not carla_config['general']['data_type'] in ['fp32','fp16','tf32','int8']:
            print('\033[91m'+"Invalid parameter data_type. Valid values are ['fp32','fp16','tf32'].")
            exit(1)
        if not carla_config['general']['pygame_output'] in ['enhanced', 'rgb','ad_task']:
            print('\033[91m'+"Invalid parameter pygame_output. Valid values are ['enhanced', 'rgb','ad_task'].")
            exit(1)
        if not type(carla_config['general']['run_enhanced_model']) == bool:
            print('\033[91m'+"Invalid parameter run_enhanced_model. Valid values are True or False.")
            exit(1)
        if not carla_config['general']['compiler'] in ['tensorrt','onnxruntime','pytorch']:
            print('\033[91m'+"Invalid parameter compiler. Valid values are ['tensorrt','onnxruntime','pytorch'].")
            exit(1)
        if not carla_config['carla_world_settings']['camera_output'] in ['enhanced', 'rgb']:
            print('\033[91m'+"Invalid parameter camera_output. Valid values are ['enhanced', 'rgb'].")
            exit(1)
        if not carla_config['carla_world_settings']['driving_mode'] in ['ad_model', 'auto','rl_train','rl_eval','manual']:
            print('\033[91m'+"Invalid parameter driving_mode. Valid values are ['ad_model', 'auto','rl_train','rl_eval','manual'].")
            exit(1)
        if not (carla_config['carla_world_settings']['perspective'] == 0 or carla_config['carla_world_settings']['perspective'] == 1 or carla_config['carla_world_settings']['perspective'] == 2):
            print('\033[91m'+"Invalid parameter perspective. Valid values are [0, 1, 2].")
            exit(1)
        if not carla_config['carla_world_settings']['weather_preset'] in ["Default","ClearNoon","ClearSunset","CloudyNoon","CloudySunset","WetNoon","WetSunset","SoftRainNoon","SoftRainSunset","HardRainNoon","HardRainSunset","WetCloudyNoon","WetCloudySunset"]:
            print('\033[91m'+'Invalid parameter weather_preset. Valid values are ["Default","ClearNoon","ClearSunset","CloudyNoon","CloudySunset","WetNoon","WetSunset","SoftRainNoon","SoftRainSunset","HardRainNoon","HardRainSunset","WetCloudyNoon","WetCloudySunset"].')
            exit(1)
        if not carla_config['carla_world_settings']['spectator_camera_mode'] in ['follow','free']:
            print('\033[91m'+"Invalid parameter spectator_camera_mode. Valid values are ['follow','free'].")
            exit(1)
        if not type(carla_config['carla_world_settings']['load_parked_vehicles']) == bool:
            print('\033[91m'+"Invalid parameter load_parked_vehicles. Valid values are True or False.")
            exit(1)
        if not type(carla_config['carla_world_settings']['sync_mode']) == bool:
            print('\033[91m'+"Invalid parameter sync_mode. Valid values are True or False.")
            exit(1)
        if not type(carla_config['dataset_settings']['export_dataset']) == bool:
            print('\033[91m' + "Invalid parameter export_dataset. Valid values are True or False.")
            exit(1)

        if carla_config['general']['compiler'] == 'onnxruntime' and carla_config['general']['data_type'] == 'fp16':
            print('\033[93m'+"Warning...onnxruntime does not work correctly with fp16. Use fp32 or tensorrt compiler instead.")

        if not carla_config['general']['compiler'] == 'tensorrt' and carla_config['general']['data_type'] == 'int8':
            print('\033[91m'+"INT8 data type is only available for tensorrt compiler.")
            exit(1)

        if not carla_config['dataset_settings']['images_format'] in ['png','jpg']:
            print('\033[91m'+"The supported images formats are png and jpg.")
            exit(1)

        if carla_config['carla_world_settings']['sync_mode'] == False:
            print('\033[93m' + "Warning...Running in asynchronous mode may lead to different problems depending on the system computing capability. We recommend using asynchronous mode only for visualization.")

        if carla_config['carla_world_settings']['sync_mode'] == False and carla_config['dataset_settings']['export_dataset'] == True:
            print('\033[91m' + "Data extraction is supported only is in synchronous mode.")
            exit(1)

        if carla_config['carla_world_settings']['camera_output'] == 'enhanced' and carla_config['general']['run_enhanced_model'] == False:
            print('\033[91m' + "Please enable 'run_enhanced_model' parameter in carla_config if you want to use enhanced camera output.")
            exit(1)

        if carla_config['general']['async_data_transfer'] == True and (not carla_config['general']['compiler'] == 'tensorrt' or not carla_config['carla_world_settings']['camera_output'] == 'enhanced' or carla_config['carla_world_settings']['sync_mode'] == True):
            print('\033[91m' + "Async data transfer is only supported in asynchronous mode using tensorrt compiler and enhanced camera output.")
            exit(1)
        
        if not carla_config['general']['method'] in ['EPE','REGEN']:
            print('\033[91m' + "The supported methods are EPE and REGEN. Available options: ['EPE','REGEN']")
            exit(1)

        if not carla_config['onnx_runtime_settings']['execution_provider'] in ['CUDAExecutionProvider', 'TensorrtExecutionProvider']:
            print('\033[91m'+"Invalid parameter for execution_provider. Valid values are ['CUDAExecutionProvider', 'TensorrtExecutionProvider'.")
            exit(1)

        if "Rain" in carla_config['carla_world_settings']['weather_preset'] and "cityscapes" in self.weight_init:
            print('\033[93m' + "Warning...If you select rain weather presets with Cityscapes pretrained models we recommend enabling no_render_mode perameter.")

    def host_device_thread(self,inputs,expected_data,method):
        while True:
            try:
                global data_dict
                if len(data_dict) == expected_data:
                    img = np.expand_dims(self.make_image(method), axis=0)
                    gb = np.expand_dims(self.make_gbuffers(), axis=0)
                    gt = np.expand_dims(self.make_gtlabels(), axis=0)
                    inputs[0].host = img
                    inputs[1].host = gb
                    inputs[2].host = gt
            except:
                pass

    def visualize_buffers(self,img_width,img_height):
        global data_dict

        try:
            images = [data_dict['SceneColor'],data_dict['SceneDepth'],data_dict['GBufferA'],data_dict['GBufferB'],data_dict['GBufferC'],data_dict['GBufferD'],data_dict['GBufferSSAO'],data_dict['CustomStencil'],data_dict['color_frame']]
            names = ['SceneColor','SceneDepth','GBufferA','GBufferB','GBufferC','GBufferD','GBufferSSAO','CustomStencil','Final Frame']

            for img_idx in range(len(images)):
                if img_width > 500:
                    new_size = (int(img_width/2), int(img_height/2))
                    images[img_idx] = cv2.resize(images[img_idx], new_size)
                text_size = cv2.getTextSize(names[img_idx], cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                images[img_idx] = cv2.putText(images[img_idx], names[img_idx], (10, text_size[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (234,182,118), 1)

            row1 = np.concatenate(images[:3], axis=1)
            row2 = np.concatenate(images[3:6], axis=1)
            row3 = np.concatenate(images[6:], axis=1)

            result_image = np.concatenate([row1, row2, row3], axis=0)

            result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
            cv2.imshow('Buffers Visualizer', result_image)
        except Exception as e:
            print(e)

    def evaluate_infer(self):
        global data_dict
        global names_dict
        global weather_mapping
        global export_dataset_path
        global enh_width
        global enh_height
        found_scenario = True


        with open(carla_config_path, 'r') as file:
            carla_config = yaml.safe_load(file)

        self.validate_parameters(carla_config)

        val_ticks = None
        try:
            with open("../scenarios/"+str(carla_config['autonomous_driving']['scenario']+".yaml"), 'r') as file:
                scenario_config = yaml.safe_load(file)
            val_ticks = scenario_config['general']['val_ticks']
        except:
            found_scenario = False

        print("Clearing older ONNX profiles...")
        file_list = os.listdir(os.getcwd())
        for filename in file_list:
            if "onnxruntime_profile__" in filename:
                file_path = os.path.join(os.getcwd(), filename)
                os.remove(file_path)
                print(f"Deleted: {file_path}")

        actor_list = []

        client = carla.Client(carla_config['carla_server_settings']['ip'], carla_config['carla_server_settings']['port'])
        client.set_timeout(carla_config['carla_server_settings']['timeout'])
        sync_mode = carla_config['carla_world_settings']['sync_mode']
        selected_town = carla_config['carla_world_settings']['town']
        selected_weather_preset = carla_config['carla_world_settings']['weather_preset']
        selected_perspective = carla_config['carla_world_settings']['perspective']
        export_dataset = carla_config['dataset_settings']['export_dataset']
        export_dataset_path = carla_config['dataset_settings']['dataset_path']
        driving_mode = carla_config['carla_world_settings']['driving_mode']
        selected_camera_output = carla_config['carla_world_settings']['camera_output']
        num_skip_frames = carla_config['carla_world_settings']['skip_frames']
        rl_action_dim = carla_config['autonomous_driving']['rl_action_dim']
        rl_model_load_episode = carla_config['autonomous_driving']['rl_model_load_episode']
        rl_model_name = carla_config['autonomous_driving']['rl_model_name']
        spectator_camera_mode = carla_config['carla_world_settings']['spectator_camera_mode']
        stabilize_ticks = carla_config['autonomous_driving']['stabilize_num_ticks']
        use_lidar = carla_config['other_sensors_settings']['use_lidar']
        use_radar = carla_config['other_sensors_settings']['use_radar']
        use_imu = carla_config['other_sensors_settings']['use_imu']
        use_gnss = carla_config['other_sensors_settings']['use_gnss']
        compiler = carla_config['general']['compiler']
        async_data_transfer = carla_config['general']['async_data_transfer']
        execution_provider = carla_config['onnx_runtime_settings']['execution_provider']
        enable_fp16_onnx = carla_config['onnx_runtime_settings']['enable_fp16']
        infer_method = carla_config['general']['method']

        onnx_path = "..\\checkpoints\\ONNX\\"+self.weight_init+".onnx"
        dtype = carla_config['general']['data_type']
        rl_ego_transform = None
        ticks_counter = 0
        data_length = 10
        inputs = None
        manual_controls = carla_config['carla_world_settings']['manual_controls']

        if '/Game/Carla/Maps/'+selected_town not in client.get_available_maps():
            print('\033[91m'+"The selected Town does not exist. Below is a list with all the available Towns:")
            print(client.get_available_maps())
            exit(1)

        if use_lidar:
            data_length += 1
        if use_radar:
            data_length += 1
        if use_imu:
            data_length += 1
        if use_gnss:
            data_length += 1
        if carla_config['dataset_settings']['export_object_annotations'] or carla_config['dataset_settings']['export_instance_gt']:
            data_length += 1

        if selected_camera_output == 'enhanced' and carla_config['general']['run_enhanced_model'] == False:
            print('\033[91m'+"When using enhanced camera output the run_enhanced_model option must be enabled.")
            exit(1)

        if infer_method == "REGEN": #REGEN supports only pytorch
            compiler = "pytorch"           

        if export_dataset == True:
            self.create_dataset_folders(carla_config)

        if compiler == 'onnxruntime' or compiler == 'tensorrt':
            if not os.path.exists(onnx_path):

                path = "..\checkpoints\ONNX"
                print('The onnx model path does not exist. Trying to generate at '+path+".")
                if not os.path.exists(path):
                    os.makedirs(path)
                    print("Created ONNX directory...")
                path = os.path.join(path,self.weight_init+".onnx")
                self.generate_onxx(path,carla_config['ego_vehicle_settings']['camera_width'],carla_config['ego_vehicle_settings']['camera_height'])
                print("ONNX file generated successfully at "+path+".")
        if compiler == 'onnxruntime':
            opts = onnxruntime.SessionOptions()
            opts.enable_profiling = True

            if dtype == 'fp16':
                if not os.path.exists(os.path.join("..\checkpoints\ONNX",self.weight_init+"_fp16.onnx")):
                    model = onnx.load(os.path.join("..\checkpoints\ONNX",self.weight_init+".onnx"))
                    model_fp16 = onnxconverter_common.convert_float_to_float16(model)
                    onnx.save(model_fp16, os.path.join("..\checkpoints\ONNX",self.weight_init+"_fp16.onnx"))
                onnx_path = os.path.join("..\checkpoints\ONNX",self.weight_init+"_fp16.onnx")
            
            if execution_provider == "TensorrtExecutionProvider" and enable_fp16_onnx:
                print("Exporting tensorrt engine, please wait...")
                session = onnxruntime.InferenceSession(
                onnx_path,
                opts,
                providers=[
                    ("TensorrtExecutionProvider", {
                        "trt_fp16_enable": True,
                    }),
                    "CUDAExecutionProvider"
                ]
                )
            else:
                if execution_provider == "TensorrtExecutionProvider":
                    print("Exporting tensorrt engine, please wait...")
                session = onnxruntime.InferenceSession(
                    onnx_path,
                    opts,
                    providers=[
                        execution_provider
                    ],
                )
            io_binding = session.io_binding()
        elif compiler == 'tensorrt':
            try:
                pass
                sys.path.insert(1, carla_config['general']['tensorrt_common_path'])
                import common
                import tensorrt as trt
                tensorrt_success = True
            except:
                print('\033[93m'+"Error importing tensorrt. Please check your TensorRT installation...Setting Pytorch as the compiler.")
                tensorrt_success = False

            if tensorrt_success:
                path, file = os.path.split(onnx_path)
                trt_file = file.split(".")[0]+"_"+str(dtype)+".trt"
                cache_file = file.split(".")[0]+".cache"
                TRT_LOGGER = trt.Logger(trt.Logger.INFO)
                trt.init_libnvinfer_plugins(TRT_LOGGER, '')
                engine = self.get_engine(onnx_path, os.path.join(path,trt_file),dtype,os.path.join(path,cache_file),carla_config['ego_vehicle_settings']['camera_width'],carla_config['ego_vehicle_settings']['camera_height'],carla_config['general']['calibration_dataset'])
                context = engine.create_execution_context()
                inputs, outputs, bindings, stream = common.allocate_buffers(engine)
            else:
                compiler = 'pytorch'

        #if pytorch compiler is not used then delete the pytorch model and free the memory
        if compiler == 'tensorrt' or compiler == 'onnxruntime' or infer_method == "REGEN":
            del self.network.generator
        
        if infer_method == "REGEN":
            generator_ema = regen_generator.define_G(
                input_nc= int(carla_config['REGEN_settings']['input_nc']),
                output_nc= int(carla_config['REGEN_settings']['output_nc']),
                ngf=int(carla_config['REGEN_settings']['ngf']),
                netG=str(carla_config['REGEN_settings']['netG']),
                norm=str(carla_config['REGEN_settings']['norm']),
                n_downsample_global= int(carla_config['REGEN_settings']['n_downsample_global']),
                n_blocks_global= int(carla_config['REGEN_settings']['n_blocks_global']),
                n_local_enhancers= int(carla_config['REGEN_settings']['n_local_enhancers'])
            ).to(self.device)

            checkpoint = torch.load(os.path.join("..\checkpoints\REGEN", str(carla_config['REGEN_settings']['checkpoint_name'])), map_location=self.device)
            generator_ema.load_state_dict(checkpoint)
            generator_ema.eval()


            

        world = client.get_world()
        world = client.load_world(selected_town, carla.MapLayer.Buildings | carla.MapLayer.ParkedVehicles)

        if carla_config['carla_world_settings']['load_parked_vehicles'] == False:
            world.unload_map_layer(carla.MapLayer.ParkedVehicles)

        settings = world.get_settings()
        settings.synchronous_mode = sync_mode
        settings.no_rendering_mode = carla_config['carla_world_settings']['no_rendering_mode']
        settings.fixed_delta_seconds = carla_config['carla_world_settings']['fixed_delta_seconds']
        world.apply_settings(settings)

        if str(selected_weather_preset) != "Default":
            world.set_weather(weather_mapping[selected_weather_preset])

        blueprint_library = world.get_blueprint_library()

        cars = ['vehicle.chevrolet.impala', 'vehicle.tesla.model3', 'vehicle.mercedes.coupe_2020']
        car_cam_coord = [[1.2, -0.3, 1.4], [0.8, -0.3, 1.4], [1.0, -0.3, 1.4]]
        if carla_config['ego_vehicle_settings']['vehicle_model'] == 'random':
            selected_random_car = random.randint(0, len(cars) - 1)
            bp = random.choice(blueprint_library.filter(cars[selected_random_car]))
            selected_vehicle_name = cars[selected_random_car]
        else:
            selected_random_car = 0
            bp = blueprint_library.filter(carla_config['ego_vehicle_settings']['vehicle_model'])[0]
            selected_vehicle_name = carla_config['ego_vehicle_settings']['vehicle_model']

        actor_list_remaining = world.get_actors()

        for actor in actor_list_remaining:
            if 'vehicle' in actor.type_id or 'walker.pedestrian' in actor.type_id:
                actor.destroy()

        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)

        if carla_config['ego_vehicle_settings']['init_spawn_point'] == 'random':
            transform = world.get_map().get_spawn_points()[random.randint(0, len(world.get_map().get_spawn_points()) - 1)]
        elif isinstance(carla_config['ego_vehicle_settings']['init_spawn_point'], int):
            transform = world.get_map().get_spawn_points()[carla_config['ego_vehicle_settings']['init_spawn_point']]
        elif isinstance(carla_config['ego_vehicle_settings']['init_spawn_point'], list):
            coords = carla_config['ego_vehicle_settings']['init_spawn_point']
            transform = carla.Transform(carla.Location(x=coords[0][0], y=coords[0][1], z=coords[0][2]), carla.Rotation(coords[1][0],coords[1][1],coords[1][2]))

        vehicle = world.spawn_actor(bp, transform)

        actor_list.append(vehicle)
        print('created %s' % vehicle.type_id)

        if driving_mode == "auto":
            vehicle.set_autopilot(True)
        else:
            vehicle.set_autopilot(False)

        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(carla_config['ego_vehicle_settings']['camera_width']))
        camera_bp.set_attribute('image_size_y', str(carla_config['ego_vehicle_settings']['camera_height']))
        camera_bp.set_attribute('fov', str(carla_config['ego_vehicle_settings']['camera_fov']))

        if selected_perspective == 1:
            camera_transform = carla.Transform(
                carla.Location(x=car_cam_coord[selected_random_car][0], y=car_cam_coord[selected_random_car][1],
                               z=car_cam_coord[selected_random_car][2]))
        elif selected_perspective == 0:
            camera_transform = carla.Transform(carla.Location(x=1.5, z=1.4))
        elif selected_perspective == 2:
            camera_transform = carla.Transform(carla.Location(x=carla_config['ego_vehicle_settings']['camera_location'][0],y=carla_config['ego_vehicle_settings']['camera_location'][1] ,z=carla_config['ego_vehicle_settings']['camera_location'][2]))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

        actor_list.append(camera)
        print('created %s' % camera.type_id)

        #camera_bp2 = blueprint_library.find('sensor.camera.semantic_segmentation')
        #camera_bp2.set_attribute('image_size_x', str(carla_config['ego_vehicle_settings']['camera_width']))
        #camera_bp2.set_attribute('image_size_y', str(carla_config['ego_vehicle_settings']['camera_height']))
        #camera_bp2.set_attribute('fov', str(carla_config['ego_vehicle_settings']['camera_fov']))
        #camera_semseg = world.spawn_actor(camera_bp2, camera_transform, attach_to=vehicle)

        #actor_list.append(camera_semseg)
        #print('created %s' % camera_semseg.type_id)

        if use_lidar:
            lidar_transform_conf = carla_config['lidar_settings']['transform']
            lidar_transform = carla.Transform(carla.Location(x=lidar_transform_conf[0][0],y=lidar_transform_conf[0][1] ,z=lidar_transform_conf[0][2]),carla.Rotation(lidar_transform_conf[0][0],lidar_transform_conf[0][1],lidar_transform_conf[0][2]))
            lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
            lidar_bp.set_attribute('channels',str(carla_config['lidar_settings']['channels']))
            lidar_bp.set_attribute('range', str(carla_config['lidar_settings']['range']))
            lidar_bp.set_attribute('points_per_second', str(carla_config['lidar_settings']['points_per_second']))
            lidar_bp.set_attribute('rotation_frequency', str(carla_config['lidar_settings']['rotation_frequency']))
            lidar_bp.set_attribute('upper_fov', str(carla_config['lidar_settings']['upper_fov']))
            lidar_bp.set_attribute('horizontal_fov', str(carla_config['lidar_settings']['horizontal_fov']))
            lidar_bp.set_attribute('atmosphere_attenuation_rate', str(carla_config['lidar_settings']['atmosphere_attenuation_rate']))
            lidar_bp.set_attribute('dropoff_general_rate', str(carla_config['lidar_settings']['dropoff_general_rate']))
            lidar_bp.set_attribute('dropoff_intensity_limit', str(carla_config['lidar_settings']['dropoff_intensity_limit']))
            lidar_bp.set_attribute('dropoff_zero_intensity', str(carla_config['lidar_settings']['dropoff_zero_intensity']))
            lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
            actor_list.append(lidar)
            print('created %s' % lidar.type_id)

        if use_radar:
            radar_transform_conf = carla_config['radar_settings']['transform']
            radar_transform = carla.Transform(carla.Location(x=radar_transform_conf[0][0],y=radar_transform_conf[0][1] ,z=radar_transform_conf[0][2]),carla.Rotation(radar_transform_conf[0][0],radar_transform_conf[0][1],radar_transform_conf[0][2]))
            radar_bp = world.get_blueprint_library().find('sensor.other.radar')
            radar_bp.set_attribute('horizontal_fov',str(carla_config['radar_settings']['horizontal_fov']))
            radar_bp.set_attribute('vertical_fov', str(carla_config['radar_settings']['vertical_fov']))
            radar_bp.set_attribute('points_per_second', str(carla_config['radar_settings']['points_per_second']))
            radar_bp.set_attribute('range', str(carla_config['radar_settings']['range']))
            radar = world.spawn_actor(radar_bp, radar_transform, attach_to=vehicle)
            actor_list.append(radar)
            print('created %s' % radar.type_id)

        if use_imu:
            imu_bp = world.get_blueprint_library().find('sensor.other.imu')
            imu = world.spawn_actor(imu_bp, camera_transform, attach_to=vehicle)
            actor_list.append(imu)
            print('created %s' % imu.type_id)

        if use_gnss:
            gnss_bp = world.get_blueprint_library().find('sensor.other.gnss')
            gnss = world.spawn_actor(gnss_bp, camera_transform, attach_to=vehicle)
            actor_list.append(gnss)
            print('created %s' % gnss.type_id)

        if carla_config['dataset_settings']['export_object_annotations'] or carla_config['dataset_settings']['export_instance_gt']:
            instance_bp = blueprint_library.find('sensor.camera.instance_segmentation')
            instance_bp.set_attribute('image_size_x', str(carla_config['ego_vehicle_settings']['camera_width']))
            instance_bp.set_attribute('image_size_y', str(carla_config['ego_vehicle_settings']['camera_height']))
            instance_bp.set_attribute('fov', str(carla_config['ego_vehicle_settings']['camera_fov']))
            actor_list.append(instance_bp)
            cam_instance = world.spawn_actor(instance_bp, camera_transform, attach_to=vehicle)


        try:

            camera.listen(lambda image: self.add_frame(image))
            camera.listen_to_gbuffer(carla.GBufferTextureID.SceneColor,
                                     lambda image: self.add_gbuffer(image, "SceneColor"))
            camera.listen_to_gbuffer(carla.GBufferTextureID.SceneDepth,
                                     lambda image: self.add_gbuffer(image, "SceneDepth"))
            camera.listen_to_gbuffer(carla.GBufferTextureID.GBufferA,
                                     lambda image: self.add_gbuffer(image, "GBufferA"))
            camera.listen_to_gbuffer(carla.GBufferTextureID.GBufferB,
                                     lambda image: self.add_gbuffer(image, "GBufferB"))
            camera.listen_to_gbuffer(carla.GBufferTextureID.GBufferC,
                                     lambda image: self.add_gbuffer(image, "GBufferC"))
            camera.listen_to_gbuffer(carla.GBufferTextureID.GBufferD,
                                     lambda image: self.add_gbuffer(image, "GBufferD"))

            camera.listen_to_gbuffer(carla.GBufferTextureID.SSAO,
                                     lambda image: self.add_gbuffer(image, "GBufferSSAO"))

            camera.listen_to_gbuffer(carla.GBufferTextureID.CustomStencil,
                                     lambda image: self.add_gbuffer(image, "CustomStencil"))

            #camera_semseg.listen(lambda image: self.add_semantic(image))

            if use_lidar:
                lidar.listen(lambda data: self.add_sensor(data,'lidar'))
            if use_radar:
                radar.listen(lambda data: self.add_sensor(data,'radar'))
            if use_gnss:
                gnss.listen(lambda data: self.add_sensor(data, 'gnss'))
            if use_imu:
                imu.listen(lambda data: self.add_sensor(data, 'imu'))
            if carla_config['dataset_settings']['export_object_annotations'] or carla_config['dataset_settings']['export_instance_gt']:
                cam_instance.listen(lambda img: self.add_sensor(img,'instance_segmentation'))


            enh_width = carla_config['ego_vehicle_settings']['camera_width']
            enh_height = carla_config['ego_vehicle_settings']['camera_height']

            renderObject = RenderObject(enh_width, enh_height)
            # Initialise the display
            pygame.init()
            gameDisplay = pygame.display.set_mode((enh_width, enh_height), pygame.HWSURFACE | pygame.DOUBLEBUF)

            self.initialize_gt_labels(enh_width,enh_height,29)

            frame_queue = queue.Queue()
            worker_thread = threading.Thread(target=self.render_thread, args=(frame_queue, renderObject, infer_method,))
            worker_thread.start()

            if async_data_transfer:
                transfer_thread = threading.Thread(target=self.host_device_thread, args=(inputs,data_length,infer_method,))
                transfer_thread.start()

            done_simulation = False
            self.network.eval()

            if dtype == "fp16" or dtype == 'tf32':
                forward_data_type = torch.float16
            else:
                forward_data_type = torch.float32

            if dtype == 'tf32':
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            else:
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False

            if driving_mode == "ad_model":
                autonomous_model = ADModel().to(self.device)
                autonomous_model.load()
                autonomous_model.eval()

            elif driving_mode == "rl_train":
                rl_environment = AutonomousDrivingEnvironment()
                rl_num_episodes_save = carla_config['autonomous_driving']['rl_num_episodes_save']
                current_episode = 0
                autonomous_agent = RLModel(rl_action_dim,replay_buffer_size=carla_config['autonomous_driving']['rl_buffer_max_size'])
                rl_total_reward = 0
                done = False
                state = rl_environment.reset(vehicle)
                rl_steps_counter = 0
                steps_per_episode = []
                rewards_per_episode = []
                critic_loss = []
                actor_loss = []
                vehicle_distance = []
                critic_loss_per_episode = []
                actor_loss_per_episode = []
                vehicle_distance_per_episode = []
                epsilon_per_episode = []
                epsilons = []
                best_total_reward = 0

                collision_sensor_bp = blueprint_library.find('sensor.other.collision')
                collision_sensor = world.spawn_actor(collision_sensor_bp, camera_transform, attach_to=vehicle)
                collision_sensor.listen(lambda event: self.on_collision(event, rl_environment))
                actor_list.append(collision_sensor)
            elif driving_mode == "rl_eval":
                rl_environment = AutonomousDrivingEnvironment()
                autonomous_agent = RLModel(rl_action_dim,replay_buffer_size=carla_config['autonomous_driving']['rl_buffer_max_size'])
                autonomous_agent.load(rl_model_name,rl_model_load_episode)
                autonomous_agent.eval_mode()
                state = rl_environment.reset(vehicle)

            if carla_config['general']['pygame_output'] == 'ad_task':
                ad_task_ref = ADTask()


            if found_scenario:
                rl_ego_transform = self.initialize_scenario(world,scenario_config,vehicle)
                self.stabilize_vehicle(world,spectator_camera_mode,camera,stabilize_ticks,data_length)
            elif found_scenario == False and driving_mode == 'rl_train':
                print('\033[91m'+"You need to select a valid scenario for reinforcement learning training...")
                sys.exit(0)

            while not done_simulation:

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        done_simulation = True

                gameDisplay.blit(renderObject.surface, (0, 0))
                pygame.display.flip()

                if spectator_camera_mode == 'follow':
                    world.get_spectator().set_transform(camera.get_transform())

                if sync_mode == True or sync_mode == False:
                    start_timer = time.time()

                    if driving_mode == "rl_train":
                        action = autonomous_agent.select_action(state, carla_config['autonomous_driving']['rl_use_exploration'])
                        rl_environment.apply_action(action, vehicle)

                    elif driving_mode == "rl_eval":
                        action = autonomous_agent.select_action(state, False)
                        rl_environment.apply_action(action, vehicle)

                    if sync_mode == True:
                        if num_skip_frames > 0:
                            for i in range(num_skip_frames):
                                data_dict = {}
                                names_dict = {}
                                world.tick()
                                if spectator_camera_mode == 'follow':
                                    world.get_spectator().set_transform(camera.get_transform())
                                if driving_mode == "manual":
                                    self.manual_controls_apply(vehicle,manual_controls)
                                while True:
                                    if len(data_dict) == data_length:
                                        break
                                data_dict = {}
                                names_dict = {}

                        world.tick()
                    ticks_counter += 1
                    if driving_mode == "auto":
                        vehicle.set_autopilot(True)

                    if driving_mode == "manual":
                        self.manual_controls_apply(vehicle,manual_controls)

                    if spectator_camera_mode == 'follow':
                        world.get_spectator().set_transform(camera.get_transform())


                    if found_scenario:
                        self.trigger_scenario(world,scenario_config,vehicle)

                    if sync_mode == True:
                        frame_found = False
                        gt_labels_found = False
                        gbuffers_found = False
                        frame_thread = None
                        gt_labels_thread = None
                        gbuffers_thread = None
                        while True:
                            if "color_frame" in data_dict and frame_found == False:
                                frame_found = True
                                frame_thread = threading.Thread(target=self.preprocess_worker, args=("frame",compiler,inputs,infer_method,))
                                frame_thread.start()

                            if "semantic_segmentation" in data_dict and gt_labels_found == False:
                                gt_labels_found = True
                                gt_labels_thread = threading.Thread(target=self.preprocess_worker, args=("gt_labels",compiler,inputs,infer_method,))
                                gt_labels_thread.start()

                            if "SceneColor" in data_dict and "SceneDepth" in data_dict and "GBufferA" in data_dict and "GBufferB" in data_dict and "GBufferC" in data_dict and "GBufferD" in data_dict and "GBufferSSAO" in data_dict and "CustomStencil" in data_dict and gbuffers_found == False:
                                gbuffers_found = True
                                gbuffers_thread = threading.Thread(target=self.preprocess_worker, args=("gbuffers",compiler,inputs,infer_method,))
                                gbuffers_thread.start()
                            if len(data_dict) == data_length:
                                if gbuffers_found == False:
                                    gbuffers_thread = threading.Thread(target=self.preprocess_worker, args=("gbuffers",compiler,inputs,infer_method,))
                                    gbuffers_thread.start()
                                if frame_found == False:
                                    frame_thread = threading.Thread(target=self.preprocess_worker, args=("frame",compiler,inputs,infer_method,))
                                    frame_thread.start()
                                if gt_labels_found == False:
                                    gt_labels_thread = threading.Thread(target=self.preprocess_worker, args=("gt_labels",compiler,inputs,infer_method,))
                                    gt_labels_thread.start()
                                break
                    else:
                        if async_data_transfer == False:
                            frame_thread = threading.Thread(target=self.preprocess_worker,args=("frame", compiler, inputs,infer_method,))
                            frame_thread.start()
                            gbuffers_thread = threading.Thread(target=self.preprocess_worker,args=("gbuffers", compiler, inputs,infer_method,))
                            gbuffers_thread.start()
                            gt_labels_thread = threading.Thread(target=self.preprocess_worker,args=("gt_labels", compiler, inputs,infer_method,))
                            gt_labels_thread.start()
                    if async_data_transfer == False:
                        frame_thread.join()
                        gt_labels_thread.join()
                        gbuffers_thread.join()
                        img = result_container['frame']
                        label_map = result_container['gt_labels']
                        gbuffers = result_container['gbuffers']


                    #print(names_dict)
                    if carla_config['general']['visualize_buffers'] == True:
                        self.visualize_buffers(enh_width, enh_height)
                    if len(data_dict) == data_length:

                        if carla_config['general']['run_enhanced_model'] == True:
                            if compiler == 'onnxruntime':
                                tdtype = np.float32
                                if dtype == 'fp16':
                                    tdtype = np.float16
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
                            elif compiler == 'tensorrt':
                                infer_timer = time.time()
                                [output] = common.do_inference_v2(context, bindings=bindings, inputs=inputs , outputs=outputs, stream=stream)
                                print("Inference time: " + str(time.time() - infer_timer))
                                new_img = torch.from_numpy(output.reshape(1, 3, 540, 960))
                            else:
                                with torch.no_grad():
                                    with torch.cuda.amp.autocast_mode.autocast(dtype=forward_data_type):

                                        if infer_method == "EPE":
                                            batch = EPEBatch(img, gbuffers=gbuffers, gt_labels=label_map, robust_labels=None, path=None,coords=None).to(self.device)
                                            infer_timer = time.time()
                                            new_img = self.network.generator(batch)
                                            print("Inference time: " + str(time.time() - infer_timer))
                                            pass
                                        else:
                                            infer_timer = time.time()
                                            new_img = generator_ema(img)
                                            print("Inference time: " + str(time.time() - infer_timer))
                                            pass


                        if carla_config['general']['pygame_output'] == 'enhanced' and carla_config['general']['run_enhanced_model'] == True:
                            frame_queue.put(new_img)
                        elif (carla_config['general']['pygame_output'] == 'enhanced' and carla_config['general']['run_enhanced_model'] == False) or carla_config['general']['pygame_output'] == 'rgb':
                            renderObject.surface = pygame.surfarray.make_surface(data_dict['color_frame'].swapaxes(0, 1))
                        else:
                            if selected_camera_output == "enhanced" and carla_config['general']['run_enhanced_model'] == True:
                                img = self.process_final_image(new_img,infer_method)
                                ad_task_frame = ad_task_ref.predict_output(img,np.ascontiguousarray(data_dict['semantic_segmentation']),world,vehicle,camera,data_dict)
                            else:
                                img = np.ascontiguousarray(data_dict['color_frame'])
                                ad_task_frame = ad_task_ref.predict_output(img,np.ascontiguousarray(data_dict['semantic_segmentation']),world,vehicle,camera,data_dict)
                            renderObject.surface = pygame.surfarray.make_surface(ad_task_frame.swapaxes(0, 1))

                        if selected_camera_output == "enhanced" and driving_mode == "ad_model":
                            enhanced_frame = self.process_final_image(new_img,infer_method)
                            controls_predicted = autonomous_model.test_single(enhanced_frame)
                            print(controls_predicted)
                        elif selected_camera_output == "rgb" and driving_mode == "ad_model":
                            controls_predicted = autonomous_model.test_single(data_dict['color_frame'])
                            print(controls_predicted)

                        if driving_mode == "ad_model":
                            if float(controls_predicted['brake']) > carla_config['autonomous_driving']['ad_brake_threshold']:
                                controls_predicted['brake'] = 1.0
                            else:
                                controls_predicted['brake'] = 0.0
                            vehicle.apply_control(carla.VehicleControl(throttle=float(controls_predicted['throttle']),
                                                                       steer=float(controls_predicted['steering']),
                                                                       brake=float(controls_predicted['brake'])))
                            print(controls_predicted)


                        if driving_mode == "rl_train" or driving_mode == "rl_eval":
                            if selected_camera_output == "enhanced":
                                next_state = self.process_final_image(new_img,infer_method)
                                next_state = autonomous_agent.preprocess_camera_frame(next_state)
                            else:
                                rgb_frame = autonomous_agent.preprocess_camera_frame(np.ascontiguousarray(data_dict['color_frame']))
                                next_state = rgb_frame


                            if driving_mode == "rl_train":
                                reward, done,next_state, _ = rl_environment.calculate_reward(action, vehicle, world, ticks_counter, next_state, data_dict)
                                autonomous_agent.add_observation(state, action, next_state, reward, done)

                                actorloss,criticloss = autonomous_agent.train()

                                critic_loss.append(criticloss)
                                actor_loss.append(actorloss)
                                epsilons.append(autonomous_agent.epsilon)
                                vehicle_distance.append(vehicle.get_location().distance(rl_ego_transform.location))
                                rl_total_reward += reward
                                rl_steps_counter += 1
                                print("total reward:" + str(rl_total_reward))
                                if done == True:
                                    critic_loss_per_episode.append(critic_loss)
                                    actor_loss_per_episode.append(actor_loss)
                                    vehicle_distance_per_episode.append(vehicle_distance)
                                    epsilon_per_episode.append(epsilons)
                                    critic_loss = []
                                    actor_loss = []
                                    vehicle_distance = []
                                    epsilons = []
                                    rewards_per_episode.append(rl_total_reward)
                                    if rl_total_reward > best_total_reward:
                                        best_total_reward = rl_total_reward
                                        autonomous_agent.save(rl_model_name,"best")
                                    rl_total_reward = 0
                                    steps_per_episode.append(rl_steps_counter)
                                    rl_steps_counter = 0
                                    state = rl_environment.reset(vehicle)
                                    self.initialize_scenario(world, scenario_config, vehicle)
                                    self.stabilize_vehicle(world, spectator_camera_mode, camera,int(stabilize_ticks/2),data_length)
                                    self.initialize_scenario(world, scenario_config, vehicle)
                                    self.stabilize_vehicle(world, spectator_camera_mode, camera, int(stabilize_ticks/2),data_length)
                                    current_episode+=1
                                    state = next_state

                                    if current_episode % rl_num_episodes_save == 0:
                                        autonomous_agent.save(rl_model_name,current_episode)
                                        self.save_rl_stats(actor_losses=actor_loss_per_episode,
                                                           critic_losses=critic_loss_per_episode,
                                                           steps=steps_per_episode, rewards=rewards_per_episode,
                                                           current_step=current_episode,
                                                           distances=vehicle_distance_per_episode,episode=current_episode,epsilon=epsilon_per_episode)
                                else:
                                    state = next_state
                            elif driving_mode == "rl_eval":
                                state = next_state
                        if (driving_mode == "rl_eval" or driving_mode == "ad_model") and val_ticks is not None: #driving_mode == "rl_eval" or driving_mode == "ad_model"
                            if val_ticks < ticks_counter:
                                sys.exit(0)

                    else:
                        print("--Unsync--")
                        print(names_dict)
                        print("-------")

                        # Update PyGame window
                        gameDisplay.fill((0, 0, 0))
                        gameDisplay.blit(renderObject.surface, (0, 0))
                        pygame.display.flip()

                    if export_dataset and carla_config['general']['run_enhanced_model'] == True and driving_mode != 'rl_train':
                        if carla_config['dataset_settings']['capture_when_static'] == True or (carla_config['dataset_settings']['capture_when_static']==False and self.is_vehicle_moving(vehicle,carla_config['dataset_settings']['speed_threshold'])):
                            random_id = self.random_id_generator()
                            if carla_config['dataset_settings']['export_status_json']:
                                self.save_vehicle_status(vehicle, random_id + str(names_dict['color_frame']))
                                self.save_world_status(selected_town, selected_weather_preset, selected_vehicle_name,
                                                       selected_perspective, sync_mode,
                                                       random_id + str(names_dict['color_frame']))
                            self.save_frames(random_id + str(names_dict['color_frame']), new_img, carla_config, infer_method)

                            if carla_config['dataset_settings']['export_object_annotations']:
                                self.save_object_detection_annotations(camera,world,vehicle,random_id + str(names_dict['color_frame']),carla_config)
                    elif export_dataset and carla_config['general']['run_enhanced_model'] == False:
                        print("Warning: To export frames you need to enable run_enhanced_model from the carla_config file.")
                    elif export_dataset and driving_mode == 'rl_train':
                        print("Warning: To export frames you need to be in evaluation or auto drive mode.")

                    if sync_mode == True:
                        names_dict = {}
                        data_dict = {}

                    print("Executin Delay: " + str(time.time() - start_timer))


        except Exception as error: #(KeyboardInterrupt,SystemExit,subprocess.CalledProcessError)
            print(error)
            settings.synchronous_mode = False
            world.apply_settings(settings)
            if compiler == 'tensorrt':
                common.free_buffers(inputs, outputs, stream)
            print('Destroying actors...')
            camera.destroy()
            #camera_semseg.destroy()
            if driving_mode == "rl_train":
                collision_sensor.destroy()
            vehicle.destroy()
            if other_actor is not None:
                other_actor.destroy()
            client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
            print('Done...')
            frame_queue.put(None)
            print('Terminating render thread...')
            print('Exiting...')
            sys.exit()

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def _load_sample(self):
        """ Loads a single example (preferably from self.args.input). """
        raise NotImplementedError
        return batch

    def _save_model(self, *, epoch=None, iterations=None, reason=None):
        raise NotImplementedError

    def _load_model(self):
        raise NotImplementedError

    def _profiler_schedule(self):
        def schedule(a):
            if a < 2:
                return torch.profiler.ProfilerAction.WARMUP
            elif a < 4:
                return torch.profiler.ProfilerAction.RECORD
            elif a == 4:
                return torch.profiler.ProfilerAction.RECORD_AND_SAFE
            else:
                return torch.profiler.ProfilerAction.NONE

        return schedule

    def dump_val(self, i, batch_id, img_vars):
        d = {('i_%s' % k): v for k, v in img_vars.items()}
        self.save_dbg(d, 'val_%d_%d' % (i, batch_id))
        pass

    def save_dbg(self, d, name=None):
        if name is None:
            name = 'dbg_%d' % self._save_id
            self._save_id += 1
            pass
        savemat(self.dbg_dir / self.weight_save / f'{name}.mat', \
                {k: d[k].detach().to('cpu').numpy() for k in d.keys()}, do_compression=True)
        pass

    def save_result(self, d, id):
        name = 'result_%d' % id
        savemat(self.dbg_dir / self.weight_save / f'{name}.mat', \
                {k: v.detach().cpu().numpy() for k, v in d.items()}, do_compression=True)
        pass

    def validate(self):
        if len(self.dataset_fake_val) > 0:
            torch.cuda.empty_cache()
            loader_fake = torch.utils.data.DataLoader(self.dataset_fake_val, \
                                                      batch_size=1, shuffle=False, \
                                                      num_workers=self.num_loaders, pin_memory=True, drop_last=False,
                                                      collate_fn=self.collate_fn_val, worker_init_fn=seed_worker)

            self.network.eval()

            toggle_grad(self.network.generator, False)
            toggle_grad(self.network.discriminator, False)

            with torch.no_grad():
                for bi, batch_fake in enumerate(loader_fake):
                    # last item of batch_fake is just index

                    gen_vars = self._forward_generator_fake(batch_fake.to(self.device), i)
                    del batch_fake
                    self.dump_val(i, bi, gen_vars)
                    del gen_vars
                    pass
                pass

            self.network.train()

            toggle_grad(self.network.generator, False)
            toggle_grad(self.network.discriminator, True)

            del loader_fake
            # del gen_vars
            torch.cuda.empty_cache()
            pass
        else:
            self._log.warning('Validation set is empty - Skipping validation.')
        pass

    def train(self):
        """Train a network."""
        print(self.dataset_train)
        self.loader = torch.utils.data.DataLoader(self.dataset_train, \
                                                  batch_size=self.batch_size, shuffle=self.shuffle_train, \
                                                  num_workers=self.num_loaders, pin_memory=(not self.unpin),
                                                  drop_last=True, collate_fn=self.collate_fn_train,
                                                  worker_init_fn=seed_worker)  # seed_worker self.num_loaders

        if self.weight_init is not None:
            self._load_model()
            pass

        self.network.train()

        e = 0

        try:
            # with torch.profiler.profile(
            #	activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
            #	schedule=self._profiler_schedule(),
            #	on_trace_ready=torch.profiler.tensorboard_trace_handler(self._profile_path),
            #	record_shapes=False,
            #	profile_memory=self._profile_memory,
            #	with_stack=True) as self._profiler:
            # with torch.autograd.profiler.profile(enabled=self._profile, use_cuda=self._profile_gpu, profile_memory=self._profile_memory, with_stack=self._profile_stack) as prof:
            while not self._should_stop(e, self.i):

                for batch in self.loader:
                    if self._should_stop(e, self.i):
                        break

                    log_scalar, log_img = self._train_network(batch.to(self.device))
                    if self._log.isEnabledFor(logging.DEBUG):
                        self._log.debug(f'GPU memory allocated: {torch.cuda.memory_allocated(device=self.device)}')
                        pass

                    self._log_sync.update(self.i, log_scalar)

                    self._dump({**log_img}, force=self._log.isEnabledFor(logging.DEBUG))

                    del log_img
                    del batch

                    self._log_sync.print(self.i)

                    if self._should_save_iteration(self.i):
                        self._save_model(iterations=self.i)
                        pass
                    if self.i > 0 and self.i % self.val_interval == 0:
                        self.validate()
                        pass
                    pass

                e += 1

                if self._should_save_epoch(e):
                    self._save_model(epochs=e)
                    pass
                pass
            pass
        except:
            if not self.no_safe_exit:
                self._save_model(iterations=self.i, reason='break')
                pass

            self._log.error(f'Unexpected error: {sys.exc_info()[0]}')
            raise
        pass

    def test(self):
        """Test a network on a dataset."""
        self.loader_fake = torch.utils.data.DataLoader(self.dataset_fake_val, \
                                                       batch_size=1, shuffle=(self.shuffle_test), \
                                                       num_workers=self.num_loaders, pin_memory=True, drop_last=False,
                                                       collate_fn=self.collate_fn_val, worker_init_fn=seed_worker)

        if self.weight_init is not None:
            self._load_model()
            pass

        self.network.eval()
        with torch.no_grad():
            for bi, batch_fake in enumerate(self.loader_fake):
                print('batch %d' % bi)
                batch_fake = [f.to(self.device, non_blocking=True) for f in batch_fake[:-1]]
                self.save_result(self.evaluate_test(batch_fake, bi), bi)
                pass
            pass
        pass

    def infer(self):
        """Run network on single example."""

        if self.weight_init is not None:
            self._load_model()
            pass

        self.network.train()
        # with torch.no_grad():
        self.evaluate_infer()
        # pass
        pass

    @classmethod
    def add_arguments(cls, parser):
        # methods available at command line

        parser.add_argument('action', type=str, choices=cls.actions)
        parser.add_argument('config', type=Path, help='Path to config file.')
        parser.add_argument('-log', '--log', type=str, default='info', choices=_logstr2level.keys())
        parser.add_argument('--log_dir', type=Path, default='./log/', help='Directory for log files.')
        parser.add_argument('--carla_config', type=Path, default='', help='Directory for the Carla (plug in) settings.')
        parser.add_argument('--gpu', type=int, default=0, help='ID of GPU. Use -1 to run on CPU. Default: 0')
        parser.add_argument('--no_safe_exit', action='store_true', default=False,
                            help='Do not save model if anything breaks.')
        args = parser.parse_args()
        if(args.action == 'infer'):
            if (args.carla_config is None) or not os.path.isfile(args.carla_config):
                print('\033[91m'+"carla_config argument does not exist. Give a valid yaml carla config file path.")
                exit(1)
            else:
                global carla_config_path
                carla_config_path = args.carla_config
        pass

    def run(self):
        self.__getattribute__(self.action)()
        pass


if __name__ == '__main__':
    self.train()
