import logging
from pathlib import Path

import imageio
import numpy as np
from skimage.transform import resize
import scipy.io as sio
import torch
import os
from .batch_types import EPEBatch
from .synthetic import SyntheticDataset
from .utils import mat2tensor, normalize_dim
from PIL import Image
import time
import threading
import concurrent.futures

def center(x, m, s):
	x[0,:,:] = (x[0,:,:] - m[0]) / s[0]
	x[1,:,:] = (x[1,:,:] - m[1]) / s[1]
	x[2,:,:] = (x[2,:,:] - m[2]) / s[2]
	return x

def material_from_gt_label(gt_labelmap):
	if gt_labelmap.shape[-1] == 4:
		gt_labelmap = gt_labelmap[:, :, :3]
	gt_labelmap = np.dot(gt_labelmap, [1, 1, 1])
	h,w = gt_labelmap.shape
	shader_map = np.zeros((h, w, 12), dtype=np.float32)
	s = time.time()
	shader_map[:,:,0] = (gt_labelmap == 11).astype(np.float32) # sky
	shader_map[:,:,1] = (np.isin(gt_labelmap, [1,2,24,25,27])).astype(np.float32) # road / static / sidewalk
	shader_map[:,:,2] = (np.isin(gt_labelmap, [14,15,16,17,18,19])).astype(np.float32) # vehicle
	shader_map[:,:,3] = (gt_labelmap == 10).astype(np.float32) # terrain
	shader_map[:,:,4] = (gt_labelmap == 9).astype(np.float32) # vegetation
	shader_map[:,:,5] = (np.isin(gt_labelmap, [12,13])).astype(np.float32) # person
	shader_map[:,:,6] = (np.isin(gt_labelmap, [6])).astype(np.float32) # infrastructure
	shader_map[:,:,7] = (gt_labelmap == 7) # traffic light
	shader_map[:,:,8] = (gt_labelmap == 8) # traffic sign
	shader_map[:,:,9] = (np.isin(gt_labelmap, [21,23])).astype(np.float32) # ego vehicle
	shader_map[:,:,10] = (np.isin(gt_labelmap, [20,3,4,5,26,28])).astype(np.float32) # building
	shader_map[:,:,11] = (np.isin(gt_labelmap, [0,22])).astype(np.float32) # unlabeled
	print(time.time() - s)

	#from matplotlib import pyplot as plt
	#custom_cmap = plt.cm.colors.ListedColormap(['black', 'white'])
	#plt.imshow(shader_map[:, :, 10],cmap=custom_cmap)
	#plt.show()

	return shader_map

def get_gbuffers(gbuffer_folder):
	gbuff_list = []
	for root, dirs, files in os.walk(os.path.abspath(gbuffer_folder)):
		for file in files:
			img = Image.open(os.path.join(root, file))
			img = np.array(img)
			#if 'CustomStencil' in file or 'GBufferSSAO' in file or 'SceneDepth' in file:
				#img = img[:, :, 0]
				#gbuff_list.append(img[:, :, np.newaxis])
				#continue
			if img.shape[-1] == 4:
				img = img[:, :, :3]
				#img = img[:, :, ::-1]

			gbuff_list.append(img)

	stacked_image = np.concatenate(gbuff_list, axis=2)
	#print(stacked_image.shape)
	#from matplotlib import pyplot as plt
	#for i in range(27):
		#plt.imshow(stacked_image[:, :, i])
		#print(np.where((stacked_image[:, :, i] == stacked_image[:, :, i+1]) == False))
		#plt.show()

	return stacked_image

def mean_std_scaling(image):
    mean = np.mean(image, axis=(0, 1))
    std_dev = np.std(image, axis=(0, 1))
    epsilon = 1e-8
    std_dev += epsilon  # Add epsilon to prevent division by zero
    standardized_image = (image - mean) / std_dev
    #print(image.shape)
    #print(np.max(standardized_image))
    #print(np.min(standardized_image))
    return standardized_image


class PfDDataset(SyntheticDataset):
	def __init__(self, paths, transform=None, gbuffers='all'):
		"""


		paths -- list of tuples with (img_path, robust_label_path, gbuffer_path, gt_label_path)
		"""

		super(PfDDataset, self).__init__('GTA')

		assert gbuffers in ['all', 'img', 'no_light', 'geometry', 'fake']

		self.transform = transform
		self.gbuffers  = gbuffers #gbuffers
		# self.shader    = class_type

		self._paths    = paths
		self._path2id  = {p[0].stem:i for i,p in enumerate(self._paths)}
		if self._log.isEnabledFor(logging.DEBUG):
			self._log.debug(f'Mapping paths to dataset IDs (showing first 30 entries):')
			for i,(k,v) in zip(range(30),self._path2id.items()):
				self._log.debug(f'path2id[{k}] = {v}')
				pass
			pass


		try:
			data = np.load(Path(__file__).parent / 'pfd_stats.npz')
			# self._img_mean  = data['i_m']
			# self._img_std   = data['i_s']
			self._gbuf_mean = data['g_m']
			self._gbuf_std  = data['g_s']
			self._log.info(f'Loaded dataset stats.')
		except:
			# self._img_mean  = None
			# self._img_std   = None
			self._gbuf_mean = None
			self._gbuf_std  = None
			pass

		self._log.info(f'Found {len(self._paths)} samples.')
		pass

	@property
	def num_gbuffer_channels(self):
		""" Number of image channels the provided G-buffers contain."""
		return {'fake':32, 'all':18, 'img':0, 'no_light':18, 'geometry':14}[self.gbuffers]


	@property
	def num_classes(self):
		""" Number of classes in the semantic segmentation maps."""
		return {'fake':12, 'all':12, 'img':0, 'no_light':0, 'geometry':12}[self.gbuffers]


	@property
	def cls2gbuf(self):
		if self.gbuffers == 'all':
			# all: just handle sky class differently
			return {\
				#8 is specular
				#0:lambda g:g[:,[0,1],:,:]
				0: lambda g: g[:, [0, 1, 2], :, :],
				1: lambda g: g[:, [0, 1, 2,3,4,5,6,7,8,9,10,11,12,16,17], :, :],
				3: lambda g: g[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 17], :, :],
				5: lambda g: g[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 17], :, :],
				6: lambda g: g[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 17], :, :],
				7: lambda g: g[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 17], :, :],
				8: lambda g: g[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 17], :, :],
				9: lambda g: g[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 17], :, :],
				10: lambda g: g[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 17], :, :],
				11: lambda g: g[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 17], :, :]
			}
		else:
			return {}
		pass


	def get_id(self, img_filename):
		return self._path2id.get(Path(img_filename).stem)


	def __getitem__(self, index):

		index  = index % self.__len__()
		img_path, robust_label_path, gbuffer_path, gt_label_path = self._paths[index]

		if not gbuffer_path.exists():
			self._log.error(f'Gbuffers at {gbuffer_path} do not exist.')
			raise FileNotFoundError
			pass

		data = np.load(gbuffer_path)

		if self.gbuffers == 'fake':
			img       = mat2tensor(imageio.imread(img_path).astype(np.float32) / 255.0)

			if img.size(0) == 4:  # Check if the tensor has 4 channels (RGBA format)
				img = img[:3, :, :]

			gbuffers  = mat2tensor(data['data'].astype(np.float32))

			gt_labels = mat2tensor(np.load(gt_label_path)['arr_0'].astype(np.float32))
			#gbuffers  = mat2tensor(data['data'].astype(np.float32))

			#gt_labels = material_from_gt_label(imageio.imread(gt_label_path))
			#gt_labels = imageio.imread(gt_label_path)
			#if gt_labels.shape[0] != img.shape[-2] or gt_labels.shape[1] != img.shape[-1]:
				#gt_labels = resize(gt_labels, (img.shape[-2], img.shape[-1]), anti_aliasing=True, mode='constant')
			#gt_labels = mat2tensor(gt_labels)
			pass
		else:
			img       = mat2tensor(imageio.imread(img_path).astype(np.float32) / 255.0)

			if img.size(0) == 4:  # Check if the tensor has 4 channels (RGBA format)
				img = img[:3, :, :]

			#gbuffers = mat2tensor(data['arr_0'][:, :, 3:16].astype(np.float32))
			gbuffers = mat2tensor(data['arr_0'].astype(np.float32))
			gt_labels = mat2tensor(np.load(gt_label_path)['arr_0'].astype(np.float32))


			#img       = mat2tensor(data['img'].astype(np.float32) / 255.0)
			#gbuffers  = mat2tensor(data['gbuffers'].astype(np.float32))
			#gt_labels = mat2tensor(data['shader'].astype(np.float32))
			pass

		if torch.max(gt_labels) > 128:
			gt_labels = gt_labels / 255.0
			pass

		if self._gbuf_mean is not None:
			gbuffers = center(gbuffers, self._gbuf_mean, self._gbuf_std)
			pass

		if not robust_label_path.exists():
			self._log.error(f'Robust labels at {robust_label_path} do not exist.')
			raise FileNotFoundError
			pass

		robust_labels = imageio.imread(robust_label_path)
		robust_labels = torch.LongTensor(robust_labels[:,:]).unsqueeze(0)

		return EPEBatch(img, gbuffers=gbuffers, gt_labels=gt_labels, robust_labels=robust_labels, path=img_path, coords=None)


	def __len__(self):
		return len(self._paths)
