try:
    import tensorrt as trt
except:
    print("Failed to import TensorRT... Check your TensorRT installation...")
from cuda import cudart
import imageio
import numpy as np
import os


class ImageBatcher:
    """
    Creates batches of pre-processed images.
    """

    def __init__(self, input, shape, dtype):
        try:
            dataset = open(input, 'r')
            data = dataset.readlines()
        except:
            print("Error loading calibration dataset txt file... Check calibration_dataset parameter inside carla_config file...")
            exit(1)


        self.img = []
        self.gb = []
        self.gt = []

        for line in data:
            line = line.rstrip('\n')
            line = line.split(",")
            self.img.append(line[0])
            self.gb.append(line[2])
            self.gt.append(line[3])

        self.total_data = len(self.img)
        self.dtype = dtype
        self.img_shape = [1,3,shape[0],shape[1]]
        self.gb_shape = [1, 18, shape[0], shape[1]]
        self.gt_shape = [1, 12, shape[0], shape[1]]
        self.batch_size = 1


    def get_batch(self):
        for i in range(len(self.img)):
            data = np.load(self.gb[i])
            img = imageio.imread(self.img[i]).astype(np.float32) / 255.0
            img = img[:, :, :3]

            gbuffers = data['arr_0'].astype(np.float32)
            gt_labels = np.load(self.gt[i])['arr_0'].astype(np.float32)

            img = img.transpose((2,0,1))
            gbuffers = gbuffers.transpose((2, 0, 1))
            gt_labels = gt_labels.transpose((2, 0, 1))

            print("["+str(i)+"/"+str(self.total_data)+"]...")

            yield np.expand_dims(img, axis=0),np.expand_dims(gbuffers, axis=0),np.expand_dims(gt_labels, axis=0)

class EngineCalibrator(trt.IInt8EntropyCalibrator2):
    """
    Implements the INT8 MinMax Calibrator.
    """

    def __init__(self, cache_file):
        """
        :param cache_file: The location of the cache file.
        """
        super().__init__()
        self.cache_file = cache_file
        self.image_batcher = None
        self.batch_allocation_img = None
        self.batch_allocation_gb = None
        self.batch_allocation_gt = None
        self.batch_generator = None

    def set_image_batcher(self, image_batcher: ImageBatcher):
        import common
        """
        Define the image batcher to use, if any. If using only the cache file, an image batcher doesn't need
        to be defined.
        :param image_batcher: The ImageBatcher object
        """
        self.image_batcher = image_batcher
        self.size = int(np.dtype(self.image_batcher.dtype).itemsize * np.prod(self.image_batcher.img_shape))
        self.batch_allocation_img = common.cuda_call(cudart.cudaMalloc(self.size))
        self.size = int(np.dtype(self.image_batcher.dtype).itemsize * np.prod(self.image_batcher.gb_shape))
        self.batch_allocation_gb = common.cuda_call(cudart.cudaMalloc(self.size))
        self.size = int(np.dtype(self.image_batcher.dtype).itemsize * np.prod(self.image_batcher.gt_shape))
        self.batch_allocation_gt = common.cuda_call(cudart.cudaMalloc(self.size))
        self.batch_generator = self.image_batcher.get_batch()

    def get_batch_size(self):
        """
        Overrides from trt.IInt8MinMaxCalibrator.
        Get the batch size to use for calibration.
        :return: Batch size.
        """
        if self.image_batcher:
            return self.image_batcher.batch_size
        return 1

    def get_batch(self, names):
        import common
        """
        Overrides from trt.IInt8MinMaxCalibrator.
        Get the next batch to use for calibration, as a list of device memory pointers.
        :param names: The names of the inputs, if useful to define the order of inputs.
        :return: A list of int-casted memory pointers.
        """
        if not self.image_batcher:
            return None
        try:
            img,gb,gt = next(self.batch_generator)
            common.memcpy_host_to_device(self.batch_allocation_img, np.ascontiguousarray(img))
            common.memcpy_host_to_device(self.batch_allocation_gb, np.ascontiguousarray(gb))
            common.memcpy_host_to_device(self.batch_allocation_gt, np.ascontiguousarray(gt))
            return [int(self.batch_allocation_img),int(self.batch_allocation_gb),int(self.batch_allocation_gt)]
        except StopIteration:
            return None

    def read_calibration_cache(self):
        """
        Overrides from trt.IInt8MinMaxCalibrator.
        Read the calibration cache file stored on disk, if it exists.
        :return: The contents of the cache file, if any.
        """
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        """
        Overrides from trt.IInt8MinMaxCalibrator.
        Store the calibration cache to a file on disk.
        :param cache: The contents of the calibration cache to store.
        """
        if self.cache_file is None:
            return
        with open(self.cache_file, "wb") as f:
            f.write(cache)