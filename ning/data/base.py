import bisect
import numpy as np
import albumentations
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset
import re
import os
from scipy.signal import hilbert
import scipy.io as scio
import numpy as np


class ConcatDatasetWithIndex(ConcatDataset):
    """Modified from original pytorch code to return dataset idx"""
    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx], dataset_idx


class ImagePaths(Dataset):
    def __init__(self, paths, size=None, random_crop=False, labels=None):
        self.size = size
        self.random_crop = random_crop

        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self._length = len(paths)

        if self.size is not None and self.size > 0:
            self.rescaler = albumentations.SmallestMaxSize(max_size = self.size)
            if not self.random_crop:
                self.cropper = albumentations.CenterCrop(height=self.size,width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size,width=self.size)
            self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])
        else:
            self.preprocessor = lambda **kwargs: kwargs

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image

    def __getitem__(self, i):
        example = dict()
        example["image"] = self.preprocess_image(self.labels["file_path_"][i])
        for k in self.labels:
            example[k] = self.labels[k][i]
        return example


class RFDataPaths(Dataset):
    def __init__(self, paths, size=None, random_crop=False, labels=None):
        self.size = size
        self.random_crop = random_crop

        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self._length = len(paths)

    def __len__(self):
        return self._length

    def preprocess_image(self, rf_path):
        path = re.sub("us_image", "rf_data", rf_path)
        path = re.sub(".png", "", path)
        files = os.listdir(path)
        files.sort(key=lambda x: int(x[5:-4]))

        tstarts = []
        rf_envs = []
        max_len = 0
        for file in files:
            mat = scio.loadmat(os.path.join(path, file))
            tstart = mat['tstart']
            rf_data = mat['rf_data']
            rf_env = np.abs(hilbert(rf_data, axis=0))

            tstarts.append(tstart)
            rf_envs.append(rf_env)

            if rf_env.shape[0] > max_len:
                max_len = rf_env.shape[0]


        for i, rf_env in enumerate(rf_envs):
            pad_width = max_len - rf_env.shape[0]
            rf_envs[i] = np.pad(rf_env, ((0, pad_width),(0, 0)), 'constant', constant_values=0)

        env = np.concatenate(rf_envs, axis=1)

        D = 10
        dB_Range=50
        env = env - np.min(env)
        env = env / np.max(env)
        env = env[1:max_len:D, :] + 0.00001
        log_env = 20 * np.log10(env)
        log_env=255/dB_Range*(log_env+dB_Range)
        [N, M] = log_env.shape
        D = int(np.floor(N/1024))
        env_disp = 255 * log_env[1:N:D, :] / np.max(log_env)
        env_disp = env_disp.astype(np.uint8)
        img = Image.fromarray(env_disp)
        img = img.resize((self.size, self.size))
        img = img.rotate(180)
        if not img.mode == "RGB":
            img = img.convert("RGB")
        img = np.array(img)

        img = (img/127.5 - 1.0).astype(np.float32)

        return img

    def __getitem__(self, i):
        example = dict()
        example["coord"] = self.preprocess_image(self.labels["file_path_"][i])
        for k in self.labels:
            example[k] = self.labels[k][i]
        return example

class NumpyPaths(ImagePaths):
    def preprocess_image(self, image_path):
        image = np.load(image_path).squeeze(0)  # 3 x 1024 x 1024
        image = np.transpose(image, (1,2,0))
        image = Image.fromarray(image, mode="RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image
