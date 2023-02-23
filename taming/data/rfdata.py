import os
import numpy as np
from torch.utils.data import Dataset

from taming.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex, RFDataPaths


class CustomBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        return example



class CustomTrain(CustomBase):
    def __init__(self, size, training_images_list_file):
        super().__init__()
        with open(training_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)


class CustomTest(CustomBase):
    def __init__(self, size, test_images_list_file):
        super().__init__()
        with open(test_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)



class RFDataUsImageTrain(Dataset):
    def __init__(self, size, training_images_list_file):
        super().__init__()
        with open(training_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        self.rf = RFDataPaths(paths=paths, size=size, random_crop=False)

        self._length = len(paths)

    def __len__(self):
        return self._length
    
    def __getitem__(self, index):
        example = {}
        example["image"] = self.data[index]["image"]
        example["coord"] = self.rf[index]["coord"]

        return example


class RFDataUsImageValidation(Dataset):
    def __init__(self, size, test_images_list_file):
        super().__init__()
        with open(test_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        self.rf = RFDataPaths(paths=paths, size=size, random_crop=False)

        self._length = len(paths)

    def __len__(self):
        return self._length
    
    def __getitem__(self, index):
        example = {}
        example["image"] = self.data[index]["image"]
        example["coord"] = self.rf[index]["coord"]

        return example
    

class RFDataTrain(CustomBase):
    def __init__(self, size, training_images_list_file):
        super().__init__()
        with open(training_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = RFDataPaths(paths=paths, size=size, random_crop=False)


class RFDataValidation(CustomBase):
    def __init__(self, size, test_images_list_file):
        super().__init__()
        with open(test_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = RFDataPaths(paths=paths, size=size, random_crop=False)



if __name__ == '__main__':
    train = CustomTrain(size=256, training_images_list_file='../../some/us_image_train.txt')


