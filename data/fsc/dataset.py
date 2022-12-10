# 2. dataset
import torch
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import json
import os
# from augment import *

class BaseDataset(Dataset):
    """
    A dataset should implement
        1. __len__ to get size of the dataset, Required
        2. __getitem__ to get a single data, Required

    """
    def __init__(self):
        super(BaseDataset, self).__init__()

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

class FSCDataset(BaseDataset):
    def __init__(
        self,
        img_dir,
        density_dir,
        meta_file,
        shot,
        transform_fn=None,
        transform_train=None
    ):
        self.img_dir = img_dir
        self.density_dir = density_dir
        self.meta_file = meta_file
        self.shot = shot
        self.transform_fn = transform_fn
        self.transform_train = transform_train

        # construct metas
        if isinstance(meta_file, str):
            meta_file = [meta_file]
        self.metas = []
        for _meta_file in meta_file:
            with open(_meta_file, "r+") as f_r:
                for line in f_r:
                    meta = json.loads(line)
                    self.metas.append(meta)

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, index):
        meta = self.metas[index]
        # read img
        img_name = meta["filename"]
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        # read density
        density_name = meta["density"]
        density_path = os.path.join(self.density_dir, density_name)
        density = np.load(density_path)
        # get boxes, h, w
        boxes = meta["boxes"]
        if self.shot:
            boxes = boxes[: self.shot]
        # points
        points = meta["points"]
        
        # list int64 => numpy float32
        boxes = np.array(boxes, dtype=np.float32)
        points = np.array(points, dtype=np.float32)
        
        # transform
        if self.transform_train:
            image, density, boxes, points = self.transform_train(
                image, density, boxes, points
            )
        if self.transform_fn:
            image, density, boxes, points = self.transform_fn(
                image, density, boxes, points
            )

        size = [height, width]
        # density, boxes, points = torch.from_numpy(density), torch.from_numpy(boxes), torch.from_numpy(points)
        return image, density, boxes, points, size, img_name
        # return 
        # return {
        #     "filename": img_name,
        #     "height": height,
        #     "width": width,
        #     "image": image,
        #     "density": density,
        #     "boxes": boxes,
        #     "points":points,
        # }