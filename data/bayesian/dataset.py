# 2. dataset
import torch
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import json
import os

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

class FSCBayesianDataset(BaseDataset):
    def __init__(
        self,
        img_dir,
        meta_file,
        transform_pre=None,
        image_augment=None,
        transform_fn=None,
    ):
        self.img_dir = img_dir
        self.meta_file = meta_file
        self.transform_pre = transform_pre
        self.image_augment = image_augment
        self.transform_fn = transform_fn

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

        # get boxes, h, w
        boxes = meta["boxes"][:3]

        # points
        points = meta["points"]
        
        # list int64 => numpy float32
        boxes = np.array(boxes, dtype=np.float32)
        points = np.array(points, dtype=np.float32)
        
        # transform
        if self.transform_pre:     # resize, hflip, vflip
            image, boxes, points = self.transform_pre(
                image, boxes, points
            )
        if self.image_augment:     # train data: image augment
            image = self.image_augment(image)
        if self.transform_fn:      # normalization, to tensor
            image = self.transform_fn(image)
        

        # target point map
        c, h, w = image.shape
        target = np.zeros((h,w))
        for (x, y) in points:
            x = max(0, min(w-1, int(x)))
            y = max(0, min(h-1, int(y)))
            target[y, x] = 1
        size = [height, width]
        target, boxes, points = torch.from_numpy(target), torch.from_numpy(boxes), torch.from_numpy(points)
        return image, target, boxes, points, size, img_name
