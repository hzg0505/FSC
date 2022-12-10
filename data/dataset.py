# 2. dataset
import torch
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import json
import os

# nearest_dis = np.clip(0.8*keypoints[:, 2], 4.0, 40.0)
def find_dis(point):
#     point = point.numpy()
    square = np.sum(point*point, axis=1)
    dis = np.sqrt(np.maximum(square[:, None] - 2*np.matmul(point, point.T) + square[None, :], 0.0))
    dis = np.mean(np.partition(dis, 3, axis=1)[:, 1:4], axis=1, keepdims=True)
    return dis

def gen_density(points, size, boxes):
    # points = points.numpy()
    h, w = size
    dis = find_dis(points)

    ratio_ = 0.0
    for box in boxes:
        y1, x1, y2, x2 = box
        h_box, w_box = y2-y1, x2-x1
        ratio = h_box/w_box
        ratio_ += ratio
    ratio_ /= boxes.shape[0]  # 平均高宽比
    # ratio_ = ratio_.numpy()
    G_map = np.zeros(size)
    for point, d in zip(points, dis):
        nearest_dis = np.clip(0.8*d, 4.0, 40.0)
        density_map = np.zeros(size)
        if ratio_ < 1.0: # 较扁的图
            ksize = [int(nearest_dis/ratio_),int(nearest_dis)]  # x,y. w,h
        else: # 较高的图
            ksize = [int(nearest_dis),int(nearest_dis*ratio_)]  # x,y. w,h
        if ksize[0]%2==0:
            ksize[0]+=1
        if ksize[1]%2==0:
            ksize[1]+=1
        x, y = point
        x = max(0, min(w-1, int(x)))
        y = max(0, min(h-1, int(y)))
        density_map[y, x] += 1 
#         ksize = [115, 115]
        sigma = float(ksize[0]*ksize[1]**0.5*0.05)
        density_map = cv2.GaussianBlur(density_map, ksize=ksize, sigmaX=sigma, sigmaY=sigma*ratio_)
        G_map += density_map
    return G_map

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
        
        
        density = gen_density(points, size=[512, 512], boxes=boxes)
        size = [height, width]
        # target point map
        # c, h, w = image.shape
        # target = np.zeros((h,w))
        # for (x, y) in points:
        #     x = max(0, min(w-1, int(x)))
        #     y = max(0, min(h-1, int(y)))
        #     target[y, x] = 1

        #

        density = np.array(density, dtype=np.float32)
        density, boxes, points = torch.from_numpy(density), torch.from_numpy(boxes), torch.from_numpy(points)
        return image, density, boxes, points, size, img_name
