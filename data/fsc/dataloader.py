# 3. dataloader
from data.fsc.dataset import FSCDataset
from data.fsc.augment import *

from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

#######################################
# 定义 公共tranfor
'''
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
class FSCAugmentations(object):
    def __init__(self, size=(512, 512), mean=IMAGENET_MEAN, std=IMAGENET_STD):
        self.size = size
        self.mean = mean
        self.std = std
        self.augment = Compose([
            RandomHFilp(),      # 水平翻转
            RandomVFilp(),      # 竖直翻转
            PhotometricDistort(),     # 光学变换
            ConvertFromInts(),    # 将 uint64 -> float32
            RandomBrightness(delta=20),  # 亮度     +   0，255
            RandomContrast(0.8, 1.2),    # 对比度   *   0， 1
            RandomLightingNoise(),       # 随机转换通道
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='RGB'),
                        
            Normalization(self.mean, self.std),   # /255.0 然后规范化
            ToTensor(),  # 训练，img:c,h,w, 所有数据变为 tensor 格式 
        ])

    def __call__(self, img, density, boxes, points):
        return self.augment(img, density, boxes, points)
size = (512,512) # h, w
'''

## 公共部分
class FSCTransforms(object):
    def __init__(self, size=(512, 512), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.size = size
        self.mean = mean
        self.std = std
        self.augment = Compose([
            Resize(self.size),    # resize ：image、density，boxes， points 相对位置不用变 
            ConvertFromInts(),    # 将 uint64 -> float32
            Normalization(self.mean, self.std),   # /255 -mean/std
            ToTensor(), # 训练，img:c,h,w, 所有数据变为 tensor 格式 
        ])

    def __call__(self, img, density, boxes, points):
        return self.augment(img, density, boxes, points)

## 训练集
class FSCTrainTrans(object):
    def __init__(self):
        self.augment = Compose([
            RandomHFilp(),      # 水平翻转
            # RandomVFilp(),      # 竖直翻转
            ConvertFromInts(),
            PhotometricDistort(),     # 光学变换
        ])

    def __call__(self, img, density, boxes, points):
        return self.augment(img, density, boxes, points)

## 测试集，验证集:除公共部分无

# transform_fn = FSCTransforms()
# train_trans = FSCTrainTrans()
#######################################
# 数据加 batch
def train_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    densitys = torch.stack(transposed_batch[1], 0)
    boxes = torch.stack(transposed_batch[2], 0) # transposed_batch[2]  
    points = transposed_batch[3]  # the number of points is not fixed, keep it as a list of tensor
    st_sizes = torch.FloatTensor(transposed_batch[4])
    file_name = transposed_batch[5]
    return images, densitys, boxes, points, st_sizes, file_name 


def data_loader(cfg, phase, distributed):
    img_dir = cfg["img_dir"]
    density_dir = cfg["density_dir"]
    shot = cfg["shot"]
    size = cfg["size"]
    mean = cfg["mean"]
    std = cfg["std"]

    if phase=='train':
        dataset = FSCDataset(
            img_dir,
            density_dir,
            cfg["train"]["meta_file"],
            shot,
            transform_fn=FSCTransforms(size=size,mean=mean,std=std),
            transform_train=FSCTrainTrans()
        )
    elif phase=='val':
        dataset = FSCDataset(
            img_dir,
            density_dir,
            cfg["val"]["meta_file"],
            shot,
            transform_fn=FSCTransforms(),
            transform_train=None
        )
    elif phase=='test':
        dataset = FSCDataset(
            img_dir,
            density_dir,
            cfg["test"]["meta_file"],
            shot,
            transform_fn=FSCTransforms(),
            transform_train=None
        )
    else:
        raise ValueError("phase must among [train, val, test]!")
    
    if distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = RandomSampler(dataset)

    # data_loader = DataLoader(
    #     dataset,
    #     batch_size= 1, #cfg["batch_size"],
    #     num_workers=cfg["workers"],
    #     pin_memory=False,
    #     sampler=sampler,
    # )
    
    data_loader = DataLoader(
                dataset,
                collate_fn=train_collate, #if phase=='train' else default_collate),
                batch_size=(cfg["batch_size"]),
                num_workers=cfg["workers"],
                pin_memory=(True if phase=='train' else False),
                sampler=sampler,
    )                      

    return data_loader

def build_dataloader(cfg_dataset, distributed=False):
    train_loader = None
    if cfg_dataset.get("train", None):
        train_loader = data_loader(cfg_dataset, phase="train", distributed=distributed)

    val_loader = None
    if cfg_dataset.get("val", None):
        val_loader = data_loader(cfg_dataset, phase="val", distributed=distributed)

    test_loader = None
    if cfg_dataset.get("test", None):
        test_loader = data_loader(cfg_dataset, phase="test", distributed=distributed)

#     if rank == 0:
#         print("build dataset done")

    return train_loader, val_loader, test_loader
