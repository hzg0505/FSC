from data.bayesian.augment import *
from data.bayesian.dataset import FSCBayesianDataset


from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from torch.utils.data import DataLoader
# from torch.utils.data.dataloader import default_collate

# 1. 水平翻转，竖直翻转，resize
class TrainTransform(object):
    def __init__(self, size=(512, 512)):
        self.size = size   # h, w
        self.augment = Compose_1([
            Resize(self.size),    # resize ：image; boxes、points 相对位置不用变 
            RandomHFilp(),        # 水平翻转
            RandomVFilp(),        # 竖直翻转
        ])
    def __call__(self, img, boxes, points):
        return self.augment(img, boxes, points)

class ValTransform(object):
    def __init__(self, size=(512, 512)):
        self.size = size   # h, w
        self.augment = Compose_1([
            Resize(self.size),    # resize ：image; boxes、points 相对位置不用变 
        ])
    def __call__(self, img, boxes, points):
        return self.augment(img, boxes, points)

# 2. 图像光学增强
class ImageAug(object):
    def __init__(self):
        self.augment = Compose([
            ConvertFromInts(),    # 将 uint64 -> float32
            PhotometricDistort(),     # 光学变换
            # Normalization(self.mean, self.std),   # /255 -mean/std
            # ToTensor(), # 训练，img:c,h,w, 所有数据变为 tensor 格式 
        ])
    def __call__(self, img):
        return self.augment(img)

# 3.公共部分
class Transform_fn(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std
        self.augment = Compose([
            ConvertFromInts(),    # 将 uint64 -> float32
            Normalization(self.mean, self.std),   # /255 -mean/std
            ToTensor(), # 训练，img:c,h,w, 所有数据变为 tensor 格式 
        ])
    def __call__(self, img):
        return self.augment(img)


# 数据加 batch
def train_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    targets = torch.stack(transposed_batch[1], 0)
    boxes = torch.stack(transposed_batch[2], 0) # transposed_batch[2]  
    points = transposed_batch[3]  # the number of points is not fixed, keep it as a list of tensor
    st_sizes = torch.FloatTensor(transposed_batch[4])
    file_name = transposed_batch[5]
    return images, targets, boxes, points, st_sizes, file_name 

def data_loader(cfg, phase, distributed):
    img_dir = cfg["img_dir"]
    size = cfg["size"]
    mean = cfg["mean"]
    std = cfg["std"]

    if phase=='train':
        dataset = FSCBayesianDataset(
            img_dir=img_dir,
            meta_file=cfg["train"]["meta_file"],
            transform_pre=TrainTransform(size=size),
            image_augment=ImageAug(),
            transform_fn=Transform_fn(std=std, mean=mean),
        )
    elif phase=='val':
        dataset = FSCBayesianDataset(
            img_dir=img_dir,
            meta_file=cfg["val"]["meta_file"],
            transform_pre=ValTransform(size=size),
            image_augment=None,
            transform_fn=Transform_fn(std=std, mean=mean),
        )
    elif phase=='test':
        dataset = FSCBayesianDataset(
            img_dir=img_dir,
            meta_file=cfg["test"]["meta_file"],
            transform_pre=ValTransform(size=size),
            image_augment=None,
            transform_fn=Transform_fn(std=std, mean=mean),
        )
    else:
        raise ValueError("phase must among [train, val, test]!")
    
    if distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = RandomSampler(dataset)
    
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
    
    return train_loader, val_loader, test_loader