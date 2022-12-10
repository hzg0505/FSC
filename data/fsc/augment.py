# 1. augment
import cv2
import numpy as np
from torchvision import transforms
from torchvision.transforms import functional as F
from numpy import random
import torch

################################################
class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, density=None, boxes=None, points=None):
        for t in self.transforms:
            img, density, boxes, points = t(img, density, boxes, points)
        return img, density, boxes, points
# 1. Transform    
class ToAbsoluteCoords(object):
    def __call__(self, image, density=None, boxes=None, points=None):
        height, width, channels = image.shape
        if boxes is not None:
            boxes[:, 0] *= height
            boxes[:, 2] *= height
            boxes[:, 1] *= width
            boxes[:, 3] *= width
        if points is not None:
            points[:, 0] *= width
            points[:, 1] *= height
        return image, density, boxes, points

class ToPercentCoords(object):
    def __call__(self, image, density=None, boxes=None, points=None):
        height, width, channels = image.shape
        if boxes is not None:
            boxes[:, 0] = boxes[:, 0]/height
            boxes[:, 2] = boxes[:, 2]/height
            boxes[:, 1] = boxes[:, 1]/width
            boxes[:, 3] = boxes[:, 3]/width
        if points is not None:
            points[:, 0] = points[:, 0]/width
            points[:, 1] = points[:, 1]/height
        return image, density, boxes, points

class RandomHFilp(object):
    def __call__(self, image, density=None, boxes=None, points=None):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            density = density[:, ::-1]
            if boxes is not None:
                boxes = boxes.copy()
#                 boxes[:, 0::2] = width - boxes[:, 2::-2]   # x1, y1, x2, y2  => w-x2, y1, w-x1, y2
                boxes[:, 1::2] = width - boxes[:, 3::-2]   # y1, x1, y2, x2  => y1, w-x2, y2, w-x1
            if points is not None:
                points[:, 0] = width - points[:, 0]
        return image, density, boxes, points    
    
class RandomVFilp(object):
    def __call__(self, image, density=None, boxes=None, points=None):
        height, _, _ = image.shape
        if random.randint(2):
            image = image[::-1, :]
            density = density[::-1, :]
            if boxes is not None:
                boxes = boxes.copy()
#                 boxes[:, 1::2] = height - boxes[:, 1::-2]   # x1, y1, x2, y2  => x1, h-y2, x2, h-y1
                boxes[:, 0::2] = height - boxes[:, 2::-2]  # y1, x1, y2, x2  => h-y2, x1, h-y1, x2
            if points is not None:
                points[:, 1] = height - points[:, 1]
        return image, density, boxes, points        

class Resize(object):
    def __init__(self, size=(512, 512)):  # h, w
        self.size = size
        self.toprecent = ToPercentCoords()
        self.toabsolute = ToAbsoluteCoords()

    def __call__(self, image, density=None, boxes=None, points=None):
        image, density, boxes, points = self.toprecent(image, density, boxes, points)
        cnt_orig = density.sum()
        density = cv2.resize(density, (self.size[1], self.size[0]))
        if not cnt_orig == 0:
            cnt_rsz = density.sum()
            density = density * (cnt_orig / cnt_rsz)
#         density = Image.fromarray(density)
        image = cv2.resize(image, (self.size[1], self.size[0]))  # w, h
        image, density, boxes, points = self.toabsolute(image, density, boxes, points)
        return image, density, boxes, points

class Normalization(object):
    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
    def __call__(self, image, density=None, boxes=None, points=None):
        image = image.astype(np.float32)
        image /= 255.0
        image -= self.mean
        image /= self.std
        return image.astype(np.float32), density, boxes, points 

class ToTensor(object):
    def __call__(self, cvimage, density=None, bboxs=None, points=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), torch.from_numpy(density), torch.from_numpy(bboxs), torch.from_numpy(points)
    
class ToCV2Image(object):
    def __call__(self, tensor, density=None, boxes=None, points=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), density, boxes, points    
    
# Random Color Jittering
class ConvertFromInts(object):  # 将图像从整形转化为浮点型
    def __call__(self, image, density=None, boxes=None, points=None):
        return image.astype(np.float32), density, boxes, points 

class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):   # 1-0.5 ~ 1+0.5
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, density=None, boxes=None, points=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, density, boxes, points
    
class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, density=None, boxes=None, points=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, density, boxes, points

class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image, density=None, boxes=None, points=None):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image, density, boxes, points    

class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, density=None, boxes=None, points=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle_fn = SwapChannels(swap)  # shuffle channels
            image, density, boxes, points = shuffle_fn(image, density, boxes, points)
        return image, density, boxes, points
    
class ConvertColor(object):
    def __init__(self, current='RGB', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, density=None, boxes=None, points=None):
        if self.current == 'RGB' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif self.current == 'HSV' and self.transform == 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        else:
            raise NotImplementedError
        return image, density, boxes, points

class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, density=None, boxes=None, points=None):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)
        return image, density, boxes, points

class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, density=None, boxes=None, points=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, density, boxes, points

class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),  
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='RGB'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, density=None, boxes=None, points=None):
        im = image.copy()
        im, density, boxes, points = self.rand_brightness(im, density, boxes, points)   # 随机亮度
        if random.randint(2):
            distort = Compose(self.pd[:-1])   # 后进行随机对比度
        else:
            distort = Compose(self.pd[1:])    # 先进行随机对比度
        im, density, boxes, points = distort(im, density, boxes, points)
        im, density, boxes, points = self.rand_light_noise(im, density, boxes, points)
        return im, density, boxes, points
    
################################################
# 交集
def intersect_t(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]

# 交并比
def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect_t(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]

class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, points=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, points