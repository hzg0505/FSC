import torch.nn.functional as F
import torch
import numpy as np
import copy
from torch import nn

def exemplar_scale_augment(image, boxes, exemplar_scales, min_stride=16): 
#     img = image.copy
    # 同一个尺度下 有多个 box  exemplar_list:[[[],[],[]], [[],[],[]], [[],[],[]]]
    image_scale_list = []
    boxes_scale_list = []
    exemplar_scale_list = []
    b, c, h, w = image.shape
    for scale in exemplar_scales:
        h_rsz = int(h * scale) // min_stride * min_stride
        w_rsz = int(w * scale) // min_stride * min_stride
        image_scale = F.interpolate(image, size=(w_rsz, h_rsz), mode="bilinear")
        scale_h = h_rsz / h
        scale_w = w_rsz / w
    
        boxes_scale = copy.deepcopy(boxes)
        boxes_scale[:, 0] *= scale_h
        boxes_scale[:, 1] *= scale_w
        boxes_scale[:, 2] *= scale_h
        boxes_scale[:, 3] *= scale_w
        
        boxes_scale[:, :2] = torch.floor(boxes_scale[:, :2])  # y_tl, x_tl: floor
        boxes_scale[:, 2:] = torch.ceil(boxes_scale[:, 2:])  # y_br, x_br: ceil
        boxes_scale[:, :2] = torch.clamp_min(boxes_scale[:, :2], 0)
        boxes_scale[:, 2] = torch.clamp_max(boxes_scale[:, 2], h_rsz)
        boxes_scale[:, 3] = torch.clamp_max(boxes_scale[:, 3], w_rsz)
        
        image_scale_list.append(image_scale) # n, img
        boxes_scale_list.append(boxes_scale) # n, boxes
        exemplar_scales = []
        for box in boxes_scale:
            y1,x1,y2,x2 = box
            y1, x1, y2, x2 = int(y1), int(x1), int(y2), int(x2)
            exemplar_scale = image_scale[:,:,y1:y2+1, x1:x2+1]
            exemplar_scales.append(exemplar_scale)
        exemplar_scale_list.append(exemplar_scales) # n, sca
    return image_scale_list, exemplar_scale_list, boxes_scale_list

def crop_roi_feat(feat, boxes, out_stride):
    """
    feat: 1 x c x h x w
    boxes: m x 4, 4: [y_tl, x_tl, y_br, x_br]
    """
    _, _, h, w = feat.shape
    boxes_scaled = boxes / out_stride
    boxes_scaled[:, :2] = torch.floor(boxes_scaled[:, :2])  # y_tl, x_tl: floor
    boxes_scaled[:, 2:] = torch.ceil(boxes_scaled[:, 2:])  # y_br, x_br: ceil
    boxes_scaled[:, :2] = torch.clamp_min(boxes_scaled[:, :2], 0)
    boxes_scaled[:, 2] = torch.clamp_max(boxes_scaled[:, 2], h)
    boxes_scaled[:, 3] = torch.clamp_max(boxes_scaled[:, 3], w)
    feat_boxes = []
    for idx_box in range(0, boxes.shape[0]):
        y_tl, x_tl, y_br, x_br = boxes_scaled[idx_box]
        y_tl, x_tl, y_br, x_br = int(y_tl), int(x_tl), int(y_br), int(x_br)
        feat_box = feat[:, :, y_tl : (y_br + 1), x_tl : (x_br + 1)]
        feat_boxes.append(feat_box)
    return feat_boxes

# 2. 模型参数初始化
def init_weights_normal(module, std=0.001):
    for m in module.modules():
        if (isinstance(m, nn.Conv2d) or 
            isinstance(m, nn.Linear) or 
            isinstance(m, nn.ConvTranspose2d)
        ):
            nn.init.normal_(m.weight.data, std=std)
            if m.bias is not None:
                m.bias.data.zero_()


def init_weights_xavier(module, method):
    for m in module.modules():
        if (
            isinstance(m, nn.Conv2d)
            or isinstance(m, nn.Linear)
            or isinstance(m, nn.ConvTranspose2d)
        ):
            if "normal" in method:          # 正态
                nn.init.xavier_normal_(m.weight.data)
            elif "uniform" in method:       # 均匀
                nn.init.xavier_uniform_(m.weight.data)
            else:
                raise NotImplementedError(f"{method} not supported")
            if m.bias is not None:
                m.bias.data.zero_()


def init_weights_msra(module, method):
    for m in module.modules():
        if (
            isinstance(m, nn.Conv2d)
            or isinstance(m, nn.Linear)
            or isinstance(m, nn.ConvTranspose2d)
        ):
            if "normal" in method:
                nn.init.kaiming_normal_(m.weight.data, a=1)
            elif "uniform" in method:
                nn.init.kaiming_uniform_(m.weight.data, a=1)
            else:
                raise NotImplementedError(f"{method} not supported")
            if m.bias is not None:
                m.bias.data.zero_()


def initialize(model, method, **kwargs):
    # initialize BN, LN, Conv, & FC with different methods
    # initialize BN, LN
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    # initialize Conv & FC
    if method == "normal":
        init_weights_normal(model, **kwargs)
    elif "msra" in method:
        init_weights_msra(model, method)
    elif "xavier" in method:
        init_weights_xavier(model, method)
    else:
        raise NotImplementedError(f"{method} not supported")


def initialize_from_cfg(model, cfg):
    if cfg is None:
        initialize(model, "normal", std=0.001)
        return

    cfg = copy.deepcopy(cfg)
    method = cfg.pop("method")
    initialize(model, method, **cfg)
