import torch
import torch.nn as nn
import os 
import torch.nn.functional as F
import copy
from networks.net_units import get_activation, build_backbone


class Exemplar_scale_aug(nn.Module):
    def __init__(self, backbone, embed_dim, match_dim, out_stride=8, max_stride=32, exemplar_scales=[0.8, 1, 1.2]):
        super(Exemplar_scale_aug, self).__init__()
        self.backbone = backbone
        self.conv = nn.Sequential(
            nn.Conv2d(embed_dim, match_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(match_dim),
            nn.ReLU(),
            nn.Conv2d(match_dim, match_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(match_dim),
            nn.ReLU()
        )
        self.out_stride = out_stride
        self.max_stride = max_stride
        self.exemplar_scales = exemplar_scales
        if 1 in self.exemplar_scales:
            self.exemplar_scales.remove(1)
    
    def crop_roi_feat(self, feat, boxes, out_stride):
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
    
    def __call__(self, image, boxes):
        feat = self.backbone(image)
        feat = self.conv(feat)
        # multi-scale exemplars
        _, _, h, w = image.shape
        # 尺度增强的特征和boxes
        feat_scale_list = []
        boxes_scale_list = []
        for scale in self.exemplar_scales:
            h_rsz = int(h * scale) // self.max_stride * self.max_stride
            w_rsz = int(w * scale) // self.max_stride * self.max_stride
            image_scale = F.interpolate(image, size=(w_rsz, h_rsz), mode="bilinear")
            scale_h = h_rsz / h
            scale_w = w_rsz / w
            boxes_scale = copy.deepcopy(boxes)  # y1, x1, y2, x2
            boxes_scale[:, 0] *= scale_h
            boxes_scale[:, 1] *= scale_w
            boxes_scale[:, 2] *= scale_h
            boxes_scale[:, 3] *= scale_w
#             feat_scale = self.extractor(image_scale)
            feat_scale = self.backbone(image_scale)
            feat_scale = self.conv(feat_scale)
            feat_scale_list.append(feat_scale)
            boxes_scale_list.append(boxes_scale)

        # 所有尺度特征和boxes
        feat_list = [feat] + feat_scale_list
        boxes_list = [boxes] + boxes_scale_list
        # exemplar 特征
        exemplars_list = []
        for feat_, boxes_ in zip(feat_list, boxes_list):
            feat_boxes = self.crop_roi_feat(feat_, boxes_, out_stride=self.out_stride)
            exemplars_list.append(feat_boxes)

        return feat, exemplars_list

def print_value(tensor, name):
    print("【{}】|max:{:5.5f} ｜ min:{:2.5f} | mean:{:5.5f}".format(name, tensor.max(), tensor.min(), tensor.mean()))

class Match(nn.Module):
    def __init__(self, head, head_dim):
        super().__init__()
        self.head = head
        self.head_dim = head_dim

        self.conv_out = nn.Sequential(
            nn.Conv2d(head_dim*2, head_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(head_dim),
            nn.ReLU()
        )
        self.norm = nn.LayerNorm([1, 64, 64])
    
    def forward(self, feat, exemplar):
        _, _, h_p, w_p = exemplar.size()
        _, _, h_f, w_f = feat.size()
        # feat: 1, c, h, w. exemplar: 1, c, h, w
        feat = feat.contiguous().view(self.head, self.head_dim, h_f, w_f)         # [head,head_dim,h,w]
        exemplar = exemplar.contiguous().view(self.head, self.head_dim, h_p, w_p) # [head,head_dim,h,w]
        pad = (w_p // 2, w_p // 2, h_p // 2, h_p // 2)
        attn_list = []
        for q, k in zip(feat, exemplar):
            # 相似性度量。  卷积、注意力、距离
            attn = F.conv2d(F.pad(q.unsqueeze(0), pad), k.unsqueeze(0))           # [1,1,h,w]
            # print_value(attn, "attn0")
            attn = self.norm(attn) 
            attn = torch.sigmoid(attn)
            attn_list.append(attn)
        attn = torch.cat(attn_list, dim=0)                                        # [head,1,h,w]
        attn_feat = torch.mul(feat, attn)                                         # [head,head_dim,h,w]
        feat = torch.concat([attn_feat, feat], dim=1)                             # [head,head_dim*2,h,w]
        feat = self.conv_out(feat)
        feat = feat.contiguous().view(1, self.head_dim*self.head, h_f, w_f)       # [1, head*head_dim,h,w]
        return feat

# 激活函数选择
def get_activation(activation='relu'):
    if activation=='relu':
        return nn.ReLU()
    elif activation=='leaky_relu':
        return nn.LeakyReLU()
    else:
        raise NotImplementedError

# 回归
class Regressor(nn.Module):
    def __init__(self, in_dim, activation):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Conv2d(in_dim, in_dim // 2, 5, padding=2),
            get_activation(activation),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_dim // 2, in_dim // 4, 3, padding=1),
            get_activation(activation),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_dim // 4, in_dim // 8, 1),
            get_activation(activation),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_dim // 8, 1, 1),
            get_activation("relu"),
        )

    def forward(self, x):
        return self.regressor(x)

def build_regressor(**kwargs):
    return Regressor(**kwargs)


class Model(nn.Module):
    def __init__(self, 
                 backbone,
                 pool,
                 out_stride=8,     # [4, 8, 16, 32]
                 shot=3, 
                 head=8,
                 head_dim=128,
                 exemplar_scales=[0.8, 1, 1.2],
                ):
        super().__init__()
        self.backbone = build_backbone(**backbone)
        self.pool_cfg = pool
        self.shot = shot
        self.head = head
        self.head_dim = head_dim
        self.exemplar_scale_aug = Exemplar_scale_aug(self.backbone, 
                                                     embed_dim=3584, 
                                                     match_dim=head*head_dim, 
                                                     out_stride=out_stride, 
                                                     max_stride=32, 
                                                     exemplar_scales=exemplar_scales)
        self.match = Match(head, head_dim)
        self.conv = nn.Sequential(
            nn.Conv2d(head_dim*head*2, head_dim*head, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(head_dim*head),
            nn.ReLU()
        )
        self.conv_out = nn.Sequential(
            nn.Conv2d(head_dim*head, head_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(head_dim),
            nn.ReLU()
        )
        self.regressor = Regressor(in_dim=head_dim,  activation='relu')
        
    def forward(self, image, boxes):
        # image: 1,c,h,w
        boxes_orig = boxes[0]
        # feat_orig: 1,c,h/out_stride,w/out_stride
        feat_orig, exemplars_list = self.exemplar_scale_aug(image, boxes_orig)
        # num_scale, num_boxes, feat. => M, K, 1,c,h',w'  => 3, 3(4), 1,c,h',w'
        match_feats_scale_shot_list = []
        feat_match = feat_orig     # 1, c, h, w
#         print(feat_match.shape)
        for i in range(self.shot):            # 不同样本
            match_feats_scale_list = []
            for exemplars in exemplars_list:  # 不同尺度
                exemplar = exemplars[i]
                if self.pool_cfg['type'] == 'max':
                    exemplar = nn.AdaptiveMaxPool2d(self.pool_cfg['size'])(exemplar)
                elif self.pool_cfg['type'] == 'avg':
                    exemplar = nn.AdaptiveAvgPool2d(self.pool_cfg['size'])(exemplar)
                else:
                    raise NotImplementedError
                match_feat = self.match(feat_match, exemplar)   # [1, head*head_dim,h,w]
                match_feats_scale_list.append(match_feat)
            match_feats = torch.stack(match_feats_scale_list)   # scales, 1, c, h, w, 

            feat_match = torch.concat([torch.sum(match_feats, dim=0), feat_match], dim=1)  # 1, 2*c, h, w. 
            feat_match = self.conv(feat_match)                  # 1, c, h, w. [head, head_dim, h, w]
        feat = self.conv_out(feat_match)
        feat = self.regressor(feat)
#         print("feat:{}".format(feat.shape))
        output = feat
        return output


def build_network(net_cfg):
    return Model(**net_cfg)

