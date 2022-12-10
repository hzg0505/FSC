import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn

def show_density(tensor, title='density'):
    fig = plt.figure()
    density = tensor.squeeze(0).squeeze(0).detach().numpy()
    plt.imshow(np.array(density), cmap=plt.cm.jet)
    plt.title(title)
    plt.show()

def show_img_boxes_points(img_tensor, boxes, points):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    mean=torch.Tensor([0.485, 0.456, 0.406])
    std=torch.Tensor([0.229, 0.224, 0.225])
    img = img_tensor.permute(0,2,3,1)
    img *= std
    img += mean
    img *= 255
    plt.imshow(np.array(img[0]).astype(np.uint64))
    _, H, W, C = img.shape
    for box in boxes:
        y1, x1, y2, x2 = box
        width = (x2-x1) #* W
        height = (y2-y1) #* H
        rect = plt.Rectangle((int(x1), int(y1)), int(width), int(height), fill=False, edgecolor = 'white',linewidth=1)
        ax.add_patch(rect)
    for dot in points:
        x, y = dot
        point = plt.Circle((x,y), 3, color='r')
        ax.add_patch(point)
    plt.show()
    
def show_tensor_img(tensor, title='image'):
    plt.figure()
    mean=torch.Tensor([0.485, 0.456, 0.406])
    std=torch.Tensor([0.229, 0.224, 0.225])
    img = tensor.permute(0,2,3,1)
    img *= std
    img += mean
    img *= 255
    plt.imshow(np.array(img[0]).astype(np.uint64))
    plt.title(title)
    plt.show()

def show_channel_weights(feat, width=100, title="weights"):   # c,1,1
    weights = nn.AdaptiveAvgPool2d(1)(feat)
    c = weights[0].shape[0]
    weight = weights.cpu() * torch.ones(c, width, 1)
    # weight = weight.squeeze(2).detach().numpy()*255
    # plt.imshow(np.array(weight).astype(np.uint64), cmap=plt.cm.jet)
    weight = weight.permute(1, 0, 2)
    weight = weight.squeeze(2).detach().numpy()
    plt.imshow(np.array(weight), cmap=plt.cm.jet)
    plt.yticks([])
    plt.title(title)
    plt.show()

