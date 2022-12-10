## 三种数据集
'''
1/ FSC-147 image、density、boxes、points、size、filename
'''
# # 数据集提供的密度图
# from data.fsc.dataloader import build_dataloader
# loader = build_dataloader(cfg.dataset, distributed=False)
# for index, sample in enumerate(loader[0]):
#     image, density, boxes, points, size, img_name = sample
#     batchsize = image.shape[0]
#     for i in range(batchsize):
#         show_img_boxes_points(image[i].unsqueeze(0), boxes[i], points[i])
#         show_density(density[i].unsqueeze(0).unsqueeze(0))
# #     show_img_boxes_points(image, boxes.squeeze[0], points)
# #     print(image.shape)
# #     print(density.shape)
#     if index==2:
#         break;

# dataset:
#   batch_size: 1
#   workers: 2
#   shot: 3
#   img_dir: &img_dir /home/zg/benchmark/FSC147_384_V2/images_384_VarV2/
#   density_dir: /home/zg/benchmark/FSC147_384_V2/gt_density_map_adaptive_384_VarV2
#   size: [512, 512] # [h, w]
#   mean: [0.485, 0.456, 0.406]
#   std: [0.229, 0.224, 0.225]
#   train:
#     meta_file: /home/zg/benchmark/FSC147_384_V2/train.json
#   val:
#     meta_file: /home/zg/benchmark/FSC147_384_V2/val.json
#   test:
#     meta_file: /home/zg/benchmark/FSC147_384_V2/test.json


'''
2/ 生成有形状信息的数据集
'''
# 根据 points，boxes 生成的密度图
# from data.dataloader import build_dataloader
# loader = build_dataloader(cfg.dataset, distributed=False)
# for index, sample in enumerate(loader[0]):
#     image, density, boxes, points, size, img_name = sample
#     batchsize = image.shape[0]
#     for i in range(batchsize):
#         show_img_boxes_points(image[i].unsqueeze(0), boxes[i], points[i])
#         show_density(density[i].unsqueeze(0).unsqueeze(0))
# #     show_img_boxes_points(image, boxes.squeeze[0], points)
# #     print(image.shape)
# #     print(density.shape)
#     if index==2:
#         break;

# dataset:
#   batch_size: 1
#   workers: 2
#   shot: 3
#   img_dir: &img_dir /home/zg/benchmark/FSC147_384_V2/images_384_VarV2/
#   size: [512, 512] # [h, w]
#   mean: [0.485, 0.456, 0.406]
#   std: [0.229, 0.224, 0.225]
#   train:
#     meta_file: /home/zg/benchmark/FSC147_384_V2/train.json
#   val:
#     meta_file: /home/zg/benchmark/FSC147_384_V2/val.json
#   test:
#     meta_file: /home/zg/benchmark/FSC147_384_V2/test.json

'''
3/ 无密度图版本数据集
'''
# from data.bayesian.dataloader import build_dataloader
# import matplotlib.pyplot as plt
# from utils.show_helper import show_density, show_img_boxes_points, show_tensor_img, show_channel_weights
# loader = build_dataloader(cfg.dataset)
# for index, sample in enumerate(loader[0]):
#     image, target, boxes, points, size, img_name = sample 
#     print(image.size(0))
#     print(len(points))
#     batchsize = image.shape[0]
#     for i in range(batchsize):
#         print("len_points:{}".format(len(points[i])))
#         print("target_sum:{}".format(target[i].sum()))
#         show_img_boxes_points(image[i].unsqueeze(0), boxes[i], points[i])
#         show_density(target[i])
#     if index==2:
#         break;

# dataset:
#   batch_size: 4
#   workers: 2
#   img_dir: &img_dir /home/zg/benchmark/FSC147_384_V2/images_384_VarV2/
#   size: [512, 512] # [h, w]
#   mean: [0.485, 0.456, 0.406]
#   std: [0.229, 0.224, 0.225]
#   train:
#     meta_file: /home/zg/benchmark/FSC147_384_V2/train.json
#   val:
#     meta_file: /home/zg/benchmark/FSC147_384_V2/val.json
#   test:
#     meta_file: /home/zg/benchmark/FSC147_384_V2/test.json