#
# import torch
# import torch.optim as optim
# import torch.optim.lr_scheduler as lr_scheduler
# from torch.utils.tensorboard import SummaryWriter
# from torchvision import transforms
# from my_dataset import MyDataSet
# from vit_model import vit_base_patch16_224_in21k as create_model
# from utils import read_split_data, train_one_epoch, evaluate
#
# data_path = "data/flower_photos"
#
# train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(data_path)
#
# data_transform = {
#     "train": transforms.Compose([transforms.RandomResizedCrop(224),
#                                  transforms.RandomHorizontalFlip(),
#                                  transforms.ToTensor(),
#                                  transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
#     "val": transforms.Compose([transforms.Resize(256),
#                                transforms.CenterCrop(224),
#                                transforms.ToTensor(),
#                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}
#
# # 实例化训练数据集
# train_dataset = MyDataSet(images_path=train_images_path,
#                           images_class=train_images_label,
#                           transform=data_transform["train"])
#
# # 实例化验证数据集
# val_dataset = MyDataSet(images_path=val_images_path,
#                         images_class=val_images_label,
#                         transform=data_transform["val"])
#
#
# print(train_dataset)
#
# print(val_dataset)
import os
import json
import random

root = "data/flower_photos"
val_rate = 0.2
flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
# 排序，保证各平台顺序一致
print(flower_class)
flower_class.sort()
# 生成类别名称以及对应的数字索引
class_indices = dict((k, v) for v, k in enumerate(flower_class))
json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)

train_images_path = []  # 存储训练集的所有图片路径
train_images_label = []  # 存储训练集图片对应索引信息
val_images_path = []  # 存储验证集的所有图片路径
val_images_label = []  # 存储验证集图片对应索引信息
every_class_num = []  # 存储每个类别的样本总数
supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
# 遍历每个文件夹下的文件
for cla in flower_class:
    cla_path = os.path.join(root, cla)
    # 遍历获取supported支持的所有文件路径
    images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
              if os.path.splitext(i)[-1] in supported]
    print(images)
    # 排序，保证各平台顺序一致
    images.sort()
    # 获取该类别对应的索引
    image_class = class_indices[cla]
    # 记录该类别的样本数量
    every_class_num.append(len(images))
    # 按比例随机采样验证样本
    val_path = random.sample(images, k = int(len(images) * val_rate))


    for img_path in images:
        if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
            val_images_path.append(img_path)
            val_images_label.append(image_class)
        else:  # 否则存入训练集
            train_images_path.append(img_path)
            train_images_label.append(image_class)

print("{} images were found in the dataset.".format(sum(every_class_num)))
print("{} images for training.".format(len(train_images_path)))
print("{} images for validation.".format(len(val_images_path)))
assert len(train_images_path) > 0, "number of training images must greater than 0."
assert len(val_images_path) > 0, "number of validation images must greater than 0."