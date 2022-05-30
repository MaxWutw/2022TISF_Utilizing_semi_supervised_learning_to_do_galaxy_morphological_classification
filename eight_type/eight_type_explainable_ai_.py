import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision.datasets import DatasetFolder
import torchvision
from tqdm.notebook import tqdm as tqdm
from torchsampler import ImbalancedDatasetSampler
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import copy
import os
import cv2
from PIL import Image
from PIL import ImageFile
import torch.autograd as autograd
import torchvision.transforms as transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image,  preprocess_image

train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.')
else:
    print('CUDA is available!')
device = "cuda" if train_on_gpu else "cpu"

supervised_path = "/home/chisc/workspace/wuzhenrong/galaxy/eight_type/train/"
val_image_path = "/home/chisc/workspace/wuzhenrong/galaxy/eight_type/validation/"
test_image_path = "/home/chisc/workspace/wuzhenrong/galaxy/eight_type/test/"
batch_size = 16
train_trans = transforms.Compose([
#                                   transforms.RandomHorizontalFlip(),
#                                   transforms.RandomRotation((-30, 30)),
#                                   transforms.Resize((256, 256)),
#                                   transforms.RandomCrop(size=(100, 100)),
                                  transforms.Resize((256, 256)),
#                                   transforms.CenterCrop(200),
#                                   transforms.Resize((256, 256)),
#                                   transforms.Resize((255, 255)),
#                                   transforms.GaussianBlur(7,3),
#                                   transforms.ColorJitter(contrast=0.8),
                                  transforms.ToTensor()])
train_data = ImageFolder(supervised_path, transform=train_trans)
train_loader = DataLoader(train_data, pin_memory=True, batch_size=batch_size, sampler=ImbalancedDatasetSampler(train_data))

val_trans = transforms.Compose([transforms.Resize((256, 256)),transforms.ToTensor()])
val_data = ImageFolder(val_image_path, transform = val_trans)
val_loader = DataLoader(val_data, shuffle = True)

test_trans = transforms.Compose([transforms.Resize((256, 256)),transforms.ToTensor()])
test_data = ImageFolder(test_image_path, transform = test_trans)
test_loader = DataLoader(test_data)

# model = torchvision.models.vgg16(pretrained=False)
model = torch.load('E_I_S_new.pkl')
model = model.to('cpu')

# Resnet18 and 50: model.layer4[-1]
# VGG and densenet161: model.features[-1]
# mnasnet1_0: model.layers[-1]
# ViT: model.blocks[-1].norm1
# SwinT: model.layers[-1].blocks[-1].norm1

target_layer = model.features[-1]


path = ["E/PGC0002149", "S0/PGC0000243", "Sa/PGC0000639", "Sb/PGC0002440", "SBa/PGC0000635", "SBb/PGC0010726", "SBc/PGC0006826", "Sc/PGC0001933"]
for i in range(8):
    img_path = f"/home/chisc/workspace/wuzhenrong/galaxy/eight_type/test/{path[i]}.png"
    print(img_path)
    rgb_img = cv2.imread(img_path, 1)[:, :, ::-1]   # 1 : read RGB
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])   # torch.Size([1, 3, 224, 224])
    cam = GradCAM(model=model, target_layer=target_layer, use_cuda=False)
    # If target_category is None, the highest scoring category
    # will be used for every image in the batch.
    # target_category can also be an integer, or a list of different integers
    # for every image in the batch.
    target_category = None # 281
    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category, eigen_smooth=True)  # [batch, 224,224]
    grayscale_cam = grayscale_cam[0]
    visualization = show_cam_on_image(rgb_img, grayscale_cam)  # (224, 224, 3)
    cv2.imwrite(f"/home/chisc/workspace/wuzhenrong/CAM-grad/cam_galaxy{i}.jpg", visualization)