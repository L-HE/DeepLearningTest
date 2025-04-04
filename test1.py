from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


#Load dataset
#CIFAR10: 작은 사이즈에 3개의 color channel을 가지고 있어, 이미지의 preprocessing 과정을 볼 수 있음.
CIFAR10_trainset = datasets.CIFAR10('~/.pytorch/CIFAR10_data/', download=True, train=True)
print(len(CIFAR10_trainset))        #50000개의 이미지를 가지고 있음

#Fetch the first image and its label
CIFAR10_img, CIFAR10_label = CIFAR10_trainset[6]
#Convert the PIL Image to a numpy array
CIFAR10_img_array = np.array(CIFAR10_img)

print(f"label: {CIFAR10_label}")        #0~9까지의 클래스 중 2를 가짐
print(f"shape: {CIFAR10_img_array.shape}")      #(가로, 세로, color channel)

#Display the image
plt.imshow(CIFAR10_img_array)   #No need for cmap='gray' for CIFAR10
plt.title(f'label: {CIFAR10_label}')
plt.show()

