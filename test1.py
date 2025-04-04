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

#이 이미지를 Neural Net이 알아보기 쉬운 형태로 바꿔야함.
#torchvision > transforms > ToTensor() : Converts a PIL Image or numpy.ndarray(HxWxC) in the range[0, 255] to a torchFloatTensor of shape(CxHxW) in the range[0.0, 1.0]
    #tensor: 다차원의 배열. 여러 차원을 가진 배열. 다양한 수치(스칼라, 벡터, 행렬 등)를 모아놓은 구조화된 덩어리.
#torchvision > transforms > Normalize() : class. Normalize a tensor image with mean and standard deviation. 정규화(1: -1~1 / 2: 0~1 평균 0.5인 정규분포를 갖도록 정규화.) -> 2번 정규화.
    # 더많은 이미지 변환을 적용할 수 있다.


#ToTensor()와 Normalize()를 하나의 pipeline화 시킴: transforms = transforms.Compose()를 이용.
transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))      #Normalizing for each RGB channel
])
#dataset에 전달해주면, 접근할 때마다 pipeline을 거쳐 변환함.
train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transforms, download=True)
#batch size만큼 최종 이미지가 반환되도록 함.
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

#Define Neural Network
class MyNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.seq_model = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=True),

            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),

            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),

            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_size, hidden_size, bias=True),

            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.seq_model(x)


mynet = MyNet(input_size=32*32*3, hidden_size=128, output_size=10)

