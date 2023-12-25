from torch import nn
import torch
from torch.nn import Conv2d,MaxPool2d,Flatten,Linear
from tensorboardX import SummaryWriter
import torchvision
from torch.utils.data import DataLoader

test_data = torchvision.datasets.CIFAR10(root='./data', train=False, transform=torchvision.transforms.ToTensor(),download=True)
test_loader = DataLoader(test_data, batch_size=1, shuffle=True, num_workers=0, drop_last=True)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model1 = nn.Sequential(
            Conv2d(3,32,5,1,2),
            MaxPool2d(2),
            Conv2d(32,32,5,1,2),
            MaxPool2d(2),
            Conv2d(32,64,5,1,2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024,64),
            Linear(64,10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


model = Model()
loss = nn.CrossEntropyLoss()
#每张图像放到神经网络中
for data in test_loader:
    #图片转成tensor类型，可以直接放入神经网络中
    input, label = data
    outputs = model(input)
    result_loss = loss(outputs, label)
    print(result_loss)
    #反向传播
    result_loss.backward()