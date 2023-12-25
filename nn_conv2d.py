import torch 
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn import Conv2d
from tensorboardX import SummaryWriter

#训练数据集太大了，用测试数据集
dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=torchvision.transforms.ToTensor(),download=True)
dataloader = DataLoader(dataset, batch_size=64)

class Test_Module(nn.Module):
    def __init__(self):
        super().__init__()
        #有一个卷积层
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    #输入进行一次卷积
    def forward(self, input):
        output = self.conv1(input)
        return output

module = Test_Module()
# print(module)

writer = SummaryWriter("logs")
step = 0
#每张图像放到神经网络中
for data in dataloader:
    #图片转成tensor类型，可以直接放入神经网络中
    input, label = data
    output = module(input)
    print(input.shape)
    print(output.shape)
    #(64,3,32,32)
    writer.add_images("input", input, step)
    #(64,6,30,30)->(xxx,3,30,30),因为变3之后，batchsize会加倍变128
    #6个channel不知道怎么显示,不知道是多少，写-1
    output = torch.reshape(output,(-1,3,30,30))
    writer.add_images("output", output, step)
    step = step + 1

writer.close()