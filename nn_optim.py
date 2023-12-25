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
#lr学习速率，开始比较大，后面比较小，太大模型不稳定，太小训练比较慢
optim = torch.optim.SGD(model.parameters(), lr=0.01)
#实际上次数都是成百上千的
for epoch in range(10):
    running_loss = 0.0
    #相当于只对数据进行了1轮学习
    for data in test_loader:
        #图片转成tensor类型，可以直接放入神经网络中
        input, label = data
        outputs = model(input)
        result_loss = loss(outputs, label)
        #梯度清零
        optim.zero_grad()
        #反向传播
        result_loss.backward()
        #参数调优
        optim.step()
        running_loss = running_loss +result_loss
    print(running_loss)