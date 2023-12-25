import torch
import torchvision
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter




dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=torchvision.transforms.ToTensor(),download=True)
dataloader = DataLoader(dataset, batch_size=64)

class module(torch.nn.Module):
    def __init__(self):
        super(module, self).__init__()
        self.linear1 = torch.nn.Linear(196608, 10)

    def forward(self, input):
        output = self.linear1(input)
        return output
    
model = module()
for data in dataloader:
    imgs, targets = data
    print(imgs.shape)
    # output = torch.reshape(imgs, (1,1,1,-1))
    output = torch.flatten(imgs)
    print(output.shape)
    output = model(output)
    print(output.shape)
    