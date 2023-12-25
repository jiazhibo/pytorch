import torch
from torch import nn
from torch.nn import ReLU
import torchvision
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

# input = torch.tensor([[1,-0.5],
#                       [-1,3]])
# input = torch.reshape(input, (-1,1,2, 2))
# print(input.shape)

dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=torchvision.transforms.ToTensor(),download=True)
dataloader = DataLoader(dataset, batch_size=64)

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        # self.relu1 = ReLU()
        self.sigmod1 = nn.Sigmoid()
    
    def forward(self, input):
        # output = self.relu1(input)
        output = self.sigmod1(input)
        return output
module = model()
# output = module(input)
# print(output)

writer = SummaryWriter("logs")
step = 0

for data in dataloader:
    imgs, labels = data
    output = module(imgs)
    writer.add_images("images", imgs, step)
    writer.add_images("sigmod", output, step)
    step += 1

writer.close()