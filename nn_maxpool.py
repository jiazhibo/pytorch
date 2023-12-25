import torch
import torchvision
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=torchvision.transforms.ToTensor(),download=True)
dataloader = DataLoader(dataset, batch_size=64)

# input = torch.tensor([[1, 2, 0, 3, 1],
#                       [0, 1, 2, 3, 1],
#                       [1, 2, 1, 0, 0],
#                       [5, 2, 3, 1, 1],
#                       [2, 1, 0, 1, 1]], dtype=torch.float32)
# input = torch.reshape(input, (-1, 1, 5, 5))
# print(input.shape)
class module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, input):
        output = self.maxpool1(input)
        return output
model = module()
# output = model(input)
# print(output)
writer = SummaryWriter("logs")
step = 0

for data in dataloader:
    imgs, labels = data
    output = model(imgs)
    writer.add_images("images", imgs, step)
    writer.add_images("maxpool", output, step)
    step += 1

writer.close()