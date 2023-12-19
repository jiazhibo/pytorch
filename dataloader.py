import torchvision
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

test_data = torchvision.datasets.CIFAR10(root='./data', train=False, transform=torchvision.transforms.ToTensor(),download=True)
test_loader = DataLoader(test_data, batch_size=4, shuffle=True, num_workers=0, drop_last=True)

img, target = test_data[0]
print(img.shape)
print(target)

writer = SummaryWriter("dataloader")

# writer.add_image("Image", img)
for echo in range(2):
    step = 0
    for data in test_loader:
        imgs, target = data
        # print(img.shape)
        # print(target)
        #这里面是images，加载一个图片是image
        writer.add_images("Echo", imgs, step)
        step = step + 1

writer.close()