# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")
image_path = "dataset2/train/ants_image/2265825502_fff99cfd2d.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)
print(type(img_array))
print(img_array.shape)
#2对应步骤
writer.add_image('ants', img_array, 2, dataformats='HWC')

for i in range(100):
    #图表的 标题 y轴 x轴
    writer.add_scalar('Loss/train', 2*i, i)

writer.close()
