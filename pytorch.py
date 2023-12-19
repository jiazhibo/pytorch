#util 工具区
from torch.utils.data import Dataset, DataLoader
#系统模块
import os
from PIL import Image 
#继承Dataset类
class MyData(Dataset):
    #提供全局变量 self:类当中全局变量
    def __init__(self,root_dir,label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        #路径拼接
        self.path = os.path.join(root_dir, label_dir)
        #把文件夹里的所有图片路径都存入列表
        self.img_path_list = os.listdir(self.path)

        
    def __getitem__(self, idx):
        img_name = self.img_path_list[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label
    
    def __len__(self):
        return len(self.img_path_list)

root_dir = 'dataset/train'
ants_label_dir = 'ants'
bees_label_dir = 'bees'
ants_dataset = MyData(root_dir, ants_label_dir)
bees_dataset = MyData(root_dir, bees_label_dir)
image, label = ants_dataset[0]
print(len(ants_dataset))
#两个数据集合并
train_dataset = ants_dataset + bees_dataset