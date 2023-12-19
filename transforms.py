from torchvision import transforms
from PIL import Image
from tensorboardX import SummaryWriter
import cv2

img_path = 'dataset2/train/ants_image/49375974_e28ba6f17e.jpg'
pil_img = Image.open(img_path)
cv_img = cv2.imread(img_path)

#tensorboard 配置
writer = SummaryWriter("logs")
#ToTensor
tensor_trans = transforms.ToTensor()
tensor_image = tensor_trans(pil_img)
writer.add_image('Tensor_img', tensor_image)

#Normalize
print(tensor_image[0][0][0])
normalize_trans = transforms.Normalize(mean=[6, 3, 2], std=[9, 3, 5])
normalize_image = normalize_trans(tensor_image)
print(normalize_image[0][0][0])
writer.add_image('Normalize', normalize_image,2)

#Resize
resize_trans = transforms.Resize((512, 512))
resize_image = resize_trans(pil_img)
resize_image = tensor_trans(resize_image)
writer.add_image('Resize', resize_image,0)

#compose
compose_trans = transforms.Compose([resize_trans, tensor_trans])
compose_image = compose_trans(pil_img)
writer.add_image('Compose', compose_image,1)
writer.close()