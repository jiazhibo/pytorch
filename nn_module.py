from torch import nn
import torch
#自己定义一个神经网络
class Test_Module(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output
    
module = Test_Module()
input = torch.tensor([1,2])
output = module(input)
print(output)