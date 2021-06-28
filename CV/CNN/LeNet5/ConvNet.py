# !/usr/local/bin/python3
# @Time : 2021/6/23 14:42
# @Author : Tianlei.Shi
# @Site :
# @File : ConvNet.py
# @Software : PyCharm

import  torch
from    torch import nn
from    torch.nn import functional as F


class Lenet5(nn.Module):

    # 定义具体的网络结构
    def __init__(self):
        super(Lenet5, self).__init__()

        self.conv = nn.Sequential(
            # x: [b, 3, 32, 32] => [b, 16, ]
            # 通道数=3,深度=16，卷积核：5*5，步长=1
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=0),
            # 卷积核：2*2，步长=2
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            # 基于上面深度为16，通道数则变为16，卷积核5*5=>[b,32,]
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

        )
        # flatten
        # 全连接操作 fc unit
        self.fc = nn.Sequential(
            # nn.Linear()函数其实就是在做一个y=wx+b的线性操作
            # x:[32*5*5]  b:[32]
            nn.Linear(32*5*5, 32),
            nn.ReLU(),
            # nn.Linear(120, 84),
            # nn.ReLU(),
            nn.Linear(32, 1)
        )

        self._initialize_weights()


    def forward(self, x):

        batchsz = x.size(0)
        # [b, 3, 32, 32] => [b, 16, 5, 5]
        x = self.conv(x)
        # [b, 16, 5, 5] => [b, 16*5*5]
        x = x.view(-1, x.size()[1:].numel())
        # [b, 16*5*5] => [b, 10]
        # logits一般用来表示未经过softmax()操作
        logits = self.fc(x)


        return logits

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)  #正态分布赋值
                nn.init.constant_(m.bias, 0)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Lenet5()

net.to(device)
net.eval()

img = torch.randn(1, 3, 32, 32).to(device)
outputs = net(img)
print(outputs)