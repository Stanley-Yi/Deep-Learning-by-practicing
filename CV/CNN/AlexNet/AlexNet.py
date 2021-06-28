# !/usr/local/bin/python3
# @Time : 2021/6/22 17:27
# @Author : Tianlei.Shi
# @Site :
# @File : AlexNet.py
# @Software : PyCharm

# https://blog.csdn.net/weixin_44023658/article/details/105798326
# https://blog.csdn.net/qq_30129009/article/details/98772599


# import torch.nn as nn
# import torch
#
#
# class AlexNet(nn.Module):
#     def __init__(self, num_classes=1000, init_weights=False):
#         super(AlexNet, self).__init__()
#         self.features = nn.Sequential(  #打包
#             nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[48, 55, 55] 自动舍去小数点后
#             nn.ReLU(inplace=True), #inplace 可以载入更大模型
#             nn.MaxPool2d(kernel_size=3, stride=2),                  # output[48, 27, 27] kernel_num为原论文一半
#             nn.Conv2d(96, 256, kernel_size=5, padding=2),           # output[128, 27, 27]
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 13, 13]
#             nn.Conv2d(256, 384, kernel_size=3, padding=1),          # output[192, 13, 13]
#             nn.ReLU(inplace=True),
#             nn.Conv2d(384, 384, kernel_size=3, padding=1),          # output[192, 13, 13]
#             nn.ReLU(inplace=True),
#             nn.Conv2d(384, 256, kernel_size=3, padding=1),          # output[128, 13, 13]
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 6, 6]
#         )
#         self.classifier = nn.Sequential(
#             nn.Dropout(p=0.5),
#             #全链接
#             nn.Linear(256 * 6 * 6, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.5),
#             nn.Linear(4096, 4096),
#             nn.ReLU(inplace=True),
#             nn.Linear(4096, num_classes),
#         )
#         if init_weights:
#             self._initialize_weights()
#
#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(-1, x.size()[1:].numel()) #展平   或者view()
#         x = self.classifier(x)
#         return x
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') #何教授方法
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)  #正态分布赋值
#                 nn.init.constant_(m.bias, 0)


import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, _init_weights=True):
        super(AlexNet, self).__init__()

        # input: 224*224*3, kernel_conv1: 11*11*3, stride: 4, num_kernel: 96
        # afterConv_size = (size + 2 * padding - kernel_size) / stride + 1
        # output: 55*55*96
        self.conv1 = nn.Conv2d(3, 96, (11, 11), (4, 4), padding=2)
        self.pooling = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=0)

        self.conv2 = nn.Conv2d(96, 256, (5, 5), padding=2)

        self.conv3 = nn.Conv2d(256, 384, (3, 3), padding=1)

        self.conv4 = nn.Conv2d(384, 384, (3, 3), padding=1)

        self.conv5 = nn.Conv2d(384, 256, (3, 3), padding=1)

        self.drop = nn.Dropout(0.5)

        self.fc1 = nn.Linear(6*6*256, 4096)

        self.fc2 = nn.Linear(4096, 4096)

        self.fc3 = nn.Linear(4096, num_classes)

        if _init_weights:
            self._initialize_weights()


    def forward(self, x):
        x = self.pooling(F.relu(self.conv1(x)))
        x = self.pooling(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pooling(F.relu(self.conv5(x)))

        x = x.view(-1, x.size()[1:].numel())
        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(F.relu(self.fc2(x)))
        x = self.fc3(x)

        return x

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
net = AlexNet(1, True)

net.to(device)
net.eval()

img = torch.randn(1, 3, 224, 224).to(device)
outputs = net(img)
print(outputs)