import os
from typing import Generator, Tuple
import cv2
import PIL
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models, utils
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import IterableDataset
from torchdata.datapipes.iter.random import RandomShuffleQueue
import more_itertools # 导入more_itertools模块
from torchvision.transforms import functional as F # 导入torchvision.transforms.functional模块，并起一个别名F
import PIL # 导入PIL模块
import PIL.ImageFile # 导入PIL.ImageFile模块

# 设置一些训练参数
NUM_CLASSES = 2  # 两个类别：Beauty和Non-Beauty
BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
ALPHA = 0.5  # 调整此值以平衡权重，根据需要进行调整

# 初始化计数器用于跟踪样本数量
num_beauty_samples = 0
num_non_beauty_samples = 0

# 定义一个名为MyIterableDataset的子类
class MyIterableDataset(IterableDataset):
    def __init__(self, generator):
        self.generator = generator

    def __iter__(self):
        return self.generator


# 检查是否支持CUDA，如果支持则使用GPU，否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备：{device}")
PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True # 忽略错误图像

def contains_person(image: PIL.Image) -> bool:
    """检测图像中是否包含人物

    Args:
        image (PIL.Image): 图像对象

    Returns:
        bool: 如果包含人物，返回True；否则返回False
    """
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 使用Haar级联分类器检测人物
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    return len(faces) > 0


def generate_soft_label(image: torch.Tensor) -> torch.Tensor:
    """根据包含人物的情况来生成软标签

    Args:
        image (torch.Tensor): 图像张量

    Returns:
        torch.Tensor: 软标签，形状为(2,)
    """
    # 将图像张量转换为图像对象
    image = F.to_pil_image(image)
    
    if contains_person(image):
        return torch.tensor([0.0, 1.0])  # 如果包含人物，将软标签设为[0.0, 1.0]（Non-Beauty）
    else:
        return torch.tensor([1.0, 0.0])  # 如果不包含人物，将软标签设为[1.0, 0.0]（Beauty）


# 定义数据转换操作
transform = transforms.Compose([
    transforms.Lambda(lambda x: transforms.functional.resize(x, 256)),  # 将图片缩放到 256x256 的尺寸
    # 使用 hflip 方法来替换 random_horizontal_flip 方法，并删除 p 参数
    transforms.Lambda(lambda x: F.hflip(x)),  # 直接水平翻转图片
    # 使用一个浮点数对象，而不是一个元组对象，作为仿射变换的 scale 参数，并随机地选择一个缩放因子
    transforms.Lambda(lambda x: F.affine(x, angle=15, translate=(0.1, 0.1), scale=random.uniform(0.9, 1.1), shear=15)),  # 随机仿射变换
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 随机颜色变化
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.01)  # 随机添加高斯噪声
])

# 创建数据集，同时生成软标签并更新计数器
data_root = 'train'  # 相对于项目路径的相对路径
train_data = []

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# 使用torchvision.datasets.ImageFolder来加载图片数据
train_data = datasets.ImageFolder(data_root, transform=transform)


def train_data_generator() -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
    """使用生成器来创建数据集，同时生成软标签并更新计数器

    Yields:
        Generator[Tuple[torch.Tensor, torch.Tensor], None, None]: 图像和软标签的元组
    """
    global num_beauty_samples, num_non_beauty_samples

    for image, label in train_data:
        # 不需要获取图像文件路径，直接使用图像张量作为参数传入函数
        soft_label = generate_soft_label(image)
        yield image, soft_label

        # 更新计数器
        num_beauty_samples += int(soft_label[0].item())
        num_non_beauty_samples += int(soft_label[1].item())


# 定义自定义的数据组合函数，用于处理不同大小的张量
def custom_collate_fn(batch):
    images = []
    soft_labels = []
    for image, soft_label in batch:
        images.append(image)
        soft_labels.append(soft_label)
    
    # 使用torch.cat来连接张量，而不是torch.stack
    images = torch.cat(images)
    soft_labels = torch.cat(soft_labels)

    return images, soft_labels


# 创建数据加载器，使用自定义的数据组合函数
train_loader = DataLoader(RandomShuffleQueue(MyIterableDataset(train_data_generator())), batch_size=BATCH_SIZE, collate_fn=custom_collate_fn)

# 定义模型并将模型移动到GPU
model = models.resnet18(pretrained=True) # 使用预训练的resnet18作为特征提取器
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES) # 替换最后一层为自定义的分类层
model = model.to(device)

# 使用交叉熵损失函数
criterion = nn.CrossEntropyLoss()

# 定义优化器，同时优化所有参数
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 定义学习率调整策略，每10个epoch降低一半的学习率
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# 创建一个tensorboard对象，用于可视化训练过程
writer = SummaryWriter()

# 开始训练模型
model.train()

for epoch in range(NUM_EPOCHS):
    running_loss = 0.0

    for i, (inputs, soft_labels) in enumerate(train_loader):
        # 将数据移动到GPU
        inputs = inputs.to(device)
        soft_labels = soft_labels.to(device)

        optimizer.zero_grad()

        # 前向传播和计算损失
        outputs = model(inputs)
        loss1 = criterion(outputs, torch.argmax(soft_labels, dim=1)) # 计算交叉熵损失，使用torch.argmax来获取真实标签
        loss2 = -torch.mean(torch.sum(soft_labels * torch.log_softmax(outputs, dim=1), dim=1)) # 计算软标签损失，使用torch.log_softmax来获取预测概率
        loss = ALPHA * loss1 + (1 - ALPHA) * loss2 # 计算总损失，使用ALPHA来平衡两种损失

        # 反向传播和更新参数
        loss.backward()
        optimizer.step()

        # 打印统计信息
        running_loss += loss.item()
        if i % 200 == 199:    # 每200个批次打印一次平均损失
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            writer.add_scalar('training loss', running_loss / 200,
                              epoch * len(train_loader) + i) # 将平均损失写入tensorboard
            
            running_loss = 0.0

    # 调整学习率
    scheduler.step()

# 保存模型
torch.save(model.state_dict(), 'model.pth')

# 测试模型
model.eval()

# 定义测试数据集和数据加载器
test_data_root = 'test'
test_data = datasets.ImageFolder(test_data_root, transform=transform)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)

# 定义评估指标
correct = 0
total = 0
confusion_matrix = torch.zeros(NUM_CLASSES, NUM_CLASSES)

# 遍历测试数据
with torch.no_grad():
    for inputs, labels in test_loader:
        # 将数据移动到GPU
        inputs = inputs.to(device)
        labels = labels.to(device)

        # 前向传播和预测
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        # 更新评估指标
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        for t, p in zip(labels.view(-1), predicted.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1

# 打印评估结果
print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))
print('Confusion matrix:\n', confusion_matrix)

# 将评估结果写入tensorboard
writer.add_scalar('test accuracy', correct / total)
writer.add_pr_curve('PR curve', labels, outputs[:, 1])
writer.close()
