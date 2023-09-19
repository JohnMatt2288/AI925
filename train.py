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

# 检查是否支持CUDA，如果支持则使用GPU，否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备：{device}")


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


# 创建数据加载器，判断数据集的参数是否是一个生成器对象，如果是，则转换为列表对象，如果不是，则继续使用原来的方式
if isinstance(train_data_generator(), Generator):
    train_loader = DataLoader(list(train_data_generator()), batch_size=BATCH_SIZE, shuffle=True)
else:
    train_loader = DataLoader(train_data_generator(), batch_size=BATCH_SIZE, shuffle=True)

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
        loss1 = criterion(outputs, torch.argmax(soft_labels, dim=1))  # 交叉熵损失，使用软标签进行计算
        loss2 = criterion(outputs, torch.argmax(outputs, dim=1))  # 交叉熵损失，使用模型预测结果进行计算

# 计算总损失，使用alpha参数来平衡两个损失的权重
loss = alpha * loss1 + (1 - alpha) * loss2

# 反向传播和更新参数
loss.backward()
optimizer.step()

# 记录损失值和累积损失值
running_loss += loss.item()
writer.add_scalar('Loss/iter', loss.item(), epoch * len(train_loader) + i)

# 每200个批次打印一次平均损失值和当前学习率
if (i + 1) % 200 == 0:
    print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 200:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
    running_loss = 0.0

# 调整学习率
scheduler.step()

# 记录每个epoch的平均损失值
writer.add_scalar('Loss/epoch', running_loss / len(train_loader), epoch)

# 保存每个epoch结束后的模型参数字典
#model_save_path = f'trained_model_epoch_{epoch + 1}.pth'  # 相对于项目路径的相对路径
#torch.save(model.state_dict(), model_save_path)
#print(f"Epoch {epoch + 1} completed, model saved.")

# 训练循环结束后，保存最终训练结束后的模型参数字典
model_save_path = 'final_trained_model.pth'  # 相对于项目路径的相对路径
torch.save(model.state_dict(), model_save_path)
print("训练完成，模型已保存。")

