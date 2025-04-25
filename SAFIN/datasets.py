# -*- coding=utf-8 -*-
import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
class BalancedEarDataset(Dataset):
    def __init__(self, left_dir, right_dir, transform=None, return_paths=False):
        self.left_dir = left_dir
        self.right_dir = right_dir

        self.left_images = sorted([f for f in os.listdir(left_dir) if f.endswith(('.jpg', '.bmp'))])
        self.right_images = sorted([f for f in os.listdir(right_dir) if f.endswith(('.jpg', '.bmp'))])


        # 检查左右耳图像数量是否一致
        if len(self.left_images) != len(self.right_images):
            raise ValueError(f"Left and right directories have different numbers of images: "
                             f"{len(self.left_images)} vs {len(self.right_images)}")

        self.transform = transform
        self.return_paths = return_paths

        # 创建正负样本对（优化随机负样本生成逻辑）
        self.pairs = []
        indices = list(range(len(self.right_images)))
        for i in range(len(self.left_images)):
            # 添加正样本对
            self.pairs.append((i, i, 1))

            # 添加负样本对（随机采样非自身的索引）
            negative_idx = random.choice([idx for idx in indices if idx != i])
            self.pairs.append((i, negative_idx, 0))

        print(f"Dataset initialized with {len(self.pairs)} pairs (balanced positive and negative samples).")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        left_idx, right_idx, label = self.pairs[idx]

        left_image_path = os.path.join(self.left_dir, self.left_images[left_idx])
        right_image_path = os.path.join(self.right_dir, self.right_images[right_idx])

        try:
            left_image = Image.open(left_image_path).convert('RGB')
            right_image = Image.open(right_image_path).convert('RGB')
        except Exception as e:
            raise ValueError(f"Error loading images: {left_image_path}, {right_image_path}. {str(e)}")

        if self.transform:
            left_image = self.transform(left_image)
            right_image = self.transform(right_image)
        # 根据 return_paths 决定是否返回路径
        if self.return_paths:
            return left_image, right_image, label, left_image_path, right_image_path
        else:
            return left_image, right_image, label, left_image_path, right_image_path

    def get_left_image_path(self, idx):
        left_idx, _, _ = self.pairs[idx]
        return os.path.join(self.left_dir, self.left_images[left_idx])

    def get_right_image_path(self, idx):
        _, right_idx, _ = self.pairs[idx]
        return os.path.join(self.right_dir, self.right_images[right_idx])


def get_transform(augment=False):
    if augment:
        return transforms.Compose([
            transforms.Resize((640, 1024)),  # 调整大小
            transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
            transforms.RandomRotation(degrees=10),  # 随机旋转
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 随机颜色抖动
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((640, 1024)),  # 调整大小
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
        ])


# 测试 BalancedEarDataset
if __name__ == "__main__":
    # 使用增强的 transform
    dataset = BalancedEarDataset("images_L", "images_R", transform=get_transform(augment=True), return_paths=True)

    positive_count = 0
    negative_count = 0

    for i in range(len(dataset)):  # 遍历整个数据集
        _, _, label, left_path, right_path = dataset[i]
        if label == 1:
            positive_count += 1
        else:
            negative_count += 1

    print(f"Positive samples: {positive_count}, Negative samples: {negative_count}")
