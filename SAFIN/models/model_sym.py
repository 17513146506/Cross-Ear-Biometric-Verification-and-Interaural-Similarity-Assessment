
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out
# Modified ResNet18 (outputs feature maps)
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

    def _make_layer(self, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = [ResNetBlock(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResNetBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x  # [batch_size, 512, 20, 20] for 640x640 input


# Symmetry Alignment Module
class SymmetryAlignmentModule(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.channel_compressor = nn.Sequential(
            nn.Conv2d(in_dim * 2, in_dim, 1),
            nn.BatchNorm2d(in_dim),
            nn.ReLU()
        )
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(in_dim, 1, 3, padding=1),
            nn.Sigmoid()
        )
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_dim, in_dim // 8, 1),
            nn.ReLU(),
            nn.Conv2d(in_dim // 8, in_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, left, right):
        right_flipped = torch.flip(right, [3])
        concat = torch.cat([left, right_flipped], dim=1)
        #print(concat.shape)
        compressed = self.channel_compressor(concat)
        spatial_weight = self.spatial_attn(compressed)
        channel_weight = self.channel_attn(compressed)
        return compressed * spatial_weight * channel_weight


class FINModule(nn.Module):
    def __init__(self, feature_dim):
        super(FINModule, self).__init__()
        # 初始特征处理，去掉 BatchNorm 或替换为 LayerNorm
        self.feature_processor = nn.Sequential(
            nn.Linear(2 * feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),  # 使用 LayerNorm 替代 BatchNorm
            nn.ReLU()
        )
        # 交互增强
        self.interaction_enhancer = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        # 输出投影
        self.output_proj = nn.Linear(feature_dim // 2, 1)
    def forward(self, left_embed, right_embed):
        diff_features = torch.abs(left_embed - right_embed)
        #print(diff_features.shape)
        prod_features = left_embed * right_embed
        combined = torch.cat([diff_features, prod_features], dim=1)
        processed = self.feature_processor(combined)
        enhanced = self.interaction_enhancer(processed)
        logits = self.output_proj(enhanced)

        return logits.view(-1)
# Enhanced Neural Network-based Similarity Model with Symmetry
class EnhancedNeuralNetworkSimilarity(nn.Module):
    def __init__(self, feature_dim=2048):
        super(EnhancedNeuralNetworkSimilarity, self).__init__()
        self.feature_extractor = ResNet18()
        self.symmetry_alignment = SymmetryAlignmentModule(in_dim=512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.FIN = FINModule(feature_dim)
        self.fc = nn.Linear(512, feature_dim)
    def forward(self, left_img, right_img):
        # Extract feature maps
        left_features = self.feature_extractor(left_img)  # [1, 512, 20, 20]
        right_features = self.feature_extractor(right_img)  # [1, 512, 20, 20]
        # Bidirectional symmetry alignment
        aligned_left = self.symmetry_alignment(left_features, right_features)  # Left aligned with flipped right
        aligned_right = self.symmetry_alignment(right_features, left_features)  # Right aligned with flipped left
        # Pool and flatten aligned features
        left_features_pooled = self.avgpool(aligned_left)  # [1, 512, 1, 1]
        #print(left_features_pooled.shape)
        right_features_pooled = self.avgpool(aligned_right)  # [1, 512, 1, 1]
        left_features_flat = torch.flatten(left_features_pooled, 1)  # [1, 512]
        right_features_flat = torch.flatten(right_features_pooled, 1)  # [1, 512]
        # Map to embeddings
        left_embed = self.fc(left_features_flat)
        #print(left_embed.shape)
        right_embed = self.fc(right_features_flat)
        # Normalize embeddings
        left_embed = F.normalize(left_embed, p=2, dim=1)
        right_embed = F.normalize(right_embed, p=2, dim=1)
        # Combine features
        logits = self.FIN(left_embed, right_embed)
        return logits.view(-1), left_embed, right_embed
class EnhancedLoss(nn.Module):
    def __init__(self, margin=1.0, initial_bce_weight=0.8, initial_symmetry_weight=0.1, initial_contrastive_weight=0.1):
        super(EnhancedLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()  # 二分类损失
        self.symmetry_loss_weight = initial_symmetry_weight
        self.bce_weight = initial_bce_weight
        self.contrastive_weight = initial_contrastive_weight
        self.margin = margin

    def compute_symmetry_loss(self, left_embed, right_embed):
        """计算对称损失，度量左右嵌入的差异"""
        symmetry_loss = F.pairwise_distance(left_embed, right_embed, p=2).mean()
        return symmetry_loss

    def compute_contrastive_loss(self, left_embed, right_embed, labels):
        """计算对比损失，使不同人的耳朵特征远离"""
        euclidean_distance = F.pairwise_distance(left_embed, right_embed, keepdim=True)
        contrastive_loss = (labels * (euclidean_distance ** 2) +
                            (1 - labels) * F.relu(self.margin - euclidean_distance) ** 2).mean()
        return contrastive_loss

    def forward(self, logits, labels, left_embed=None, right_embed=None):
        # 计算 BCE 损失（用于分类任务）
        bce_loss = self.bce_loss(logits, labels.float())
        symmetry_loss = self.compute_symmetry_loss(left_embed, right_embed) if left_embed is not None else 0.0
        contrastive_loss = self.compute_contrastive_loss(left_embed, right_embed, labels) if left_embed is not None else 0.0
        # 组合损失
        total_loss = (self.bce_weight * bce_loss + self.contrastive_weight * contrastive_loss +self.symmetry_loss_weight*symmetry_loss)
        return  total_loss
    def update_weights(self, epoch, total_epochs):
        """训练过程中动态调整损失权重"""
        self.bce_weight = 1.0 - (epoch / total_epochs) * 0.3
        self.symmetry_loss_weight = (epoch / total_epochs) * 0.2
        self.contrastive_weight = (epoch / total_epochs) * 0.5  # 逐渐增加对比损失的权重

if __name__ == "__main__":
    model = EnhancedNeuralNetworkSimilarity(feature_dim=512)

    # 计算模型的参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {total_params}")

    # Simulate input tensors
    left_img = torch.randn(1, 3, 640, 640)
    right_img = torch.randn(1, 3, 640, 640)

    # Forward pass
    logits, left_embed, right_embed = model(left_img, right_img)

    # Example label
    label = torch.tensor([1])

    # Compute loss
    loss_fn = EnhancedLoss()
    loss = loss_fn(logits, label, left_embed, right_embed)

    print("Logits:", logits.item())
    print("Loss:", loss.item())
