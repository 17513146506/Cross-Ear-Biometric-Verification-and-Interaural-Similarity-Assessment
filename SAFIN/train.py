# -*- coding: utf-8 -*-

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import  CosineAnnealingLR
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from datasets import BalancedEarDataset, get_transform
from model import EnhancedNeuralNetworkSimilarity, EnhancedLoss  # 修改为增强版模型和损失函数
from tqdm import tqdm
import matplotlib.pyplot as plt
def train(args):
    # 打印初始信息
    print(f"Training on device: {args.device}")
    print(f"Training with dataset: Left Images at '{args.left_dir}', Right Images at '{args.right_dir}'")
    print(f"Batch size: {args.batch_size}, Epochs: {args.epochs}, Learning rate: {args.learning_rate}")
    print(f"Saving best model to: {args.save_path}")
    print(f"Using feature dimension: {args.feature_dim}")
    print("=" * 80)
    # 加载数据集
    transform = get_transform(augment=True)  # 启用数据增强
    dataset = BalancedEarDataset(args.left_dir, args.right_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # 初始化模型、损失函数、优化器等
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = EnhancedNeuralNetworkSimilarity(feature_dim=args.feature_dim).to(device)  # 使用增强版模型
    criterion = EnhancedLoss()
    # 在train.py中修改优化器和调度器
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    metrics = {'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

    best_f1 = 0.0
    best_model_state = None

    scaler = torch.cuda.amp.GradScaler()  # 混合精度训练

    # 开始训练
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        all_labels = []
        all_preds = []
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch + 1}")
        # 训练过程
        for batch_idx, (left_img, right_img, label,Left_path,Right_path) in progress_bar:
            left_img, right_img, label = left_img.to(device), right_img.to(device), label.to(device)
             # 打印每个批次的数据，按行输出
            #print(f"Batch {batch_idx + 1}:")
            #for i in range(len(Left_path)):
                #print(f"Leftimg: {Left_path[i]}, Rightimg: {Right_path[i]}, Label: {label[i].item()}")
            with torch.cuda.amp.autocast():  # 自动混合精度
                logits, left_embed, right_embed = model(left_img, right_img)  # logits 直接作为输出
                # Create the loss function instance
                loss_fn = EnhancedLoss()
                # Compute loss
                loss = loss_fn(logits, label,embeddings=left_embed)  # Remove the 'embeddings' argument here

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            # 修改train.py中的预测逻辑
            preds = (torch.sigmoid(logits) > 0.5).float()
            all_labels.extend(label.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
        # 计算当前 epoch 的指标
        epoch_accuracy = accuracy_score(all_labels, all_preds)
        epoch_precision = precision_score(all_labels, all_preds, zero_division=1)
        epoch_recall = recall_score(all_labels, all_preds, zero_division=1)
        epoch_f1 = f1_score(all_labels, all_preds, zero_division=1)

        metrics['loss'].append(epoch_loss)
        metrics['accuracy'].append(epoch_accuracy)
        metrics['precision'].append(epoch_precision)
        metrics['recall'].append(epoch_recall)
        metrics['f1'].append(epoch_f1)

        print(
            f"Epoch [{epoch + 1}/{args.epochs}] Loss: {epoch_loss:.4f}, "
            f"Accuracy: {epoch_accuracy:.4f}, Precision: {epoch_precision:.4f}, "
            f"Recall: {epoch_recall:.4f}, F1: {epoch_f1:.4f}"
        )

        # 保存最佳模型
        if epoch_f1 > best_f1:
            best_f1 = epoch_f1
            best_model_state = model.state_dict()
            print(f"Best model updated at epoch {epoch + 1} with F1: {best_f1:.4f}")

        scheduler.step()  # 更新学习率
        loss_fn.update_weights(epoch, args.epochs)
        # 保存最佳模型到指定路径
    if best_model_state is not None:
        torch.save(best_model_state, args.save_path)
        print(f"Best model saved to {args.save_path} with F1: {best_f1:.4f}")

    # 绘制训练曲线
    plot_metrics(metrics, save_path="training_metrics_SAM.png")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Enhanced Ear Matching Model")
    parser.add_argument("--left_dir", type=str, default="images_L", help="Path to left ear images directory")
    parser.add_argument("--right_dir", type=str, default="images_R", help="Path to right ear images directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")  # 增大批量大小
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")  # 更高的学习率
    parser.add_argument("--feature_dim", type=int, default=512, help="Dimension of the feature embeddings")
    parser.add_argument("--device", type=str, default="cuda:3", help="Device to use for training")
    parser.add_argument("--save_path", type=str, default="best.pth", help="Path to save the trained model")
    args = parser.parse_args()
    train(args)
