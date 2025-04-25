import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from datasets_p import BalancedEarDataset, get_transform
from model_sym import EnhancedNeuralNetworkSimilarity
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve,confusion_matrix
def find_best_threshold_tpr_fpr(similarity_scores, true_labels):
    # 计算FPR, TPR 和阈值
    fpr, tpr, thresholds = roc_curve(true_labels, similarity_scores)
    # 计算 1 - FPR
    one_minus_fpr = 1 - fpr
    # 找到 TPR 与 1-FPR 最接近的交点
    best_index = np.argmin(np.abs(tpr - one_minus_fpr))
    best_threshold = thresholds[best_index]
    # 绘制 TPR 与 1 - FPR 曲线
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, tpr, label="TPR (True Positive Rate)", color='red')
    plt.plot(thresholds, one_minus_fpr, label="1 - FPR (True Negative Rate)", color='blue')
    # 标出最佳阈值
    plt.axvline(x=best_threshold, color='green', linestyle='--', label=f"Best Threshold: {best_threshold:.4f}")
    plt.xlabel("Threshold")
    plt.ylabel("Rate")
    plt.title("TPR vs 1-FPR Curve and Best Threshold")
    plt.legend()
    plt.grid(True)
    # 保存TPR vs 1-FPR曲线图
    tpr_fpr_path = "tpr_vs_1_fpr_with_best_threshold0.png"
    plt.savefig(tpr_fpr_path, dpi=300)
    plt.close()
    
    # 计算 EER (Equal Error Rate)
    # EER是FPR = 1 - TPR时的点
    eer_index = np.argmin(np.abs(fpr - (1 - tpr)))
    eer = fpr[eer_index]
    # 绘制ROC曲线
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label="ROC Curve", color='blue')
    # 绘制反斜对角线
    plt.plot([0, 1], [1, 0], color='gray', linestyle='--', label='Diagonal')
    # 标出EER点
    plt.scatter(eer, 1 - eer, color='red', marker='x', label=f'EER: {eer:.4f}')
    # 添加图形细节
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC Curve with EER")
    plt.legend()
    plt.grid(True)
    
    # 保存ROC曲线图
    roc_curve_path = "roc_curve_with_eer0-1.1.png"
    plt.savefig(roc_curve_path, dpi=300)
    plt.close()

    return best_threshold, eer, tpr_fpr_path, roc_curve_path
def plot_confusion_matrix(true_labels, predicted_labels, save_path="confusion_matrix.png"):
    # 确保 save_path 不为空
    if not save_path:
        save_path = "confusion_matrix.png"
    
    # 获取目录路径并确保目录存在
    dir_path = os.path.dirname(save_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    # 计算混淆矩阵
    cm = confusion_matrix(true_labels, predicted_labels)
    
    # 创建混淆矩阵的热力图
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['True 0', 'True 1'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    
    # 保存混淆矩阵图像
    plt.savefig(save_path, dpi=300)
    plt.close()
def predict(args):
    # 设备选择
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = EnhancedNeuralNetworkSimilarity(feature_dim=args.feature_dim).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # 加载数据集
    transform = get_transform()
    dataset = BalancedEarDataset(args.left_dir, args.right_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 统计数据
    true_labels = []
    similarity_scores = []

    with torch.no_grad():
        for left_img, right_img, label, left_path, right_path in dataloader:
            left_img, right_img, label = left_img.to(device), right_img.to(device), label.item()

            # 计算相似度并归一化到 0-1
            similarity, _, _ = model(left_img, right_img)
            similarity = torch.sigmoid(similarity).item()
            
            similarity_scores.append(similarity)
            true_labels.append(label)
            # 打印当前预测结果
            print(f"Left Image: {left_path}, Right Image: {right_path}, Similarity: {similarity:.4f}, Label: {label}")

    # **自动选择最佳阈值**
    best_threshold, eer, tpr_fpr_path, roc_curve_path = find_best_threshold_tpr_fpr(similarity_scores, true_labels)

    # 重新计算 TP, TN, FP, FN，使用最优阈值
    predicted_labels = [1 if sim >= best_threshold else 0 for sim in similarity_scores]

    TP = sum(1 for p, t in zip(predicted_labels, true_labels) if p == 1 and t == 1)
    TN = sum(1 for p, t in zip(predicted_labels, true_labels) if p == 0 and t == 0)
    FP = sum(1 for p, t in zip(predicted_labels, true_labels) if p == 1 and t == 0)
    FN = sum(1 for p, t in zip(predicted_labels, true_labels) if p == 0 and t == 1)
    # 计算指标
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, zero_division=0)
    recall = recall_score(true_labels, predicted_labels, zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, zero_division=0)
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
    fpr = FP / (FP + TN) if (FP + TN) != 0 else 0

    # 输出结果
    print("\nEvaluation Metrics:")
    print(f"Best Threshold: {best_threshold:.4f}")
    print(f"EER: {eer:.4f}")  # Now this will work correctly
    print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"False Positive Rate: {fpr:.4f}")
    # 绘制 TP、TN、FP、FN 统计曲线
    plot_results(TP, TN, FP, FN, args.save_path)
    plot_confusion_matrix(true_labels, predicted_labels, "confusion_matrix2.png")

def plot_results(TP, TN, FP, FN, save_path="tp_tn_fp_fn_distribution0.png"):
    labels = ['TP', 'TN', 'FP', 'FN']
    values = [TP, TN, FP, FN]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.bar(labels, values, color=['green', 'blue', 'red', 'orange'])
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.title('TP, TN, FP, FN Distribution')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.savefig(save_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Predict Ear Matching Model")
    parser.add_argument("--left_dir", type=str, default="images_L", help="Path to left ear images directory")
    parser.add_argument("--right_dir", type=str, default="images_R", help="Path to right ear images directory")
    parser.add_argument("--model_path", type=str, default="best_8.pth", help="Path to trained model")
    parser.add_argument("--feature_dim", type=int, default=512, help="Feature dimension of the model")
    parser.add_argument("--device", type=str, default="cuda:2", help="Device to use for prediction")
    parser.add_argument("--save_path", type=str, default="./tp_tn_fp_fn_distribution_2.png", help="Path to save the results plot")
    args = parser.parse_args()

    predict(args)
