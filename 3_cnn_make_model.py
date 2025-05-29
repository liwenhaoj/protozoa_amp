import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_curve, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# 设置随机种子
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything()

# 数据增强：随机替换氨基酸（模拟突变）
def augment_sequence(seq, max_len=100, mutation_rate=0.15):
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    seq = list(seq)
    for i in range(len(seq)):
        if random.random() < mutation_rate:  # 以一定概率替换
            seq[i] = random.choice(amino_acids)
    return ''.join(seq[:max_len])

# 定义氨基酸序列的编码方式
def encode_sequence(seq, max_len=100):
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    aa_to_idx = {aa: idx + 1 for idx, aa in enumerate(amino_acids)}  # 从1开始编码
    encoded = [aa_to_idx.get(aa, 0) for aa in seq[:max_len]]  # 0表示未知氨基酸
    return np.pad(encoded, (0, max_len - len(encoded)), 'constant')

# 构建数据集
class AminoAcidDataset(Dataset):
    def __init__(self, sequences, labels, max_len=100, augment=False):
        if augment:  # 数据增强
            self.sequences = [encode_sequence(augment_sequence(seq), max_len) for seq in sequences]
        else:
            self.sequences = [encode_sequence(seq, max_len) for seq in sequences]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float)

# 定义改进的卷积神经网络
class AminoAcidCNN(nn.Module):
    def __init__(self, num_classes=1, vocab_size=21, embed_dim=64, max_len=100):
        super(AminoAcidCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embed_dim, 128, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout_conv = nn.Dropout(0.3)  # 卷积层后添加 Dropout
        self.dropout_fc = nn.Dropout(0.5)    # 全连接层前 Dropout
        self.fc = nn.Linear(64 * (max_len // 4), num_classes)  # 修正后的输入维度

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = self.dropout_conv(self.pool(self.relu(self.conv1(x))))
        x = self.dropout_conv(self.pool(self.relu(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout_fc(x)
        x = self.fc(x)
        return torch.sigmoid(x).squeeze(1)

# 加载数据
database_file = r"TrainingAMP_3.csv"
df = pd.read_csv(database_file)
sequences = df.iloc[:, 1].tolist()
labels = df.iloc[:, 0].tolist()

# 将数据划分为训练集、验证集和测试集
train_sequences, test_sequences, train_labels, test_labels = train_test_split(
    sequences, labels, test_size=0.2, random_state=42
)
train_sequences, val_sequences, train_labels, val_labels = train_test_split(
    train_sequences, train_labels, test_size=0.2, random_state=42
)

# 创建数据集和数据加载器
train_dataset = AminoAcidDataset(train_sequences, train_labels, augment=True)
val_dataset = AminoAcidDataset(val_sequences, val_labels, augment=False)
test_dataset = AminoAcidDataset(test_sequences, test_labels, augment=False)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 模型训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AminoAcidCNN(num_classes=1).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.00005, weight_decay=1e-3)  # 增大 weight_decay 到 1e-3
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)  # patience 减至 2

# 记录训练过程中的损失和准确率
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
best_val_loss = float('inf')  # 用于保存最佳模型

epochs = 20  
for epoch in range(epochs):
    model.train()
    epoch_train_loss = 0
    correct_train = 0
    total_train = 0
    for batch in train_dataloader:
        sequences, labels = batch
        sequences, labels = sequences.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item()
        predictions = (outputs > 0.5).float()
        correct_train += (predictions == labels).sum().item()
        total_train += labels.size(0)

    train_losses.append(epoch_train_loss / len(train_dataloader))
    train_accuracies.append(correct_train / total_train)

    # 验证集
    model.eval()
    epoch_val_loss = 0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for batch in val_dataloader:
            sequences, labels = batch
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            epoch_val_loss += loss.item()
            predictions = (outputs > 0.5).float()
            correct_val += (predictions == labels).sum().item()
            total_val += labels.size(0)

    val_losses.append(epoch_val_loss / len(val_dataloader))
    val_accuracies.append(correct_val / total_val)

    # 更新学习率
    scheduler.step(val_losses[-1])

    # 保存验证损失最低的模型
    if val_losses[-1] < best_val_loss:
        best_val_loss = val_losses[-1]
        torch.save(model.state_dict(), 'best_amino_acid_cnn.pth')

    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, "
          f"Train Acc: {train_accuracies[-1]:.4f}, Val Acc: {val_accuracies[-1]:.4f}")

# 绘制损失和准确率曲线
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend()

plt.savefig('loss_accuracy_curve.pdf', format='pdf')
plt.show()

# 测试模型
model.load_state_dict(torch.load('best_amino_acid_cnn.pth'))  # 加载最佳模型
model.eval()
all_labels = []
all_preds = []
y_true = []
y_probs = []

with torch.no_grad():
    for batch in test_dataloader:
        sequences, labels = batch
        sequences, labels = sequences.to(device), labels.to(device)
        outputs = model(sequences)
        predictions = (outputs > 0.5).float()

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predictions.cpu().numpy())
        y_true.extend(labels.cpu().numpy())
        y_probs.extend(outputs.cpu().numpy())

# 计算混淆矩阵和 F1 分数
conf_matrix = confusion_matrix(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
print(f"Test F1 Score: {f1:.4f}")

plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.pdf', format='pdf')
plt.show()

# 绘制 ROC 曲线
fpr, tpr, thresholds = roc_curve(y_true, y_probs)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='b', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig('roc_curve.pdf', format='pdf')
plt.show()

# 绘制精确率-召回率曲线
precision, recall, _ = precision_recall_curve(y_true, y_probs)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.', label='Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.savefig('precision_recall_curve.pdf', format='pdf')
plt.show()

# 保存最终模型权重
torch.save(model.state_dict(), 'amino_acid_cnn_final.pth')

roc_output = pd.DataFrame({'label': y_true, 'prob': y_probs})
roc_output.to_csv('roc_curve_output.csv', index=False, float_format="%.10f")