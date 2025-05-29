import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from transformers import BertModel, AdamW
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from torch.optim import AdamW
from sklearn.preprocessing import label_binarize

# 1.准备数据集


def load_imdb_data(data_file):
    df = pd.read_csv(data_file)  # 读取CSV文件
    texts = df['Sequence'].tolist()  # 提取Sequence列
    labels = df['Label'].tolist()  # 提取标签列
    return texts, labels


data_file = r"TrainingAMP_3.csv"
sequences, labels = load_imdb_data(data_file)

# 2.数据集划分
# 划分训练集和测试集
train_sequences, test_sequences, train_labels, test_labels = train_test_split(
    sequences, labels, test_size=0.2, random_state=42
)
# 进一步划分训练集和验证集
train_sequences, val_sequences, train_labels, val_labels = train_test_split(
    train_sequences, train_labels, test_size=0.2, random_state=42
)

# 3.自定义分词器
# 将肽序列转换为模型可接受的输入格式。
# BERT 的分词器是为自然语言设计的，而肽序列是生物序列，因此需要自定义分词器。
# 分词器将每个氨基酸转换为一个 token，并生成 input_ids 和 attention_mask


class PeptideTokenizer:
    def __init__(self):
        self.vocab = {"A": 1, "C": 2, "D": 3, "E": 4, "F": 5,
                      "G": 6, "H": 7, "I": 8, "K": 9, "L": 10,
                      "M": 11, "N": 12, "P": 13, "Q": 14, "R": 15,
                      "S": 16, "T": 17, "V": 18, "W": 19, "Y": 20}
        self.pad_token_id = 0  # 填充符的 ID

    def __call__(self, sequences, return_tensors=None, max_length=None, padding=False, truncation=False):
        input_ids = []
        attention_mask = []
        for seq in sequences:
            # 将序列转换为 ID
            ids = [self.vocab.get(aa, self.pad_token_id) for aa in seq]
            # 截断或填充
            if truncation and len(ids) > max_length:
                ids = ids[:max_length]
            if padding and len(ids) < max_length:
                ids = ids + [self.pad_token_id] * (max_length - len(ids))
            input_ids.append(ids)
            # 生成注意力掩码
            mask = [1] * len(ids) + [0] * (max_length - len(ids)) if padding else [1] * len(ids)
            attention_mask.append(mask)
        # 转换为张量
        if return_tensors == "pt":
            input_ids = torch.tensor(input_ids)
            attention_mask = torch.tensor(attention_mask)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


#  4. 创建数据集类
#  将数据封装为 PyTorch 的 Dataset 对象，方便使用 DataLoader 进行批量加载。
#  实现 __len__ 和 __getitem__ 方法，支持索引访问和长度查询。

class PeptideClassificationDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_length):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        # 将肽序列转化为模型可接受的输入格式
        encoding = self.tokenizer([sequence], return_tensors='pt', max_length=self.max_length,
                                  padding=True, truncation=True)
        return {
            'input_ids': encoding['input_ids'].squeeze(0),  # 输入 ID
            'attention_mask': encoding['attention_mask'].squeeze(0),  # 注意力掩码
            'label': torch.tensor(label)  # 标签
        }


#  5. 定义 BERT 模型
#  使用 BERT 模型提取肽序列的特征，并添加一个分类头进行二分类。
#  BERT 的 [CLS] 位置的隐藏状态用于分类任务

class BERTForPeptideClassification(nn.Module):
    def __init__(self, bert_model_name, num_classes=2):
        super(BERTForPeptideClassification, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        # 获取 BERT 输出
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 提取 [CLS] 位置的隐藏状态
        cls_output = outputs.last_hidden_state[:, 0, :]
        # 分类
        logits = self.classifier(cls_output)
        return logits


# 6评估函数
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            # 保存预测结果和概率
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(torch.softmax(logits, dim=1).cpu().numpy())
    accuracy = correct / total
    avg_loss = total_loss / len(dataloader)
    return avg_loss, accuracy, all_preds, all_labels, all_probs


# 绘制损失曲线和准确率曲线
def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, save_path):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()

    plt.savefig(save_path)
    plt.close()

# 绘制混淆矩阵
def plot_confusion_matrix(all_labels, all_preds, save_path):
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(save_path)
    plt.close()

# 绘制 ROC 曲线
def plot_roc_curve(all_labels, all_probs, save_path):
    # 确保 all_probs 是二维 NumPy 数组
    all_probs = np.array(all_probs)
    if all_probs.ndim != 2:
        raise ValueError("all_probs 必须是二维数组，形状为 (n_samples, n_classes)")

    # 计算 ROC 曲线
    fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
    roc_auc = auc(fpr, tpr)

    # 绘制 ROC 曲线
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(save_path)
    plt.close()

# 绘制精确率-召回率曲线
def plot_precision_recall_curve(all_labels, all_probs, save_path):
    # 确保 all_probs 是二维 NumPy 数组
    all_probs = np.array(all_probs)
    if all_probs.ndim != 2:
        raise ValueError("all_probs 必须是二维数组，形状为 (n_samples, n_classes)")

    # 计算精确率-召回率曲线
    precision, recall, _ = precision_recall_curve(all_labels, all_probs[:, 1])

    # 绘制曲线
    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, color="blue", lw=2, label="Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.savefig(save_path)
    plt.close()





# 8. 训练模型
# 通过优化损失函数（如交叉熵损失）来调整模型参数，使模型能够正确分类肽序列。
# 使用验证集监控模型性能，避免过拟合。
# 初始化模型、优化器和损失函数
# 初始化分词器和数据集
tokenizer = PeptideTokenizer()
max_length = 50
train_dataset = PeptideClassificationDataset(train_sequences, train_labels, tokenizer, max_length)
val_dataset = PeptideClassificationDataset(val_sequences, val_labels, tokenizer, max_length)
test_dataset = PeptideClassificationDataset(test_sequences, test_labels, tokenizer, max_length)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

# 初始化模型、优化器和损失函数
model = BERTForPeptideClassification(bert_model_name="bert-base-uncased")
# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)
optimizer = AdamW(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()

# 训练循环
num_epochs = 20
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        # 前向传播
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    # 计算训练集损失和准确率
    train_loss = total_loss / len(train_loader)
    train_accuracy = correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # 验证集评估
    val_loss, val_accuracy, _, _, _ = evaluate(model, val_loader, criterion, device)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    print(
        f"Epoch{epoch + 1},Train Loss:{train_loss},Train Accuracy:{train_accuracy},Val Loss:{val_loss},Val Accuracy:{val_accuracy}")
    # 保存模型
    torch.save(model.state_dict(), f"bert_peptide_classification_model.pth_{epoch+ 1}")
# 测试集评估
test_loss, test_accuracy, all_preds, all_labels, all_probs = evaluate(model, test_loader, criterion, device)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
auc_data = r"auc_data.csv"
# 确保 all_probs 是 NumPy 数组
all_probs = np.array(all_probs)

# 创建 DataFrame
df = pd.DataFrame({'label': all_labels, 'prob': all_probs[:, 1]})
df.to_csv(auc_data, index=False)
# 绘制并保存图表
# 绘制并保存图表
plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, "metrics_curves.pdf")
plot_confusion_matrix(all_labels, all_preds, "confusion_matrix.pdf")
plot_roc_curve(all_labels, all_probs, "roc_curve.pdf")
plot_precision_recall_curve(all_labels, all_probs, "precision_recall_curve.pdf")

# 保存模型
torch.save(model.state_dict(), "bert_peptide_classification_model.pth")