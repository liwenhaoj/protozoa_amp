import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

# 定义FASTA文件解析函数
def parse_fasta(file_path):
    sequences = []
    seq_ids = []
    try:
        with open(file_path, 'r') as f:
            current_seq = ''
            current_id = ''
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_id:  # 保存上一个序列
                        sequences.append(current_seq)
                        seq_ids.append(current_id)
                    current_id = line[1:]  # 去掉'>'
                    current_seq = ''
                else:
                    current_seq += line.upper()  # 转换为大写以确保一致性
            if current_id:  # 保存最后一个序列
                sequences.append(current_seq)
                seq_ids.append(current_id)
        return seq_ids, sequences
    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
        return [], []

# 氨基酸编码（与训练时一致）
def encode_sequence(seq, max_len=100):
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    aa_to_idx = {aa: idx + 1 for idx, aa in enumerate(amino_acids)}  # 从1开始编码
    encoded = [aa_to_idx.get(aa, 0) for aa in seq[:max_len]]  # 0表示未知氨基酸
    return np.pad(encoded, (0, max_len - len(encoded)), 'constant')

# 预测数据集类
class PredictionDataset(Dataset):
    def __init__(self, sequences, max_len=100):
        self.sequences = [encode_sequence(seq, max_len) for seq in sequences]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long)

# 模型定义（与训练时完全一致）
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

# 预测函数
def predict_fasta(model, fasta_path, batch_size=32, threshold=0.5):
    # 解析FASTA文件
    seq_ids, sequences = parse_fasta(fasta_path)
    if not sequences:
        print("错误：没有解析到任何序列")
        return []

    # 创建数据加载器
    dataset = PredictionDataset(sequences)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_probs = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            outputs = model(batch)
            all_probs.extend(outputs.cpu().numpy())

    # 生成结果
    results = []
    for seq_id, seq, prob in zip(seq_ids, sequences, all_probs):
        prediction = 1 if prob >= threshold else 0
        results.append(f"{seq_id}\t{seq}\t{prob:.4f}\t{prediction}")

    return results

if __name__ == "__main__":
    # 加载模型
    model = AminoAcidCNN()
    model_path = 'best_amino_acid_cnn.pth'  # 使用训练时保存的最佳模型
    try:
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    except FileNotFoundError:
        print(f"错误：找不到模型文件 {model_path}")
        exit(1)
    except RuntimeError as e:
        print(f"错误：加载模型失败，可能是模型结构不匹配或文件损坏：{e}")
        exit(1)

    # 输入输出路径
    input_fasta = r"candicate.fasta"  # 替换为你的FASTA文件路径
    output_file = "predictions.txt"  # 输出文件名

    # 执行预测
    prediction_results = predict_fasta(model, input_fasta)

    # 保存结果
    if prediction_results:
        with open(output_file, 'w') as f:
            f.write("Sequence_ID\tSequence\tProbability\tPrediction\n")
            f.write("\n".join(prediction_results))
        print(f"预测完成！结果已保存到 {output_file}")
        print("\n示例结果（前3条）：")
        print("Sequence_ID\tSequence\tProbability\tPrediction")
        print("\n".join(prediction_results[:3]))
    else:
        print("无预测结果可保存")