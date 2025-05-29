import torch
from transformers import BertModel
import torch.nn as nn


# 定义模型结构
class BERTForPeptideClassification(nn.Module):
    def __init__(self, bert_model_name, num_classes=2):
        super(BERTForPeptideClassification, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits


##  输入文件路径
model_file = r"bert_peptide_classification_model.pth_13"
sequence_file = r"AMPS_1.fasta"
# 加载模型权重
model = BERTForPeptideClassification(bert_model_name="bert-base-uncased")
model.load_state_dict(torch.load(model_file))
model.eval()  # 设置为评估模式

# 自定义分词器
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


# 读取新的序列文件（需要修改以保留序列名称）
sequences_dict = {}
current_seq_name = ""
with open(sequence_file, "r") as f:
    for line in f:
        line = line.strip()
        if line.startswith(">"):
            current_seq_name = line[1:]  # 移除 ">" 获取序列名称
        elif line and current_seq_name:  # 确保有序列名称且行不为空
            sequences_dict[current_seq_name] = line

# 获取序列列表用于预测
new_sequences = list(sequences_dict.values())
seq_names = list(sequences_dict.keys())

# 初始化分词器并进行预测（这部分保持不变）
tokenizer = PeptideTokenizer()
max_length = 20
inputs = tokenizer(new_sequences, return_tensors='pt', max_length=max_length, padding=True, truncation=True)
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_ids = input_ids.to(device)
attention_mask = attention_mask.to(device)
model = model.to(device)

with torch.no_grad():
    logits = model(input_ids=input_ids, attention_mask=attention_mask)
    probs = torch.softmax(logits, dim=1)

predicted_probs = probs[:, 1].cpu().numpy()

# 修改输出格式并保存到文件
output_file = "bert_predict_13.txt"
with open(output_file, "w") as f:
    for seq_name, seq, prob in zip(seq_names, new_sequences, predicted_probs):
        # 使用 \t 分隔，概率保留4位小数
        line = f"{seq_name}\t{seq}\t{prob:.4f}\n"
        f.write(line)
        print(line, end='')  # 同时打印到控制台

print(f"\nResults have been saved to {output_file}")