import torch
import random
from transformers import BertTokenizer, BertModel
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 初始化BERT模型和tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 示例文本
text = ''
with open("D:\\Codespace\\morphology\\1706.03762v7_no_custom_names.txt", 'r') as f:
    text = f.read()

# 对文本进行分块
max_length = 512
tokens = tokenizer.tokenize(text)
chunks = [tokens[i:i+max_length] for i in range(0, len(tokens), max_length)]

# 存储每个块的嵌入表示
all_embeddings = []

# 处理每个块
for chunk in chunks:
    inputs = tokenizer(chunk, return_tensors='pt', truncation=True, padding=True, is_split_into_words=True)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 提取每个token的嵌入
    token_embeddings = outputs.last_hidden_state.squeeze().numpy()
    all_embeddings.append(token_embeddings)

# 将所有块的嵌入表示合并
all_embeddings = np.vstack(all_embeddings)

# 随机采样部分单词及其嵌入表示
sample_size = 200  # 选择合适的采样数量
sample_indices = random.sample(range(len(tokens)), sample_size)

# 保证 "attention" 一词的token在采样中
attention_indices = [i for i, token in enumerate(tokens) if token.lower() == "attention"]
if attention_indices:
    sample_indices = list(set(sample_indices + attention_indices))
    sample_indices = sample_indices[:sample_size]

sample_tokens = [tokens[i] for i in sample_indices]
sample_embeddings = all_embeddings[sample_indices]

# 去除带#的token
clean_sample_tokens = []
clean_sample_embeddings = []
for i, token in enumerate(sample_tokens):
    if '##' not in token:
        clean_sample_tokens.append(token)
        clean_sample_embeddings.append(sample_embeddings[i])

# 处理 "attention" 一词的token
attention_embeddings = []
final_tokens = []
final_embeddings = []
for i, token in enumerate(clean_sample_tokens):
    if token.lower() == "attention":
        attention_embeddings.append(clean_sample_embeddings[i])
    else:
        final_tokens.append(token)
        final_embeddings.append(clean_sample_embeddings[i])

# 如果存在 "attention" 的token，则计算其平均嵌入表示
if attention_embeddings:
    attention_average = np.mean(attention_embeddings, axis=0)
    final_tokens.append("attention")
    final_embeddings.append(attention_average)

final_embeddings = np.array(final_embeddings)

# 使用PCA降维到2维
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(final_embeddings)

# KNN聚类分析
knn = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(X_reduced)
distances, indices = knn.kneighbors(X_reduced)

# 绘制KNN聚类关系
plt.figure(figsize=(15, 15))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1])

for i in range(len(X_reduced)):
    for j in indices[i]:
        plt.plot([X_reduced[i, 0], X_reduced[j, 0]], [X_reduced[i, 1], X_reduced[j, 1]], 'k-', alpha=0.3)

# 标注单词
for i, word in enumerate(final_tokens):
    plt.annotate(word, (X_reduced[i, 0], X_reduced[i, 1]))

plt.title("Word Embeddings Visualization with PCA (BERT Sampled and Cleaned)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()
