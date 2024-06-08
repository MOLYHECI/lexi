import torch
import random
from transformers import BertTokenizer, BertModel
from gensim.models import KeyedVectors
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import gensim.downloader as api


# 初始化BERT模型和tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# 加载预训练的Word2Vec模型
word2vec_model = api.load('word2vec-google-news-300')
# 读取文本
text = ''
with open("D:\\Codespace\\morphology\\modified_text.txt", 'r') as f:
    text = f.read()

# 对文本进行分块
max_length = 512
tokens = tokenizer.tokenize(text)
chunks = [tokens[i:i+max_length] for i in range(0, len(tokens), max_length)]

# 存储每个块的BERT嵌入表示
all_bert_embeddings = []

# 处理每个块
for chunk in chunks:
    inputs = tokenizer(chunk, return_tensors='pt', truncation=True, padding=True, is_split_into_words=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    
    # 提取每个token的嵌入
    token_embeddings = outputs.last_hidden_state.squeeze().numpy()
    all_bert_embeddings.append(token_embeddings)

# 将所有块的嵌入表示合并
all_bert_embeddings = np.vstack(all_bert_embeddings)

# 忽略长度少于4个字符的单词
filtered_indices = [i for i, token in enumerate(tokens) if len(token) >= 4]
filtered_tokens = [tokens[i] for i in filtered_indices]
filtered_embeddings = all_bert_embeddings[filtered_indices]

# 确保过滤后的token和嵌入表示数量一致
assert len(filtered_tokens) == len(filtered_embeddings), "Filtered tokens and embeddings count mismatch"

# 随机采样部分单词及其嵌入表示
sample_size = 200  # 选择合适的采样数量
sample_size = min(sample_size, len(filtered_tokens))  # 防止样本数量超过过滤后的token数量
sample_indices = random.sample(range(len(filtered_tokens)), sample_size)

# 保证 "a" 一词的token在采样中
a_indices = [i for i, token in enumerate(filtered_tokens) if token.lower() == "a"]
if a_indices:
    sample_indices = list(set(sample_indices + a_indices))
    sample_indices = sample_indices[:sample_size]

sample_tokens = [filtered_tokens[i] for i in sample_indices]
sample_embeddings = filtered_embeddings[sample_indices]

# 去除带#的token
clean_sample_tokens = []
clean_sample_bert_embeddings = []
for i, token in enumerate(sample_tokens):
    if '##' not in token:
        clean_sample_tokens.append(token)
        clean_sample_bert_embeddings.append(sample_embeddings[i])

# 处理 "a" 一词的token
a_embeddings = []
final_tokens = []
final_embeddings = []
for i, token in enumerate(clean_sample_tokens):
    if token.lower() == "a":
        a_embeddings.append(clean_sample_bert_embeddings[i])
    else:
        final_tokens.append(token)
        final_embeddings.append(clean_sample_bert_embeddings[i])

# 如果存在 "a" 的token，则计算其平均嵌入表示
if a_embeddings:
    a_average = np.mean(a_embeddings, axis=0)
    final_tokens.append("a")
    final_embeddings.append(a_average)

final_embeddings = np.array(final_embeddings)

# 获取Word2Vec嵌入表示
clean_sample_word2vec_embeddings = []
for token in final_tokens:
    if token.lower() in word2vec_model:
        clean_sample_word2vec_embeddings.append(word2vec_model[token.lower()])
    else:
        # 如果Word2Vec中没有该token的嵌入，使用零向量填充
        clean_sample_word2vec_embeddings.append(np.zeros(word2vec_model.vector_size))

clean_sample_word2vec_embeddings = np.array(clean_sample_word2vec_embeddings)

# 使用PCA降维到2维
pca = PCA(n_components=2)
bert_reduced = pca.fit_transform(final_embeddings)
word2vec_reduced = pca.fit_transform(clean_sample_word2vec_embeddings)

# 找到“a”在final_tokens中的索引
a_index = final_tokens.index("a") if "a" in final_tokens else None

# 确保对齐
if a_index is not None:
    bert_offset = bert_reduced - bert_reduced[a_index]
    word2vec_offset = word2vec_reduced - word2vec_reduced[a_index]
else:
    bert_offset = bert_reduced
    word2vec_offset = word2vec_reduced

# 可视化
plt.figure(figsize=(15, 15))

# 绘制BERT嵌入结果
plt.scatter(bert_offset[:, 0], bert_offset[:, 1], color='blue', label='BERT')
for i, word in enumerate(final_tokens):
    plt.annotate(word, (bert_offset[i, 0], bert_offset[i, 1]), color='blue')

# 绘制Word2Vec嵌入结果
plt.scatter(word2vec_offset[:, 0], word2vec_offset[:, 1], color='red', label='Word2Vec')
for i, word in enumerate(final_tokens):
    plt.annotate(word, (word2vec_offset[i, 0], word2vec_offset[i, 1]), color='red')

plt.title("Word Embeddings Visualization with PCA (BERT and Word2Vec)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.show()