import random
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import gensim.downloader as api

# 下载并加载预训练的Word2Vec模型
word2vec_model = api.load('word2vec-google-news-300')

# 读取文本
text = ''
with open("D:\\Codespace\\morphology\\1706.03762v7_no_custom_names.txt", 'r') as f:
    text = f.read()

# 文本预处理
tokens = text.split()

# 随机采样部分单词及其嵌入表示
sample_size = 200  # 选择合适的采样数量
sample_indices = random.sample(range(len(tokens)), sample_size)

sample_tokens = [tokens[i] for i in sample_indices]

# 获取Word2Vec嵌入表示
clean_sample_tokens = []
clean_sample_embeddings = []
for token in sample_tokens:
    if token.lower() in word2vec_model:
        clean_sample_tokens.append(token)
        clean_sample_embeddings.append(word2vec_model[token.lower()])

clean_sample_embeddings = np.array(clean_sample_embeddings)

# 使用PCA降维到2维
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(clean_sample_embeddings)

# KNN聚类分析
knn = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(X_reduced)
distances, indices = knn.kneighbors(X_reduced)

# 可视化
plt.figure(figsize=(15, 15))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1])

# 绘制KNN聚类关系
for i in range(len(X_reduced)):
    for j in indices[i]:
        plt.plot([X_reduced[i, 0], X_reduced[j, 0]], [X_reduced[i, 1], X_reduced[j, 1]], 'k-', alpha=0.3)

# 标注单词
for i, word in enumerate(clean_sample_tokens):
    plt.annotate(word, (X_reduced[i, 0], X_reduced[i, 1]))

plt.title("Word Embeddings Visualization with PCA (Word2Vec Sampled and Cleaned)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()
