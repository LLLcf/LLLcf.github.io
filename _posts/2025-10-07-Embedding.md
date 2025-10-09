---
layout: post
title: "Embedding解读"
date:   2025-10-07
tags: [NLP]
comments: true
author: 炼丹怪
---

### Embedding解读

**Embedding原理**：将文本/图片等数据抽象为向量表示，从高维向量空间中对文本/图片等信息进行语义表示。

> *一个好的Embedding模型应该包含*
> 1. **良好的语义表示**（能捕捉词与词之间的语义关系）
> 2. **嵌入维度合适**（dim过高导致过拟合、向量空间稀疏、计算成本上升；dim过低导致没有区分度、泛化能力受限）

---

### 1. TF-IDF（基于文档中词频的统计方法）
字词的重要性随着它在文件中出现的次数成正比增加，但同时会随着它在语料库中出现的频率成反比下降。

#### 1.1 核心公式

**词频（TF, Term Frequency）**  
$$\text{TF}(t, d) = \frac{\text{count}(t, d)}{|d|}$$

**逆文档频率（IDF, Inverse Document Frequency）**  
$$\text{IDF}(t, D) = \log\left(\frac{|D|}{|\{d \in D : t \in d\}| + 1}\right)$$

**TF-IDF**  
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

---

#### 1.2 检索代码实现

```python
# TF-IDF实现
import math
import re
from collections import Counter, defaultdict
import jieba

class TFIDF:
    def __init__(self, use_jieba=True, custom_dict=None):
        """
        TF-IDF 实现，类BM25风格
        """
        self.vocabulary = set()
        self.idf = {}
        self.doc_count = 0
        self.doc_term_count = defaultdict(int)
        self.use_jieba = use_jieba
        self.documents = []
        self.doc_vectors = []
        if self.use_jieba:
            if custom_dict:
                jieba.load_userdict(custom_dict)
            jieba.add_word('RAG')
            jieba.add_word('Retrieval')
            jieba.add_word('Augmented')
            jieba.add_word('Generation')
            jieba.add_word('Passage')
            jieba.add_word('seq2seq')
            jieba.add_word('BERT')
            jieba.add_word('GPT')
            jieba.add_word('Transformer')
            jieba.add_word('NLP')
            
    def preprocess_text(self, text):
        text = text.lower()
        if self.use_jieba:
            return self._mixed_segmentation(text)
        else:
            return self._english_segmentation(text)
    
    def _english_segmentation(self, text):
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.split()
        words = [word for word in words if word.strip()]
        return words
    
    def _mixed_segmentation(self, text):
        words = list(jieba.cut(text))
        processed_words = []
        for word in words:
            word = word.strip()
            if not word:
                continue
            if self._is_english_word(word):
                processed_words.extend(self._process_english_word(word))
            else:
                processed_words.append(word)
        
        return processed_words
    
    def _is_english_word(self, word):
        return bool(re.match(r'^[a-z0-9]+$', word))
    
    def _process_english_word(self, word):
        if '-' in word and len(word) > 1:
            parts = word.split('-')
            return [part for part in parts if len(part) > 1]
        if word.endswith('es') and len(word) > 2:
            base_word = word[:-2]
            if self._is_valid_english_word(base_word):
                return [base_word]
        elif word.endswith('s') and len(word) > 1:
            base_word = word[:-1]
            if self._is_valid_english_word(base_word):
                return [base_word]
        return [word]
    
    def _is_valid_english_word(self, word):
        return len(word) > 1 and word.isalpha()
    
    def fit(self, documents):
        self.documents = documents
        self.doc_count = len(documents)
        self.vocabulary = set()

        doc_words_list = []
        for doc in documents:
            words = self.preprocess_text(doc)
            doc_words_list.append(words)
            unique_words = set(words)
            self.vocabulary.update(unique_words)
            for word in unique_words:
                self.doc_term_count[word] += 1

        for word in self.vocabulary:
            self.idf[word] = math.log(self.doc_count / (self.doc_term_count[word] + 1))

        self.doc_vectors = self._compute_document_vectors(documents)
        
        return doc_words_list
    
    def _compute_document_vectors(self, documents):
        tfidf_matrix = []
        for doc in documents:
            words = self.preprocess_text(doc)
            tf = self.compute_tf(words)
            
            tfidf_vector = {}
            for word in self.vocabulary:
                tf_value = tf.get(word, 0)
                idf_value = self.idf.get(word, 0)
                tfidf_vector[word] = tf_value * idf_value
            tfidf_matrix.append(tfidf_vector)
        
        return tfidf_matrix
    
    def compute_tf(self, words):
        total_words = len(words)
        if total_words == 0:
            return {}
        word_count = Counter(words)
        tf = {}
        for word, count in word_count.items():
            tf[word] = count / total_words
        return tf
    
    def transform(self, documents):
        if not self.idf:
            raise ValueError("请先调用fit方法训练模型")
        return self._compute_document_vectors(documents)
    
    def get_feature_names(self):
        return sorted(list(self.vocabulary))
    
    def transform_to_dense_matrix(self, tfidf_matrix=None):
        if tfidf_matrix is None:
            tfidf_matrix = self.doc_vectors
        feature_names = self.get_feature_names()
        dense_matrix = []
        for vector in tfidf_matrix:
            row = [vector.get(word, 0) for word in feature_names]
            dense_matrix.append(row)
            
        return dense_matrix, feature_names
    
    def _cosine_similarity(self, vec1, vec2):
        if len(vec1) != len(vec2):
            raise ValueError("向量维度不一致")
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)
    
    def score(self, query, doc_index):
        if doc_index >= len(self.documents):
            return 0
        query_vector = self._query_to_vector(query)
        doc_vector = self._doc_to_vector(doc_index)
        return self._cosine_similarity(query_vector, doc_vector)
    
    def _query_to_vector(self, query):
        query_tfidf = self.transform([query])[0]
        feature_names = self.get_feature_names()
        return [query_tfidf.get(word, 0) for word in feature_names]
    
    def _doc_to_vector(self, doc_index):
        feature_names = self.get_feature_names()
        doc_vector = self.doc_vectors[doc_index]
        return [doc_vector.get(word, 0) for word in feature_names]
    
    def search(self, query, top_k=None):
        scores = []
        for i in range(len(self.documents)):
            score = self.score(query, i)
            scores.append((i, score, self.documents[i]))
        # 按得分降序排序
        scores.sort(key=lambda x: x[1], reverse=True)
        if top_k is not None:
            return scores[:top_k]
        else:
            return scores
```

#### 1.3 基于TF-IDF的检索应用

### 1.4 示例检索
```python
documents = [
    # 包含RAG技术概要的文档
    {
        "content": "RAG（Retrieval-Augmented Generation）技术是一种结合检索和生成的混合模型。它的技术概要包括三个核心组件：检索器从大规模知识库中检索相关文档，生成器基于检索到的信息生成回答，重排模块对结果进行优化。RAG能够有效减少大语言模型的幻觉问题，提高回答的准确性和可信度。",
        "contains_rag": True,
        "description": "详细描述RAG技术概要"
    },
    {
        "content": "RAG模型的技术架构主要分为两个阶段：检索阶段使用Dense Passage Retrieval或BM25算法从外部知识源获取相关信息，生成阶段将检索结果与原始问题结合，通过seq2seq模型生成最终答案。这种检索增强的生成方式显著提升了模型在知识密集型任务上的表现。",
        "contains_rag": True,
        "description": "描述RAG技术架构"
    },

    # 不包含RAG技术概要的文档
    {
        "content": "深度学习是机器学习的一个分支，它基于人工神经网络，特别是深度神经网络。深度学习模型能够从大量数据中自动学习特征表示，在计算机视觉、自然语言处理等领域取得了突破性进展。",
        "contains_rag": False,
        "description": "关于深度学习的介绍"
    },
    {
        "content": "Transformer架构是当前自然语言处理领域的主流模型，它基于自注意力机制，摒弃了传统的循环和卷积结构。BERT、GPT等预训练语言模型都是基于Transformer构建的，在各种NLP任务上表现出色。",
        "contains_rag": False,
        "description": "关于Transformer架构的介绍"
    },
    {
        "content": "知识图谱是一种结构化的语义知识库，用于描述现实世界中的实体及其关系。它通常采用三元组形式存储数据，在智能搜索、推荐系统、问答系统等应用中发挥着重要作用。",
        "contains_rag": False,
        "description": "关于知识图谱的介绍"
    }
]
query = "RAG的技术概要"

doc_contents = [doc["content"] for doc in documents]
doc_labels = [f"文档{i+1} ({'包含RAG' if doc['contains_rag'] else '不包含RAG'})" for i, doc in enumerate(documents)]

tfidf = TFIDF(use_jieba=True)
tfidf.fit(doc_contents)
results_tfidf = tfidf.search(query)
```
### 1.5 检索结果
```python
results_tfidf [(0, 0.2755555303106965, 'RAG（Retrieval-Augmented Generation）技术是一种结合检索和生成的混合模型。它的技术概要包括三个核心组件：检索器从大规模知识库中检索相关文档，生成器基于检索到的信息生成回答，重排模块对结果进行优化。RAG能够有效减少大语言模型的幻觉问题，提高回答的准确性和可信度。'), (1, 0.08436747205275927, 'RAG模型的技术架构主要分为两个阶段：检索阶段使用Dense Passage Retrieval或BM25算法从外部知识源获取相关信息，生成阶段将检索结果与原始问题结合，通过seq2seq模型生成最终答案。这种检索增强的生成方式显著提升了模型在知识密集型任务上的表现。'), (3, 0.01781050508009941, 'Transformer架构是当前自然语言处理领域的主流模型，它基于自注意力机制，摒弃了传统的循环和卷积结构。BERT、GPT等预训练语言模型都是基于Transformer构建的，在各种NLP任务上表现出色。'), (4, 0.011169981872058355, '知识图谱是一种结构化的语义知识库，用于描述现实世界中的实体及其关系。它通常采用三元组形式存储数据，在智能搜索、推荐系统、问答系统等应用中发挥着重要作用。'), (2, 0.004713992245323638, '深度学习是机器学习的一个分支，它基于人工神经网络，特别是深度神经网络。深度学习模型能够从大量数据中自动学习特征表示，在计算机视觉、自然语言处理等领域取得了突破性进展。')]
```

### 2. BM25算法
(基于TF-IDF算法,同时引入了文档的长度信息来计算文档D和查询Q之间的相关性)

#### 2.1 BM25算法公式详解

BM25（Best Match 25）是一种广泛用于信息检索的排名函数，其公式如下：

$$
\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot \left(1 - b + b \cdot \frac{d}{\text{avgd}}\right)}
$$

##### 公式参数说明

- $ q $：查询（query），由若干词项 $ t $ 组成
- $ d $：文档（document）
- $ f(t, d) $：词项 $ t $ 在文档 $ d $ 中的出现频次（term frequency）
- $ d $：文档 $ d $ 的长度（通常以词数计）
- $ avg_d $：语料库中所有文档的平均长度
- $ k_1 $ 和 $ b $：可调超参数，通常 $ k_1 \in [1.2, 2.0] $，$ b = 0.75 $
- $ \text{IDF}(t) $：词项 $ t $ 的逆文档频率

##### IDF计算公式

$$
\text{IDF}(t) = \log \left( \frac{N - n(t) + 0.5}{n(t) + 0.5} + 1 \right)
$$

其中：

- $$ N $$：语料库中文档总数
- $ n(t) $：包含词项 $ t $ 的文档数量

该 IDF 公式通过平滑处理（+0.5）避免除零错误，并确保 IDF 值非负。

##### 公式组成部分分析

1. **IDF项**：衡量词项 $ t $ 的重要性，文档频率越低，IDF值越高
2. **频率项**：$ \frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1} $，当词频增加时，得分增长但会饱和
3. **长度归一项**：$ 1 - b + b \cdot \frac{|d|}{\text{avg\_len\_d}} $，对文档长度进行归一化，较短文档获得更高分数


#### 2.2 BM25检索代码实现

```python
class BM25:
    def __init__(self, k1=1.5, b=0.75, use_jieba=True, custom_dict=None):

        self.k1 = k1
        self.b = b
        self.documents = []
        self.doc_lengths = []
        self.avg_doc_length = 0
        self.use_jieba = use_jieba
        self.doc_freqs = defaultdict(int)
        self.term_freqs = []
        self.vocabulary = set()
        if self.use_jieba:
            if custom_dict:
                jieba.load_userdict(custom_dict)
            jieba.add_word('RAG')
            jieba.add_word('Retrieval')
            jieba.add_word('Augmented')
            jieba.add_word('Generation')
            jieba.add_word('Passage')
            jieba.add_word('seq2seq')
            jieba.add_word('BERT')
            jieba.add_word('GPT')
            jieba.add_word('Transformer')
            jieba.add_word('NLP')
            
    def preprocess_text(self, text):
        text = text.lower()
        words = list(jieba.cut(text))
        processed_words = []
        for word in words:
            word = word.strip()
            if not word:
                continue
            if re.match(r'^[a-z0-9\u4e00-\u9fff]+$', word):
                processed_words.append(word)
        return processed_words
    
    def fit(self, documents):

        self.documents = documents
        self.doc_lengths = []
        self.term_freqs = []
        for doc in documents:
            words = self.preprocess_text(doc)
            self.doc_lengths.append(len(words))
            term_freq = Counter(words)
            self.term_freqs.append(term_freq)
            for word in set(words):
                self.vocabulary.add(word)
                self.doc_freqs[word] += 1

        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0
        self.idf = {}
        N = len(documents)
        for word in self.vocabulary:
            # BM25的IDF公式
            self.idf[word] = math.log((N - self.doc_freqs[word] + 0.5) / (self.doc_freqs[word] + 0.5) + 1)
    
    def score(self, query, doc_index):

        if doc_index >= len(self.documents):
            return 0
        words = self.preprocess_text(query)
        score = 0
        doc_length = self.doc_lengths[doc_index]
        term_freq = self.term_freqs[doc_index]
        for word in words:
            if word not in self.vocabulary:
                continue
            f = term_freq.get(word, 0)
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
            if denominator > 0:
                score += self.idf[word] * (numerator / denominator)
        return score
    
    def search(self, query, top_k=None):

        scores = []
        for i in range(len(self.documents)):
            score = self.score(query, i)
            scores.append((i, score, self.documents[i]))
        scores.sort(key=lambda x: x[1], reverse=True)
        if top_k is not None:
            return scores[:top_k]
        else:
            return scores
```
#### 2.3 示例检索
```python
bm25 = BM25(k1=1.5, b=0.75,use_jieba=True)
bm25.fit(doc_contents)
results_bm25 = bm25.search(query)
```

#### 2.4 检索结果
```python
results_bm25 [(0, 3.6708436530427986, 'RAG（Retrieval-Augmented Generation）技术是一种结合检索和生成的混合模型。它的技术概要包括三个核心组件：检索器从大规模知识库中检索相关文档，生成器基于检索到的信息生成回答，重排模块对结果进行优化。RAG能够有效减少大语言模型的幻觉问题，提高回答的准确性和可信度。'), (1, 1.739185335384677, 'RAG模型的技术架构主要分为两个阶段：检索阶段使用Dense Passage Retrieval或BM25算法从外部知识源获取相关信息，生成阶段将检索结果与原始问题结合，通过seq2seq模型生成最终答案。这种检索增强的生成方式显著提升了模型在知识密集型任务上的表现。'), (3, 0.1491262525021976, 'Transformer架构是当前自然语言处理领域的主流模型，它基于自注意力机制，摒弃了传统的循环和卷积结构。BERT、GPT等预训练语言模型都是基于Transformer构建的，在各种NLP任务上表现出色。'), (4, 0.13261017093672978, '知识图谱是一种结构化的语义知识库，用于描述现实世界中的实体及其关系。它通常采用三元组形式存储数据，在智能搜索、推荐系统、问答系统等应用中发挥着重要作用。'), (2, 0.09537707835370463, '深度学习是机器学习的一个分支，它基于人工神经网络，特别是深度神经网络。深度学习模型能够从大量数据中自动学习特征表示，在计算机视觉、自然语言处理等领域取得了突破性进展。')]
```

