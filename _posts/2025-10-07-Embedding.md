---
layout: post
title: "Embedding-从过去到现在(含代码实现检索任务)"
date:   2025-10-07
tags: [NLP]
comments: true
author: 炼丹怪
---

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
3. **长度归一项**：$ 1 - b + b \cdot \frac{d}{\text{avgd}} $，对文档长度进行归一化，较短文档获得更高分数

#### 2.2 BM25检索代码实现

```python
# BM25实现
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

### 3. Word2Vec

将单词转换为Embedding的网络架构，通过定义并优化辅助目标(CBOW 预测中间缺失词；Skipgram 预测相邻单词)实现。
网络训练完成后，将最后一层抛弃，**得到Embedding向量是真正的目标**
![CBOW示例](https://LLLcf.github.io/images/CBOW.png)

Word2Vec 由于训练过程中利用了上下文窗口内的信息，因此能够捕捉词语的语义关系

#### 3.1 word2vec训练代码实现
```python
import numpy as np
import math
import re
from collections import defaultdict
import jieba
from gensim.models import Word2Vec

class Word2VecRetrieval:
    def __init__(self, use_jieba=True, vector_size=100, window=2, min_count=1, workers=2):
        self.use_jieba = use_jieba
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model = None
        self.documents = []
        self.doc_vectors = []
        self.vocabulary = set()
        if self.use_jieba:
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
            words = list(jieba.cut(text))
        else:
            text = re.sub(r'[^\w\s]', ' ', text)
            words = text.split()
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
        tokenized_docs = []
        for doc in documents:
            words = self.preprocess_text(doc)
            tokenized_docs.append(words)
            self.vocabulary.update(words)
        self.model = Word2Vec(
            sentences=tokenized_docs,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=0
        )
        self.doc_vectors = self._compute_document_vectors(tokenized_docs)
        return tokenized_docs
    
    def _compute_document_vectors(self, tokenized_docs):
        doc_vectors = []
        for words in tokenized_docs:
            word_vectors = []
            for word in words:
                if word in self.model.wv:
                    word_vectors.append(self.model.wv[word])
            if len(word_vectors) > 0:
                doc_vector = np.mean(word_vectors, axis=0)
            else:
                doc_vector = np.zeros(self.vector_size)
            doc_vectors.append(doc_vector)
        return doc_vectors
    
    def _query_to_vector(self, query):
        words = self.preprocess_text(query)
        word_vectors = []
        for word in words:
            if word in self.model.wv:
                word_vectors.append(self.model.wv[word])
        if len(word_vectors) > 0:
            return np.mean(word_vectors, axis=0)
        else:
            return np.zeros(self.vector_size)
    
    def _cosine_similarity(self, vec1, vec2):
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def score(self, query, doc_index):
        if doc_index >= len(self.documents):
            return 0
        query_vector = self._query_to_vector(query)
        doc_vector = self.doc_vectors[doc_index]
        return self._cosine_similarity(query_vector, doc_vector)
    
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
也可利用训练好的word2vec模型进行检索。

### 4.基于Transformer架构的预训练嵌入模型

#### 4.1 BERT（Encoder-only架构）
BERT中的CLS token用于表示输入文本的语义结果，可利用CLS结果计算语义相似度以实现检索功能。
```python

```

#### 4.2 Qwen3_Embedding（Decoder-only架构）
采用双塔结构，适用于大规模召回阶段。该模型接收单段文本作为输入，取模型最后一层「EOS」标记对应的隐藏状态向量，作为输入文本的语义表示，生成独立的语义向量。
```python
import torch
import vllm
from vllm import LLM

class QwenEmbeddingRetrieval:
    def __init__(self, model_path, task_description=None):
        self.model_path = model_path
        self.task_description = task_description or "Given a web search query, retrieve relevant passages that answer the query"
        self.model = None
        self.documents = []
        self.document_embeddings = None
        self._initialize_model()
    
    def _initialize_model(self):
        self.model = LLM(model=self.model_path, task="embed")

    def get_detailed_instruct(self, query: str) -> str:
        return f'Instruct: {self.task_description}\nQuery:{query}'
    
    def fit(self, documents):
        self.documents = documents
        input_texts = documents
        outputs = self.model.embed(input_texts)
        self.document_embeddings = torch.tensor([o.outputs.embedding for o in outputs])
        return self.document_embeddings
    
    def score(self, query, doc_index):
        if doc_index >= len(self.documents) or self.document_embeddings is None:
            return 0
        query_embedding = self._get_query_embedding(query)
        doc_embedding = self.document_embeddings[doc_index]
        similarity = torch.dot(query_embedding, doc_embedding).item()
        return similarity
    
    def _get_query_embedding(self, query):
        instructed_query = self.get_detailed_instruct(query)
        outputs = self.model.embed([instructed_query])
        query_embedding = torch.tensor([o.outputs.embedding for o in outputs])[0]
        return query_embedding
    
    def search(self, query, top_k=None):
        if self.document_embeddings is None:
            raise ValueError("请先调用fit方法处理文档")
        query_embedding = self._get_query_embedding(query)
        scores = (query_embedding @ self.document_embeddings.T).tolist()
        results = []
        for i, score in enumerate(scores):
            results.append((i, score, self.documents[i]))
        results.sort(key=lambda x: x[1], reverse=True)
        if top_k is not None:
            return results[:top_k]
        else:
            return results
    
    def batch_search(self, queries, top_k=None):
        if self.document_embeddings is None:
            raise ValueError("请先调用fit方法处理文档")
        instructed_queries = [self.get_detailed_instruct(query) for query in queries]
        outputs = self.model.embed(instructed_queries)
        query_embeddings = torch.tensor([o.outputs.embedding for o in outputs])
        batch_scores = (query_embeddings @ self.document_embeddings.T)
        all_results = []
        for i, scores in enumerate(batch_scores.tolist()):
            results = []
            for j, score in enumerate(scores):
                results.append((j, score, self.documents[j]))
            results.sort(key=lambda x: x[1], reverse=True)
            if top_k is not None:
                all_results.append(results[:top_k])
            else:
                all_results.append(results)
        
        return all_results
```

#### 4.3 Qwen3_Reranker（Decoder-only架构）
采用单塔交叉编码结构，适用于精排阶段。该模型接收文本对（如用户查询与候选文档）作为输入，通过深度交互，利用单塔结构计算并输出两个文本的相关性得分。

```python
import torch
import math
from typing import List, Tuple, Optional, Dict
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel
from vllm.inputs.data import TokensPrompt

class QwenReranker:
    def __init__(self, model_path: str, task_description: str = None, max_length: int = 8192):
        self.model_path = model_path
        self.task_description = task_description or "Given a web search query, retrieve relevant passages that answer the query"
        self.max_length = max_length
        self.model = None
        self.tokenizer = None
        self.sampling_params = None
        self.true_token = None
        self.false_token = None
        self.suffix_tokens = None
        
        # 初始化模型和tokenizer
        self._initialize_model()
    
    def _initialize_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        number_of_gpu = torch.cuda.device_count()
        self.model = LLM(
            model=self.model_path,
            tensor_parallel_size=number_of_gpu,
            max_model_len=10000,
            enable_prefix_caching=True,
            gpu_memory_utilization=0.8
        )
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)
        self.true_token = self.tokenizer("yes", add_special_tokens=False).input_ids[0]
        self.false_token = self.tokenizer("no", add_special_tokens=False).input_ids[0]
        self.sampling_params = SamplingParams(
            temperature=0, 
            max_tokens=1,
            logprobs=20, 
            allowed_token_ids=[self.true_token, self.false_token],
        )
    
    def format_instruction(self, instruction: str, query: str, doc: str) -> List[Dict]:
        text = [
            {
                "role": "system", 
                "content": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
            },
            {
                "role": "user", 
                "content": f"<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {doc}"
            }
        ]
        return text
    
    def process_inputs(self, pairs: List[Tuple[str, str]]) -> List[TokensPrompt]:
        messages = []
        for query, doc in pairs:
            message = self.format_instruction(self.task_description, query, doc)
            messages.append(message)
        tokenized_messages = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False, enable_thinking=False
        )
        processed_messages = []
        for ele in tokenized_messages:
            truncated = ele[:self.max_length - len(self.suffix_tokens)]
            processed = truncated + self.suffix_tokens
            processed_messages.append(TokensPrompt(prompt_token_ids=processed))
        return processed_messages
    
    def compute_scores(self, inputs: List[TokensPrompt]) -> List[float]:
        outputs = self.model.generate(inputs, self.sampling_params, use_tqdm=False)
        scores = []
        for i in range(len(outputs)):
            final_logits = outputs[i].outputs[0].logprobs[-1]
            token_count = len(outputs[i].outputs[0].token_ids)
            true_logit = final_logits.get(self.true_token, -10).logprob
            false_logit = final_logits.get(self.false_token, -10).logprob
            true_score = math.exp(true_logit)
            false_score = math.exp(false_logit)
            score = true_score / (true_score + false_score) if (true_score + false_score) > 0 else 0.0
            scores.append(score)
        return scores
    
    def rerank(self, query: str, documents: List[str], top_k: Optional[int] = None) -> List[Tuple[int, float, str]]:
        pairs = [(query, doc) for doc in documents]
        inputs = self.process_inputs(pairs)
        scores = self.compute_scores(inputs)
        results = []
        for i, (score, doc) in enumerate(zip(scores, documents)):
            results.append((i, score, doc))
        results.sort(key=lambda x: x[1], reverse=True)
        if top_k is not None:
            return results[:top_k]
        else:
            return results
    
    def batch_rerank(self, queries: List[str], documents_list: List[List[str]], top_k: Optional[int] = None) -> List[List[Tuple[int, float, str]]]:
        all_results = []
        
        for query, documents in zip(queries, documents_list):
            results = self.rerank(query, documents, top_k)
            all_results.append(results)
        return all_results
    
    def score(self, query: str, document: str) -> float:
        pairs = [(query, document)]
        inputs = self.process_inputs(pairs)
        scores = self.compute_scores(inputs)
        return scores[0] if scores else 0.0
    
    def close(self):
        if hasattr(self, 'model') and self.model is not None:
            destroy_model_parallel()
            self.model = None

    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
```

#### 检索结果

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

bm25 = BM25(k1=1.5, b=0.75,use_jieba=True)
bm25.fit(doc_contents)
results_bm25 = bm25.search(query)

w2v_retrieval = Word2VecRetrieval(use_jieba=True, vector_size=100, min_count=1)
w2v_retrieval.fit(doc_contents)
results_w2v = w2v_retrieval.search(query)

model_bert = "/root/lanyun-fs/models/Bert"
bert_retrieval = ChineseBERTRetrieval(model_bert)
bert_retrieval.fit(doc_contents)
results_bert = bert_retrieval.search(query)

model_emb = "/root/lanyun-fs/models/Qwen3-Embedding-0.6B"
qwen3_emb_retrieval = QwenEmbeddingRetrieval(model_emb)
qwen3_emb_retrieval.fit(doc_contents)
results_qwen3_emb = qwen3_emb_retrieval.search(query)

model_rerank = "/root/lanyun-fs/models/Qwen3-Reranker-0.6B"
with QwenReranker(model_rerank) as reranker:
    results_rerank = reranker.rerank(query, doc_contents)

print('results_tfidf', results_tfidf)
print()
print('results_bm25', results_bm25)
print()
print('results_w2v', results_w2v)
print()
print('results_bert', results_bert)
print()
print('results_qwen3_emb', results_qwen3_emb)
print()
print('results_rerank', results_rerank)
```

```python
results_tfidf [(0, 0.2755555303106965, 'RAG（Retrieval-Augmented Generation）技术是一种结合检索和生成的混合模型。它的技术概要包括三个核心组件：检索器从大规模知识库中检索相关文档，生成器基于检索到的信息生成回答，重排模块对结果进行优化。RAG能够有效减少大语言模型的幻觉问题，提高回答的准确性和可信度。'), (1, 0.08436747205275927, 'RAG模型的技术架构主要分为两个阶段：检索阶段使用Dense Passage Retrieval或BM25算法从外部知识源获取相关信息，生成阶段将检索结果与原始问题结合，通过seq2seq模型生成最终答案。这种检索增强的生成方式显著提升了模型在知识密集型任务上的表现。'), (3, 0.01781050508009941, 'Transformer架构是当前自然语言处理领域的主流模型，它基于自注意力机制，摒弃了传统的循环和卷积结构。BERT、GPT等预训练语言模型都是基于Transformer构建的，在各种NLP任务上表现出色。'), (4, 0.011169981872058355, '知识图谱是一种结构化的语义知识库，用于描述现实世界中的实体及其关系。它通常采用三元组形式存储数据，在智能搜索、推荐系统、问答系统等应用中发挥着重要作用。'), (2, 0.004713992245323638, '深度学习是机器学习的一个分支，它基于人工神经网络，特别是深度神经网络。深度学习模型能够从大量数据中自动学习特征表示，在计算机视觉、自然语言处理等领域取得了突破性进展。')]

results_bm25 [(0, 3.6708436530427986, 'RAG（Retrieval-Augmented Generation）技术是一种结合检索和生成的混合模型。它的技术概要包括三个核心组件：检索器从大规模知识库中检索相关文档，生成器基于检索到的信息生成回答，重排模块对结果进行优化。RAG能够有效减少大语言模型的幻觉问题，提高回答的准确性和可信度。'), (1, 1.739185335384677, 'RAG模型的技术架构主要分为两个阶段：检索阶段使用Dense Passage Retrieval或BM25算法从外部知识源获取相关信息，生成阶段将检索结果与原始问题结合，通过seq2seq模型生成最终答案。这种检索增强的生成方式显著提升了模型在知识密集型任务上的表现。'), (3, 0.1491262525021976, 'Transformer架构是当前自然语言处理领域的主流模型，它基于自注意力机制，摒弃了传统的循环和卷积结构。BERT、GPT等预训练语言模型都是基于Transformer构建的，在各种NLP任务上表现出色。'), (4, 0.13261017093672978, '知识图谱是一种结构化的语义知识库，用于描述现实世界中的实体及其关系。它通常采用三元组形式存储数据，在智能搜索、推荐系统、问答系统等应用中发挥着重要作用。'), (2, 0.09537707835370463, '深度学习是机器学习的一个分支，它基于人工神经网络，特别是深度神经网络。深度学习模型能够从大量数据中自动学习特征表示，在计算机视觉、自然语言处理等领域取得了突破性进展。')]

results_w2v [(0, 0.5339111, 'RAG（Retrieval-Augmented Generation）技术是一种结合检索和生成的混合模型。它的技术概要包括三个核心组件：检索器从大规模知识库中检索相关文档，生成器基于检索到的信息生成回答，重排模块对结果进行优化。RAG能够有效减少大语言模型的幻觉问题，提高回答的准确性和可信度。'), (1, 0.30031347, 'RAG模型的技术架构主要分为两个阶段：检索阶段使用Dense Passage Retrieval或BM25算法从外部知识源获取相关信息，生成阶段将检索结果与原始问题结合，通过seq2seq模型生成最终答案。这种检索增强的生成方式显著提升了模型在知识密集型任务上的表现。'), (3, 0.2623386, 'Transformer架构是当前自然语言处理领域的主流模型，它基于自注意力机制，摒弃了传统的循环和卷积结构。BERT、GPT等预训练语言模型都是基于Transformer构建的，在各种NLP任务上表现出色。'), (2, 0.11050085, '深度学习是机器学习的一个分支，它基于人工神经网络，特别是深度神经网络。深度学习模型能够从大量数据中自动学习特征表示，在计算机视觉、自然语言处理等领域取得了突破性进展。'), (4, 0.101617895, '知识图谱是一种结构化的语义知识库，用于描述现实世界中的实体及其关系。它通常采用三元组形式存储数据，在智能搜索、推荐系统、问答系统等应用中发挥着重要作用。')]

results_bert [(0, 0.7009080052375793, 'RAG（Retrieval-Augmented Generation）技术是一种结合检索和生成的混合模型。它的技术概要包括三个核心组件：检索器从大规模知识库中检索相关文档，生成器基于检索到的信息生成回答，重排模块对结果进行优化。RAG能够有效减少大语言模型的幻觉问题，提高回答的准确性和可信度。'), (3, 0.6701851487159729, 'Transformer架构是当前自然语言处理领域的主流模型，它基于自注意力机制，摒弃了传统的循环和卷积结构。BERT、GPT等预训练语言模型都是基于Transformer构建的，在各种NLP任务上表现出色。'), (4, 0.6655139923095703, '知识图谱是一种结构化的语义知识库，用于描述现实世界中的实体及其关系。它通常采用三元组形式存储数据，在智能搜索、推荐系统、问答系统等应用中发挥着重要作用。'), (1, 0.6538002490997314, 'RAG模型的技术架构主要分为两个阶段：检索阶段使用Dense Passage Retrieval或BM25算法从外部知识源获取相关信息，生成阶段将检索结果与原始问题结合，通过seq2seq模型生成最终答案。这种检索增强的生成方式显著提升了模型在知识密集型任务上的表现。'), (2, 0.6036232113838196, '深度学习是机器学习的一个分支，它基于人工神经网络，特别是深度神经网络。深度学习模型能够从大量数据中自动学习特征表示，在计算机视觉、自然语言处理等领域取得了突破性进展。')]

results_qwen3_emb [(0, 0.8492569923400879, 'RAG（Retrieval-Augmented Generation）技术是一种结合检索和生成的混合模型。它的技术概要包括三个核心组件：检索器从大规模知识库中检索相关文档，生成器基于检索到的信息生成回答，重排模块对结果进行优化。RAG能够有效减少大语言模型的幻觉问题，提高回答的准确性和可信度。'), (1, 0.7184826135635376, 'RAG模型的技术架构主要分为两个阶段：检索阶段使用Dense Passage Retrieval或BM25算法从外部知识源获取相关信息，生成阶段将检索结果与原始问题结合，通过seq2seq模型生成最终答案。这种检索增强的生成方式显著提升了模型在知识密集型任务上的表现。'), (4, 0.2813783884048462, '知识图谱是一种结构化的语义知识库，用于描述现实世界中的实体及其关系。它通常采用三元组形式存储数据，在智能搜索、推荐系统、问答系统等应用中发挥着重要作用。'), (3, 0.2789081335067749, 'Transformer架构是当前自然语言处理领域的主流模型，它基于自注意力机制，摒弃了传统的循环和卷积结构。BERT、GPT等预训练语言模型都是基于Transformer构建的，在各种NLP任务上表现出色。'), (2, 0.272296279668808, '深度学习是机器学习的一个分支，它基于人工神经网络，特别是深度神经网络。深度学习模型能够从大量数据中自动学习特征表示，在计算机视觉、自然语言处理等领域取得了突破性进展。')]

results_rerank [(0, 0.99999627336116, 'RAG（Retrieval-Augmented Generation）技术是一种结合检索和生成的混合模型。它的技术概要包括三个核心组件：检索器从大规模知识库中检索相关文档，生成器基于检索到的信息生成回答，重排模块对结果进行优化。RAG能够有效减少大语言模型的幻觉问题，提高回答的准确性和可信度。'), (1, 0.9999724643121783, 'RAG模型的技术架构主要分为两个阶段：检索阶段使用Dense Passage Retrieval或BM25算法从外部知识源获取相关信息，生成阶段将检索结果与原始问题结合，通过seq2seq模型生成最终答案。这种检索增强的生成方式显著提升了模型在知识密集型任务上的表现。'), (3, 0.012431653620200801, 'Transformer架构是当前自然语言处理领域的主流模型，它基于自注意力机制，摒弃了传统的循环和卷积结构。BERT、GPT等预训练语言模型都是基于Transformer构建的，在各种NLP任务上表现出色。'), (4, 0.009708474753363424, '知识图谱是一种结构化的语义知识库，用于描述现实世界中的实体及其关系。它通常采用三元组形式存储数据，在智能搜索、推荐系统、问答系统等应用中发挥着重要作用。'), (2, 0.00831578015177299, '深度学习是机器学习的一个分支，它基于人工神经网络，特别是深度神经网络。深度学习模型能够从大量数据中自动学习特征表示，在计算机视觉、自然语言处理等领域取得了突破性进展。')]
```