---
layout: post
title: "2.Embedding微调-对齐下游检索(含代码实现)"
date: 2025-10-08
tags: [NLP]
comments: true
author: 炼丹怪
---


- **问题分析**：通用预训练嵌入（Embedding）模型在垂直领域场景中性能常受限制，主要原因在于其缺乏特定领域的专属知识，导致语义理解与任务适配不足
- **解决策略**：通过模型微调手段，实现预训练模型的领域适配，有效提升模型在垂直任务中的表现
- **核心机制**：嵌入模型微调的核心在于对模型嵌入层知识进行针对性调整，优化其表征能力，使其更贴合领域数据的语义分布与特征规律

---
#### 1.微调数据构造
[数据集](https://challenge.xfyun.cn/topic/info?type=open-vertical-retrieval&option=stsj)

##### 数据样本结构
每条训练样本包含以下组成部分：
- **1条query**：待查询的问题或文本
- **1条正样本**：与query相关的正确答案文本
- **3条负样本**：与query不相关的错误答案文本

##### 负样本构造策略

###### BM25相似度计算
- 使用BM25算法计算query与所有候选文本的相似度分数
- 基于相似度分数对负样本进行分类筛选

###### 负样本分类

**🔴 难负样本（Hard Negatives）**
- **筛选标准**：BM25分数较高的负样本
- **具体范围**：排序前10名中标记为不相关的文本
- **特点分析**：
  - 与查询的文本相似度高
  - 表面语义接近但实际不相关
  - 模型容易混淆，属于"难例"
- **训练价值**：强迫模型学习更精细的区分特征

**🟢 易分负样本（Easy Negatives）**
- **筛选标准**：BM25分数较低的负样本
- **具体范围**：排序后10名中标记为不相关的文本
- **特点分析**：
  - 与查询的文本相似度低
  - 表面语义差异大
  - 模型容易区分，属于"易例"
- **训练价值**：保证基础的分类边界学习

```python
process_data = []
target_neg_number = 3
for data in tqdm(train_data):
    relevant, un_relevant = [], []
    all_content = []
    document_labels = []  
    query = data['query']
    content_private = data['content_private']
    content_public = data['content_public']
    for content in content_private:
        content_text = content['content']
        if not content_text:
            continue
        all_content.append(content_text)
        document_labels.append(not content['is_relevant'])
        if content['is_relevant']:
            relevant.append(content_text)
        else:
            un_relevant.append(content_text)
    for content in content_public:
        content_text = content['content']
        if not content_text:
            continue
        all_content.append(content_text)
        document_labels.append(not content['is_relevant'])
        if content['is_relevant']:
            relevant.append(content_text)
        else:
            un_relevant.append(content_text)
    
    tokenized_corpus = [list(jieba.cut(doc)) for doc in all_content]
    tokenized_query = list(jieba.cut(query))
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(tokenized_query)
    sorted_indices = np.argsort(bm25_scores)[::-1]

    # 提取难负样本和易分负样本
    hard_negatives = []
    easy_negatives = []
    for idx in sorted_indices[:10]:
        if document_labels[idx]:
            hard_negatives.append({
                "content": all_content[idx],
                "bm25_score": float(bm25_scores[idx]),
            })
    end_idx = max(0, len(sorted_indices) - 10)
    for idx in sorted_indices[end_idx:]:
        if document_labels[idx]:
            easy_negatives.append({
                "content": all_content[idx],
                "bm25_score": float(bm25_scores[idx]),
            })

    for pos in relevant:
        selected_negatives = []
        # 情况1：难负+易分负样本总数不足，用普通负样本补充
        if len(hard_negatives) + len(easy_negatives) < target_neg_number:
            selected_negatives = [h['content'] for h in hard_negatives] + [e['content'] for e in easy_negatives]
            need_more = target_neg_number - len(selected_negatives)
            if need_more > 0 and un_relevant:
                available = [n for n in un_relevant if n not in selected_negatives]
                if available:
                    selected_negatives += random.sample(available, min(need_more, len(available)))
        # 情况2：负样本总数足够，优先组合难负和易分负样本
        else:
            if len(hard_negatives) >= 1:
                selected_negatives.append(random.choice(hard_negatives)['content'])
                remaining = target_neg_number - 1
                if len(easy_negatives) >= remaining:
                    selected_negatives += [e['content'] for e in random.sample(easy_negatives, remaining)]
                else:
                    selected_negatives += [h['content'] for h in random.sample(hard_negatives, remaining)]
            else:
                selected_negatives = [e['content'] for e in random.sample(easy_negatives, target_neg_number)]
        if len(selected_negatives) < target_neg_number and un_relevant:
            need_more = target_neg_number - len(selected_negatives)
            available = [n for n in un_relevant if n not in selected_negatives]
            if available:
                selected_negatives += random.sample(available, min(need_more, len(available)))
        process_data.append({'query': query, 'positive': pos, 'negative': selected_negatives})

with open(f'train_data_with_neg_num_{target_neg_number}.json', 'w', encoding='utf-8') as f:
    json.dump(process_data, f, ensure_ascii=False, indent=4)
print("\n难负样本构建完成！")
```
---

---
#### 2.模型原始和微调代码

0.6B模型代码

```python
# qwen3_emb
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
            # results.append((i, score, self.documents[i]))
            results.append((i, score))
        # results.sort(key=lambda x: x[1], reverse=True)
        # if top_k is not None:
        #     return results[:top_k]
        # else:
        #     return results

        return results
```
8B模型代码
```python
class ClientEmbeddingRetrieval:
    def __init__(self, model, api_key, base_url, task_description=None):
        self.task_description = task_description or "Given a web search query, retrieve relevant passages that answer the query"
        self.documents = []
        self.document_embeddings = None
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self._initialize_model()
    
    def _initialize_model(self):
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def get_detailed_instruct(self, query: str) -> str:
        return f'Instruct: {self.task_description}\nQuery:{query}'
        
    def fit(self, documents):
        self.documents = documents
        input_texts = documents
        response = self.client.embeddings.create(input=input_texts,model=self.model)
        self.document_embeddings = torch.tensor([res.embedding for res in response.data])
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
        response = self.client.embeddings.create(input=[instructed_query],model=self.model)
        query_embedding = torch.tensor([res.embedding for res in response.data])[0]
        return query_embedding
    
    def search(self, query, top_k=None):
        if self.document_embeddings is None:
            raise ValueError("请先调用fit方法处理文档")
        query_embedding = self._get_query_embedding(query)
        scores = (query_embedding @ self.document_embeddings.T).tolist()
        results = []
        for i, score in enumerate(scores):
            # results.append((i, score, self.documents[i]))
            results.append((i, score))
        # results.sort(key=lambda x: x[1], reverse=True)
        # if top_k is not None:
        #     return results[:top_k]
        # else:
        #     return results

        return results
```

```python
# 
class MultiNegTripletDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=512, num_negatives_per_sample=3):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.num_negatives_per_sample = num_negatives_per_sample

        self.filtered_data = []
        for item in self.data:
            if len(item['negative']) >= self.num_negatives_per_sample:
                selected_negatives = random.sample(item['negative'], self.num_negatives_per_sample)
                self.filtered_data.append({
                    'query': item['query'],
                    'positive': item['positive'],
                    'negative': selected_negatives
                })
        print(f"过滤后保留 {len(self.filtered_data)}/{len(self.data)} 条数据（确保每个样本有足够的负样本）")

    def __len__(self):
        return len(self.filtered_data)

    def __getitem__(self, idx):
        item = self.filtered_data[idx]
        query = item['query']
        positive = item['positive']
        negatives = item['negative']
        
        query_encoding = self.tokenizer(
            query,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        positive_encoding = self.tokenizer(
            positive,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        negative_encodings = []
        for neg in negatives:
            neg_encoding = self.tokenizer(
                neg,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            negative_encodings.append({
                'input_ids': neg_encoding['input_ids'].squeeze(0),
                'attention_mask': neg_encoding['attention_mask'].squeeze(0)
            })
        
        result = {
            'query_input_ids': query_encoding['input_ids'].squeeze(0),
            'query_attention_mask': query_encoding['attention_mask'].squeeze(0),
            'positive_input_ids': positive_encoding['input_ids'].squeeze(0),
            'positive_attention_mask': positive_encoding['attention_mask'].squeeze(0),
        }
        
        for i, neg_encoding in enumerate(negative_encodings):
            result[f'negative_{i}_input_ids'] = neg_encoding['input_ids']
            result[f'negative_{i}_attention_mask'] = neg_encoding['attention_mask']
        return result

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def prepare_model(model_path, use_lora=True, use_gradient_checkpointing=True):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModel.from_pretrained(
        model_path,
        device_map='auto',
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("已启用梯度检查点")

    def qwen_pooling(model_output, attention_mask):
        return model_output.last_hidden_state[:, 0, :]
    
    if use_lora:
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model, tokenizer, qwen_pooling

class MultiNegTripletLoss(torch.nn.Module):
    def __init__(self, margin=1.0, temperature=0.05):
        super(MultiNegTripletLoss, self).__init__()
        self.margin = margin
        self.temperature = temperature
        
    def forward(self, query_emb, positive_emb, negative_embs):
        query_emb = torch.nn.functional.normalize(query_emb, p=2, dim=1)
        positive_emb = torch.nn.functional.normalize(positive_emb, p=2, dim=1)
        pos_sim = torch.sum(query_emb * positive_emb, dim=1) / self.temperature
        neg_sims = []
        for neg_emb in negative_embs:
            neg_emb = torch.nn.functional.normalize(neg_emb, p=2, dim=1)
            neg_sim = torch.sum(query_emb * neg_emb, dim=1) / self.temperature
            neg_sims.append(neg_sim.unsqueeze(1))
        all_neg_sims = torch.cat(neg_sims, dim=1)
        labels = torch.zeros(query_emb.size(0), dtype=torch.long).to(query_emb.device)
        logits = torch.cat([pos_sim.unsqueeze(1), all_neg_sims], dim=1)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        return loss

class MultiNegDataCollator:
    def __init__(self, num_negatives_per_sample=3):
        self.num_negatives_per_sample = num_negatives_per_sample
        
    def __call__(self, batch):
        result = {}
        for key in batch[0].keys():
            if key.startswith('query_') or key.startswith('positive_'):
                result[key] = torch.stack([item[key] for item in batch])
            elif key.startswith('negative_'):
                result[key] = torch.stack([item[key] for item in batch])
        return result

class MultiNegEmbeddingTrainer(Trainer):
    def __init__(self, *args, mean_pooling_fn=None, num_negatives_per_sample=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.mean_pooling_fn = mean_pooling_fn
        self.loss_fn = MultiNegTripletLoss()
        self.num_negatives_per_sample = num_negatives_per_sample

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        model.train()
        query_outputs = model(
            input_ids=inputs['query_input_ids'],
            attention_mask=inputs['query_attention_mask'],
            output_hidden_states=True
        )
        positive_outputs = model(
            input_ids=inputs['positive_input_ids'],
            attention_mask=inputs['positive_attention_mask'],
            output_hidden_states=True
        )
        query_emb = self.mean_pooling_fn(query_outputs, inputs['query_attention_mask'])
        positive_emb = self.mean_pooling_fn(positive_outputs, inputs['positive_attention_mask'])
        negative_embs = []
        for i in range(self.num_negatives_per_sample):
            neg_outputs = model(
                input_ids=inputs[f'negative_{i}_input_ids'],
                attention_mask=inputs[f'negative_{i}_attention_mask'],
                output_hidden_states=True
            )
            neg_emb = self.mean_pooling_fn(neg_outputs, inputs[f'negative_{i}_attention_mask'])
            negative_embs.append(neg_emb)
        loss = self.loss_fn(query_emb, positive_emb, negative_embs)
        return (loss, (query_emb, positive_emb, negative_embs)) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        return (loss, None, None)

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
        )
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=num_training_steps,
        )

```
---

---
#### 3.模型微调
采取lora微调
```python
def main():
    model_path = "/root/lanyun-fs/models/Qwen3-Embedding-0.6B"
    data_path = 'train_data_with_neg_num_3.json'
    output_dir = './qwen3_embedding_model_multi_neg_lora'
    batch_size = 16
    num_epochs = 3
    learning_rate = 1e-4
    num_negatives_per_sample = 3
    
    print("加载数据...")
    data = load_data(data_path)
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

    print("准备模型...")
    model, tokenizer, pooling_fn = prepare_model(
        model_path, 
        use_lora=True, 
        use_gradient_checkpointing=True
    )

    print("创建数据集...")
    train_dataset = MultiNegTripletDataset(
        train_data, 
        tokenizer, 
        max_len=1024,
        num_negatives_per_sample=num_negatives_per_sample
    )

    val_dataset = MultiNegTripletDataset(
        val_data, 
        tokenizer, 
        max_len=1024,
        num_negatives_per_sample=num_negatives_per_sample
    )

    if len(train_dataset) == 0:
        raise ValueError("训练数据集为空，请检查数据过滤条件")
    if len(val_dataset) == 0:
        raise ValueError("验证数据集为空，请检查数据过滤条件")

    data_collator = MultiNegDataCollator(num_negatives_per_sample=num_negatives_per_sample)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=int(0.1 * num_epochs * len(train_dataset) / batch_size),
        learning_rate=learning_rate,
        logging_dir='./logs',
        logging_steps=2,
        eval_strategy='steps',
        eval_steps=50,
        save_strategy='steps',
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        fp16=True,
        gradient_accumulation_steps=2,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        optim="adamw_torch",
        ddp_find_unused_parameters=False,
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        lr_scheduler_type="cosine",
        disable_tqdm=False,
        prediction_loss_only=True,
    )

    print("创建训练器...")
    trainer = MultiNegEmbeddingTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        mean_pooling_fn=pooling_fn,
        num_negatives_per_sample=num_negatives_per_sample
    )

    print("开始训练...")
    trainer.train()

    print("保存模型...")
    model.save_pretrained(f"{output_dir}/final_model")
    tokenizer.save_pretrained(f"{output_dir}/final_tokenizer")
    print(f"模型已保存至 {output_dir}")
```
---

---
#### 4.权重合并
```python
base_model_path = "/root/lanyun-fs/models/Qwen3-Embedding-0.6B"
model = AutoModel.from_pretrained(base_model_path)
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
lora_path = './qwen3_embedding_model_multi_neg_lora/final_model'
model = PeftModel.from_pretrained(model, lora_path)
model = model.merge_and_unload()
model.save_pretrained("merged_model/Qwen3-Embedding-0.6B")
tokenizer.save_pretrained("merged_model/Qwen3-Embedding-0.6B")
```
---

---
#### 5.消融比较
评估代码
```python
with open('eval_data.json', 'r', encoding='utf-8') as f:
    eval_dataset = json.load(f)
    
model_emb_lora = "./merged_model/Qwen3-Embedding-0.6B"
qwen3_emb_lora_retrieval = QwenEmbeddingRetrieval(model_emb_lora)

model_emb = "/root/lanyun-fs/models/Qwen3-Embedding-0.6B"
qwen3_emb_retrieval = QwenEmbeddingRetrieval(model_emb)

scores_lora = []
scores_base = []
labels = []
for e_data in tqdm(eval_dataset):

    query = e_data['query']
    doc_contents = [doc["content"] for doc in e_data['documents']]
    doc_labels = [doc["label"] for doc in e_data['documents']]

    qwen3_emb_lora_retrieval.fit(doc_contents)
    results_qwen3_emb_lora = qwen3_emb_lora_retrieval.search(query)
    score_lora = [res[1] for res in results_qwen3_emb_lora]

    qwen3_emb_retrieval.fit(doc_contents)
    results_qwen3_emb = qwen3_emb_retrieval.search(query)
    score = [res[1] for res in results_qwen3_emb]
    
    scores_lora.extend(score_lora)
    scores_base.extend(score)    
    labels.extend(doc_labels)

df = pd.DataFrame({
    'doc_label': labels,     
    'scores_lora': scores_lora,
    'scores_base': scores_base,   
})
```
---

---
#### 7.训练和评估结果

| 学习率（Learning Rate）变化 | 梯度范数（Grad Norm）变化 |
|-----------------------------|---------------------------|
| ![学习率变化](https://LLLcf.github.io/images/embedding_finetune/learning_rate.png) <br> **学习率趋势** | ![梯度范数变化](https://LLLcf.github.io/images/embedding_finetune/grad_norm.png) <br> **核梯度变化** |
| 训练损失（Train Loss）变化  | 验证损失（Eval Loss）变化  |
| ![训练损失变化](https://LLLcf.github.io/images/embedding_finetune/train_loss.png) <br> **训练Loss 变化** | ![验证损失变化](https://LLLcf.github.io/images/embedding_finetune/eval_loss.png) <br> **测试Loss 变化** |


| 阈值 (Threshold) | 0.6B模型 F1分数 | 8B模型 F1分数 | lora_0.6B模型 F1分数 |
|:----------------:|:---------------:|:------------:|:-------------------:|
|       0.4        |     0.6210      |    0.6246    |       0.5879        |
|       0.5        |     0.6480      |    0.6548    |       0.6150        |
|       0.6        |     0.5950      |    0.6075    |       0.6516        |
|       0.7        |     0.3748      |    0.3625    |       0.5627        |

- 1.模型预测结果与阈值强相关
- 2.LoRA微调改变了模型的预测分布，使其在高阈值下表现更优，可能更适合对 “预测可靠性” 要求高的场景

---