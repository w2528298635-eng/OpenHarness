# RepoPilot 专业代码 Embedding 与 Cross-Encoder 设计

## 目标

将 RepoPilot 的通用 `BAAI/bge-small-en-v1.5` 替换为 MIT 许可、面向代码检索的
`nomic-ai/CodeRankEmbed`，并在现有词法/语义双路融合后增加本地
`BAAI/bge-reranker-v2-m3` Cross-Encoder 精排。使用同一冻结45题 SWE-bench
Verified 开发清单量化专业向量模型和精排器的独立贡献。

## 固定模型

- Embedding：`nomic-ai/CodeRankEmbed`，revision
  `3c4b60807d71f79b43f3c4363786d9493691f8b1`，MIT。
- Reranker：`BAAI/bge-reranker-v2-m3`，revision
  `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`，Apache-2.0。
- 模型文件、向量和精排分数只保存在 `E:\RepoPilot\models`，不提交权重。

## 数据流

```text
Issue
  → QueryPlanner 多查询
  → lexical top-100 + CodeRankEmbed dense top-100
  → 归一化加权融合 top-40
  → bge-reranker-v2-m3 对 (Issue, candidate) 交叉编码
  → top-12
  → ContextBuilder 按字符预算注入 Prompt
```

Embedding 输入包含文件路径、符号、代码类型和完整代码块；查询使用模型要求的
`Represent this query for searching relevant code: ` 前缀。Cross-Encoder 输入使用
原始 Issue 和同一代码表示。精排器只允许重新排列融合候选，不能产生新文件。

## 接口与配置

- `LocalEmbeddingEncoder` 接受 model、revision、query_prefix 和 max_seq_length；缓存
  identity 必须包含这些配置，禁止复用旧 BGE 向量。
- 新增 `LocalCrossEncoderReranker.rank(query, matches, top_k)`；worker 返回排序、缓存
  命中和失败详情。
- `RetrievalConfig` 和定位评测 checkpoint 记录模型与精排配置；不同实验配置禁止写入
  同一 checkpoint。
- Reranker 默认只在显式启用时运行。模型不可用、超时或输出非法时，正常 Agent 路径
  回退到融合排序并记录原因；正式评测默认禁止静默回退，以免生成伪结果。

## 评测

冻结 `docs/evidence/swebench/formal-manifest.json`，至少比较：

1. 已发布通用 BGE 双路召回基线。
2. CodeRankEmbed 双路召回，无 Cross-Encoder。
3. CodeRankEmbed 双路召回 + Cross-Encoder。

主要指标为 Recall@1/3/5、Hit@5、MRR、无关上下文率和平均检索耗时。Gold patch
只在检索完成后用于评分。45题均完成且 checkpoint 配置一致才允许发布结果。

## 边界

- 本阶段不训练或微调模型，不引入7B本地 Reranker，不做端到端补丁生成评测。
- 45题已经参与开发，结果属于开发证据而非未见测试集结论。
- 新模型可能提高或降低指标；报告必须保留负面结果，不按结果删除任务。
