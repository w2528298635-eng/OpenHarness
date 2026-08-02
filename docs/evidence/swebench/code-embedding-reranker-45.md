# 专业代码 Embedding 与 Cross-Encoder：45题定位评测

## 结论

RepoPilot 将通用文本向量模型替换为固定 revision 的
[`nomic-ai/CodeRankEmbed`](https://huggingface.co/nomic-ai/CodeRankEmbed)，并在词法/语义双路召回的
top-40 候选内使用固定 revision 的
[`BAAI/bge-reranker-v2-m3`](https://huggingface.co/BAAI/bge-reranker-v2-m3) 做交叉编码精排。
两路候选分数和精排分数分别做 min-max 归一化，最终按 0.5 / 0.5 加权，避免精排的一次误判完全抹掉候选生成阶段的证据。

在冻结的45题 SWE-bench Verified 开发集（10 easy / 15 medium / 20 hard）上，完整配置相对旧通用 BGE 双路检索：

- Recall@1：15.56% → 22.22%，绝对提升6.67个百分点，相对提升42.86%。
- Recall@5：28.89% → 32.22%，绝对提升3.33个百分点，相对提升11.54%。
- Hit@5：33.33% → 37.78%，绝对提升4.44个百分点，相对提升13.33%。
- MRR：0.239 → 0.307，相对提升28.68%。
- 无关上下文率：85.12% → 83.04%，下降2.08个百分点。

## 三组消融

| 配置 | Recall@1 | Recall@3 | Recall@5 | Hit@5 | MRR | 无关上下文率 |
|---|---:|---:|---:|---:|---:|---:|
| 通用 BGE 双路检索 | 15.56% | 27.78% | 28.89% | 33.33% | 0.239 | 85.12% |
| CodeRankEmbed，无精排 | 16.67% | 31.11% | 31.11% | 33.33% | 0.252 | 86.56% |
| CodeRankEmbed + 加权 Cross-Encoder | **22.22%** | **32.22%** | **32.22%** | **37.78%** | **0.307** | **83.04%** |

CodeRankEmbed 单独替换后扩大了前3/前5召回，但没有改善 Hit@5，且噪声率略升。Cross-Encoder 的主要贡献是把已经进入候选集的正确代码推到更靠前位置，并减少最终上下文噪声。相对 CodeRank-only，完整配置的 Recall@1 相对提升33.33%，MRR相对提升22.06%，无关上下文率下降3.52个百分点。

## 分难度结果

| 难度 | 任务数 | Recall@1 | Recall@5 | MRR | 无关上下文率 |
|---|---:|---:|---:|---:|---:|
| Easy | 10 | 40.00% | 60.00% | 0.500 | 82.87% |
| Medium | 15 | 26.67% | 36.67% | 0.322 | 80.57% |
| Hard | 20 | 10.00% | 15.00% | 0.200 | 84.99% |

困难题被刻意过采样：它们占本子集44.44%，而在完整500题中只占9%。因此这里更接近定位压力测试，不能把结果直接换算成官方榜单成绩。

## 配对统计

对 CodeRank-only 与完整配置的同45题差值做10,000次配对 bootstrap（seed `20260802`）：

| 指标增量 | 均值 | 95%区间 |
|---|---:|---:|
| Recall@1 | +0.0556 | [0.0000, 0.1222] |
| Recall@5 | +0.0111 | [-0.0556, 0.0667] |
| Hit@5 | +0.0444 | [-0.0444, 0.1333] |
| MRR | +0.0556 | **[0.0074, 0.1074]** |
| 无关上下文率 | -0.0352 | **[-0.0691, -0.0050]** |

MRR和噪声率区间没有跨0；Recall与Hit的区间仍跨0，因此只能称为方向性改善，不能声称已得到统计确定的召回提升。

## 固定配置

- 数据集 revision：`91aa3ed51b709be6457e12d00300a6a596d4c6a3`
- 清单逻辑 SHA-256：`9b12758c4b8967d92d07200addb94ac12abe9244b2a208917ca854493e75cde0`
- CodeRankEmbed revision：`3c4b60807d71f79b43f3c4363786d9493691f8b1`，MIT。
- bge-reranker-v2-m3 revision：`953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`，Apache-2.0。
- 词法/语义候选融合权重：0.45 / 0.55。
- 精排候选数：40；候选/精排融合权重：0.5 / 0.5；最终 top-k：12。
- 上下文字符预算：12,000；Query Planner 开；结构扩展关；严格精排开。

检查点会持久化模型名、revision、最大长度、融合权重和所有检索开关，并拒绝不同配置续写同一文件。模型和向量/分数缓存留在 `E:\RepoPilot`，不会提交权重。

## 复现命令

```powershell
openh repopilot swebench localize docs\evidence\swebench\formal-manifest.json `
  --checkpoint .openharness-swebench\coderank-reranker-45.json `
  --repository-root C:\path\to\prepared\worktrees `
  --strategy hybrid --query-planning --no-structural-expansion `
  --embedding-model nomic-ai/CodeRankEmbed `
  --embedding-revision 3c4b60807d71f79b43f3c4363786d9493691f8b1 `
  --embedding-max-seq-length 512 `
  --reranker cross_encoder `
  --reranker-model BAAI/bge-reranker-v2-m3 `
  --reranker-revision 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e `
  --reranker-max-length 512 --reranker-candidate-k 40 `
  --reranker-weight 0.5 --reranker-strict `
  --char-budget 12000 --top-k 12
```

本机运行环境为 Ryzen 7 5800H、RTX 3050 Ti Laptop GPU、PyTorch 2.11.0+cu128。Embedding 使用GPU，Cross-Encoder使用CPU。各实验的缓存冷热状态不同，所以耗时仅作操作记录，不能当作公平速度对比。

## 解释边界

这是文件/代码块定位评测，不是Agent生成补丁并通过隐藏测试的端到端修复率；45题是公开开发集，不是未见验证集。Gold patch只在排序完成后用于打分，没有进入查询、候选生成或精排输入。机器可读结果见 [`code-embedding-reranker-45-summary.json`](code-embedding-reranker-45-summary.json)。
