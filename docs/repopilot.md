# RepoPilot：从零运行代码修复 Agent

RepoPilot 是这个 fork 新增的代码修复 Agent 应用，不是 OpenHarness 上游自带功能。
OpenHarness 提供模型调用、流式事件、工具注册和工具执行等通用基础设施；
RepoPilot 在其上增加“分析—计划—修改—验证—恢复”的确定性工作流。

可以把大模型理解为会阅读和写代码的工程师，把 OpenHarness 理解为工程师使用的
电脑、工具箱和通信系统，把 RepoPilot 理解为项目经理、质量负责人和审计系统。
模型提出分析并编辑代码，但只有真实验证器有权宣布修复成功。

## 一次运行发生了什么

```text
读取 task.yaml
  → 为本次运行创建隔离 Git worktree
  → PRECHECK 运行原始测试，确认 Bug 确实存在
  → ANALYZE 检索仓库并要求模型给出带源码证据的根因
  → PLAN 要求模型给出受文件范围约束的修复计划
  → EXECUTE 只开放受限读写工具，由模型编辑代码
  → VERIFY 由调度器使用参数数组和 shell=False 运行 pytest
  → 通过则 COMPLETE；失败则在预算内进入 REPAIR 或 REPLAN
  → 每个阶段保存状态、事件、diff、验证日志和汇总
```

这不是让模型在一个长对话里自由决定所有步骤。`WorkflowRuntime` 负责阶段调度，
`TransitionPolicy` 负责确定性状态转移，OpenHarness 只在需要模型推理或编辑的阶段
执行 Agent loop。

## Windows / PowerShell 快速开始

先激活安装过 OpenHarness 的 Python 3.11 环境。在仓库根目录准备示例 Git 仓库：

```powershell
cd examples\repopilot\discount_bug
git init
git add discount.py test_discount.py
git -c user.name=RepoPilot -c user.email=repopilot@example.invalid commit -m "buggy baseline"
python -m pytest -q
cd ..\..\..
```

此时应看到测试失败。配置 OpenAI-compatible provider。不要把密钥写进 YAML 或提交
到 Git：

```powershell
$env:OPENHARNESS_MODEL = "deepseek-chat"
$env:OPENHARNESS_BASE_URL = "https://api.deepseek.com"
$env:OPENHARNESS_API_FORMAT = "openai"
$env:OPENHARNESS_OPENAI_API_KEY = $env:DEEPSEEK_API_KEY
```

如果用户主目录不可写，显式使用短路径保存 worktree 和运行产物：

```powershell
$env:OPENHARNESS_REPOPILOT_WORKTREE_ROOT = "C:\repopilot-wt"
$env:OPENHARNESS_REPOPILOT_RUN_ROOT = "C:\repopilot-runs"
```

运行：

```powershell
openh repopilot run examples\repopilot\task.example.yaml
```

真实烟测应输出类似：

```text
run_id: 20260729T175456Z-7080c6e3
phase: COMPLETE
reason: verified
worktree: <isolated-worktree>
```

本次真实运行使用 DeepSeek，开启本地检索，耗时 21.57 秒，3 次模型阶段调用，
记录 27,792 total tokens，只修改 `discount.py`，最终 `2 passed`。

## 查看、恢复和清理

```powershell
openh repopilot show <run-id> --repo examples\repopilot\discount_bug
openh repopilot report <run-id> --repo examples\repopilot\discount_bug
openh repopilot resume <run-id> --repo examples\repopilot\discount_bug
openh repopilot cleanup <run-id> --repo examples\repopilot\discount_bug
```

`cleanup` 只移除隔离 worktree，不删除运行报告；含未提交修改的 worktree 默认拒绝
删除，需要明确传入 `--force`。

## 任务 YAML

```yaml
repo_path: C:\path\to\trusted-python-repository
issue: 清楚描述能够复现的 Bug
verify_command: [python, -m, pytest, -q, tests/test_target.py]
allowed_paths: ["src/**"]
retrieval:
  enabled: true
  strategy: hybrid
  query_planning: true
  structural_expansion: true
  max_file_bytes: 200000
  max_chunk_chars: 4000
  context_char_budget: 12000
  top_k: 12
budgets:
  max_phase_calls: 8
  max_repair_attempts: 3
  max_replan_attempts: 2
  max_wall_seconds: 1800
  max_changed_files: 12
  verify_timeout_seconds: 300
```

验证命令必须是参数数组，只接受 `pytest`、`py.test` 或
`<python> -m pytest`，不经 Shell 拼接。首版只运行受信任的本地 Python 仓库，
Git worktree 是隔离副本，不是安全沙箱。

## 运行产物

默认位置：

```text
<repo>/.openharness/repopilot/runs/<run-id>/
  state.json                 可恢复状态
  events.jsonl               脱敏、有类型的生命周期事件
  analysis.json              结构化根因
  plan.json                  结构化修复计划
  context-analyze-*.json     检索选择轨迹与 Prompt 版本
  diff.patch                 实际代码改动
  verification-*.json/.log   测试结果和完整日志
  summary.json               时长、调用、token、文件和产物索引
  report.md                  人类可读报告
```

`Action` 表示模型或调度器准备执行的动作，`Observation` 是动作的真实结果。
Observation 必须回传给模型，因为模型需要它判断下一步；同时写入事件日志是为了
失败复盘、断点恢复、评测和面试时展示完整轨迹。

## 代码 RAG：规划、双路召回与结构扩展

`RepositoryIndex` 会忽略 Git、缓存、虚拟环境、二进制和超大文件；Python 文件按
AST 顶层类/函数切块，其他文本按大小切块。`QueryPlanner` 从问题描述中抽取异常名、
代码标识符和模块路径，并生成多条面向代码搜索的查询。关键词通道使用 TF-IDF 风格
词项分数和符号/路径加权；语义通道使用本地 `BAAI/bge-small-en-v1.5` 对整个仓库独立
取 top-100。两路分数归一化后按 `0.45 / 0.55` 融合，因此语义通道可以召回完全不在
关键词候选集中的代码。

融合结果随后按代码结构扩展：补入同文件定义，以及引用种子符号的其他片段。
`ContextBuilder` 再优先放入验证失败、已怀疑文件和高分片段，按字符预算截断，并保存
路径、行号、分数和选择原因。向量缓存在 E 盘 SQLite 中，后续相同代码片段无需重复
编码。若没有安装本地 embedding 环境，可以把 `strategy` 改回 `lexical`。

为了做公平消融，同一套代码支持以下环境开关：

```powershell
$env:REPOPILOT_HYBRID_RETRIEVAL = "1"      # 开启独立语义召回
$env:REPOPILOT_QUERY_PLANNING = "0"        # 关闭 Query Planner
$env:REPOPILOT_STRUCTURAL_EXPANSION = "0"  # 关闭结构扩展
```

正式评测建议使用可恢复 CLI，而不是环境变量。每个实验臂必须使用不同 checkpoint；
checkpoint 会记录配置并拒绝混用，避免把不同算法的结果拼在一起：

```powershell
openh repopilot swebench localize docs\evidence\swebench\pilot-manifest.json `
  --checkpoint .openharness-swebench\lexical-pilot3.json `
  --repository-root C:\path\to\prepared\worktrees `
  --strategy lexical --no-query-planning --no-structural-expansion

openh repopilot swebench localize docs\evidence\swebench\pilot-manifest.json `
  --checkpoint .openharness-swebench\dual-planned-structural-pilot3.json `
  --repository-root C:\path\to\prepared\worktrees `
  --strategy hybrid --query-planning --structural-expansion

openh repopilot swebench localization-report `
  .openharness-swebench\dual-planned-structural-pilot3.json `
  docs\evidence\swebench\pilot-manifest.json `
  --output .openharness-swebench\dual-planned-structural-pilot3-summary.json
```

Embedding 缓存按实际编码文本的 SHA-256 寻址，而不是按会随行号变化的 chunk ID
寻址；因此同一仓库不同 commit 中未变化的代码可以复用向量。Query Planner 的多条
查询会在一次 worker 调用中共同编码，并以各查询相似度的最大值形成语义排序，避免
重复加载模型。

三题开发集的组合结果从 Recall@5 33.33% 提升到 66.67%，MRR 从 0.333 提升到
0.444；样本很小，只能作为诊断证据，不能外推成整体成功率。详见
[三题双路检索记录](evidence/swebench/dual-retrieval-pilot3.md)。

## 评测、API 和第二工作流

```powershell
# 不调用模型：验证 10 个 fixture、补丁、测试和指标管线
openh repopilot evaluate examples\repopilot\evaluation\manifest.yaml `
  --strategy scripted --output .\evaluation-output

# 真实模型矩阵会产生 API 费用
openh repopilot evaluate examples\repopilot\evaluation\manifest.yaml `
  --strategy model_no_retrieval `
  --strategy model_with_retrieval `
  --allow-live-matrix `
  --output .\evaluation-output

# 只读、带源码引用的第二工作流
openh repopilot insight src\openharness\repopilot `
  --question "How does the workflow runtime checkpoint state?"
```

安装 API extra 后：

```powershell
pip install -e ".[api]"
openh repopilot serve --host 127.0.0.1 --port 8000
```

`POST /runs` 返回后台 operation，`GET /operations/{id}` 查询完成状态，再使用真实
run_id 查询状态、事件和产物。服务默认只允许 loopback；它是本地演示 API，没有
身份认证或分布式 worker，不能直接暴露到公网。

更多内容见：

- [分层架构与时序](repopilot-architecture.md)
- [真实评测方法与结果](repopilot-evaluation.md)
- [简历表述与面试问答](repopilot-resume-and-interview.md)
