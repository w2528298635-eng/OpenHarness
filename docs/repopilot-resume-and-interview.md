# RepoPilot 简历与面试材料

## 保守版简历

**RepoPilot — 基于 OpenHarness 的可恢复代码修复 Agent**

- 基于 OpenHarness QueryEngine 和工具注册机制，实现
  PRECHECK–ANALYZE–PLAN–EXECUTE–VERIFY–REPAIR/REPLAN 状态工作流，将模型推理
  与确定性验证、预算和安全策略分离。
- 设计 Git worktree 隔离、阶段工具白名单、路径级修改范围、pytest 验证、失败分类、
  原子 checkpoint 和脱敏事件日志，支持运行恢复与产物追踪。
- 实现 AST/文本切块、可解释词法检索、上下文预算和版本化 Prompt，并以 10 个
  可复现 Python Bug 对照有/无检索策略。
- 抽出共享应用服务和 FastAPI/CLI 适配层，并复用工作流、检索和事件基础设施实现
  只读 Repository Insight 第二工作流。
- RepoPilot 测试 `118 passed`；一次 DeepSeek smoke 和 20 次模型评测运行均保存
  run_id、diff、验证日志和 token 记录。

## 强结果版简历

在确认面试官能看到仓库和评测限制时，可使用：

- 构建可恢复代码修复 Agent 平台，在 10 个独立小型 Python 缺陷任务上完成
  DeepSeek 有/无检索各 10 次端到端运行，两组均 10/10 pytest 验证通过且修改范围
  100% 合规；同时明确记录单次重复和小任务天花板限制。
- 将一次性 scheduler 重构为可复用 WorkflowRuntime、PhaseExecutor、
  CompositeVerifier、FailurePolicy 和 typed event/checkpoint 体系，复用于修复与
  只读仓库洞察两条工作流。

不要写“生产级”“提升成功率”“节省 token”或“多 Agent 系统”。当前证据不支持这些
说法。

## 60 秒项目介绍

“RepoPilot 是我在 OpenHarness fork 上实现的代码修复 Agent。OpenHarness 继续负责
模型、流式 ReAct 和工具执行，我没有重复造这一层；我新增的是确定性工作流。
系统先用 pytest 复现 Bug，再让模型分阶段做根因分析、计划和受限编辑，最终是否成功
只由验证器决定。每次运行在 Git worktree 中完成，有路径白名单、预算、失败分类、
checkpoint 和脱敏事件。为了研究上下文，我又实现了 AST 切块、词法检索和可解释的
context trace，并用同一套 10 个任务对比有无检索。结果两组都 10/10，因此我没有
宣称 RAG 提升，只能说明链路可复现、范围合规；有检索还略多 token。最后我把运行
能力抽成 service，接了 CLI、FastAPI，并复用工作流做了只读 Repository Insight。”

## 高频问答

### 1. OpenHarness 和 RepoPilot 分别做什么？

OpenHarness 是通用 Agent harness，负责 provider、QueryEngine、流式消息、
ToolRegistry 和工具执行。RepoPilot 是本 fork 新增的领域应用，负责修复阶段、
状态转移、验证、预算、worktree、检索、持久化和评测。入口分别看
`phase_runner.py` 和 `scheduler.py`。

### 2. 什么叫状态转移？

状态表示系统当前处于哪个阶段，转移是“根据真实结果从一个阶段进入下一个阶段”。
例如 PRECHECK 的 pytest 失败证明 Bug 可复现，转到 ANALYZE；VERIFY 通过转到
COMPLETE，失败则按分类和预算转到 REPAIR、REPLAN 或 FAILED。规则在 `policy.py`。

### 3. 模型怎样选择工具，代码怎样路由？

OpenHarness 把工具名、说明和 JSON Schema 发给模型。模型返回 function/tool call，
QueryEngine 用工具名在 ToolRegistry 查找实现，校验参数后执行，再把结果作为
Observation 加入消息历史。RepoPilot 的 `ScopedToolRegistry` 会根据阶段移除不允许
的工具和越界路径。

### 4. 为什么 Observation 必须写回历史？

模型只看到消息。工具真实读到了什么、编辑是否成功、发生了什么错误，如果不作为
Observation 回传，模型就无法基于新事实继续推理，只能猜测。RepoPilot 还把它写入
events，供恢复、复盘和评测。

### 5. 为什么权限不能只写在 Prompt？

Prompt 是给概率模型的文字说明，可能被误解、忽略或被输入诱导。安全边界必须由
代码强制执行：阶段工具白名单、参数校验、allowed paths、无 shell 验证和预算终止。
Prompt 负责引导，代码负责阻止。

### 6. 为什么每个阶段使用新的 runtime？

阶段职责和工具权限不同。独立 runtime 能保证 ANALYZE 只读、PLAN 无工具、
EXECUTE 才可编辑；跨阶段只传递 Pydantic 校验后的 AnalysisResult/RepairPlan，
降低一个长会话中隐式上下文污染。代价是重复 Prompt 和 token 开销。

### 7. Pydantic 数据模型有什么价值？

它把“应该长什么样”变成代码约束。例如 RepairPlan 必须包含 hypothesis、steps、
expected_files 和 expected_behavior。模型返回 JSON 后先校验，错误数据不能直接
进入状态机；同时可生成 JSON Schema 给 function calling 和文档使用。

### 8. 失败怎样恢复？

验证器先把输出归类为 assertion、collection、dependency、timeout、scope 等，
FailurePolicy 再结合当前阶段、是否重复和剩余预算选择 retry、repair、replan 或
stop。重复 action/diff 和墙钟、token、次数预算防止死循环。

### 9. Git worktree 解决什么？

它让同一 Git 仓库同时拥有另一个分支的真实工作目录，共享对象库但文件互相隔离。
模型只修改 worktree，原仓库保持不变；我们可以检查 diff、保留失败现场并显式清理。
它不是安全沙箱，恶意代码仍可能访问系统资源。

### 10. 这个 RAG 怎样工作？

先按 AST 顶层符号或文本大小切块，忽略缓存、二进制和超大文件；查询使用 TF-IDF
风格词项分数，加符号和路径 bonus。ContextBuilder 先放失败输出和怀疑文件，再放
高分片段直到字符预算，并记录路径、行号、分数和原因。它是可解释本地检索，不是
向量数据库。

### 11. 为什么评测不能证明 RAG 有效？

两组模型在 10 个小任务上都 10/10，存在天花板效应；只有 1 次重复。有检索组还多
约 2.7% total tokens，中位耗时差异只有约 0.11 秒。因此只能证明实现和测量管线
可用，不能得出成功率、速度或成本改善结论。

### 12. API 为什么返回 operation 而不是立即返回 run_id？

run_id 在 scheduler 创建 worktree 后才生成。HTTP 提交应快速返回 `202`，所以先
返回 operation_id，后台运行完成后再关联真实 run_id。取消是协作式的，在安全阶段
边界检查，而不是强杀正在写文件的协程。

### 13. 如何证明框架可复用？

代码修复和 Repository Insight 都使用 WorkflowRuntime、RunStore、typed events、
RepositoryIndex 和 ContextBuilder。第二工作流是
SCAN–RETRIEVE–ANALYZE–REPORT，只读且输出源码引用，证明基础设施没有硬编码为
discount 修复。

### 14. 如果继续做，优先改什么？

增加真实仓库任务和 3 次以上重复；把检索升级为混合符号图/embedding/reranker；
加入 Docker 或受限执行环境；把 FastAPI 后台任务迁移到持久队列；增加人工审批、
PR/CI 集成；构造能稳定触发 REPAIR/REPLAN 的困难评测。
