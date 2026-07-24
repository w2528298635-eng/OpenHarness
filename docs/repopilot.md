# RepoPilot：新手运行与源码导读

RepoPilot 是 OpenHarness 上的一层“确定性项目经理”。OpenHarness 原本让模型在 ReAct 循环中自行判断下一步；RepoPilot 则把修 Bug 拆成固定阶段，并把是否成功的裁决权交给真实的 `pytest`。

## 它解决了什么问题

模型擅长读代码、提出假设和修改文件，但它可能误判“已经修好”。RepoPilot 的外层状态机负责流程，内层 OpenHarness Agent 只负责当前阶段：

```text
PRECHECK → ANALYZE → PLAN → EXECUTE → VERIFY
                                      ├─ 通过 → COMPLETE
                                      └─ 失败 → REPAIR / REPLAN → VERIFY
```

- `PRECHECK`：先确认原始 Bug 确实能复现。
- `ANALYZE`：只读代码，输出带证据的结构化根因。
- `PLAN`：输出文件范围明确的修复计划。
- `EXECUTE`：只开放读取和编辑工具，不开放 `bash`。
- `VERIFY`：调度器用 `shell=False` 执行用户给定的 pytest 参数数组。
- `REPAIR/REPLAN`：验证失败后，在预算内修补或换假设。

面试亮点不是“再包一层 Prompt”，而是把概率性的模型决策与确定性的工程控制分开。

## Windows / PowerShell 快速运行

先激活安装过 OpenHarness 的 Python 3.11 Conda 环境，然后准备示例仓库：

```powershell
cd examples\repopilot\discount_bug
git init
git add .
git -c user.name=RepoPilot -c user.email=repopilot@example.com commit -m "buggy baseline"
python -m pytest -q
cd ..\..\..
```

测试应失败，证明 Bug 可复现。配置 OpenAI-compatible provider 后运行：

```powershell
openh repopilot run examples\repopilot\task.example.yaml
```

终端会打印 `run_id`、最终状态和隔离工作树路径。原仓库不会被直接修改。

```powershell
openh repopilot show <run-id> --repo examples\repopilot\discount_bug
openh repopilot report <run-id> --repo examples\repopilot\discount_bug
openh repopilot resume <run-id> --repo examples\repopilot\discount_bug
openh repopilot benchmark examples\repopilot\benchmark.example.yaml
```

当前 benchmark 只报告实际 RepoPilot 运行数据，不虚构基线或提升比例。

## 任务 YAML

```yaml
repo_path: C:\path\to\your\git-repo
issue: 清楚描述能复现的 Python Bug
verify_command: [python, -m, pytest, -q, tests/test_target.py]
allowed_paths: ["src/**", "tests/**"]
```

验证命令必须是参数数组，不是 Shell 字符串。首版只接受 `pytest`、`py.test` 或 `<python> -m pytest`，不会自动安装依赖。

## 运行产物

每次运行保存在原仓库：

```text
.openharness/repopilot/runs/<run-id>/
  state.json
  events.jsonl
  analysis.json
  plan.json
  diff.patch
  verification-<attempt>.json
  verification-<attempt>.log
  report.md
```

`Action` 是模型或调度器准备做的动作，`Observation` 是动作的真实结果。二者都进入 `events.jsonl`，所以面试时可以展示一条完整、可追踪的 Agent 轨迹。

## 关键源码

- `repopilot/scheduler.py`：外层状态机。
- `repopilot/phase_runner.py`：为每个模型阶段创建新的 OpenHarness Runtime。
- `repopilot/tools.py`：阶段工具白名单。
- `repopilot/verifier.py`：安全运行 pytest 并分类失败。
- `repopilot/policy.py`：纯函数式转移与预算规则。
- `repopilot/store.py`：原子检查点和事件日志。
- `repopilot/workspace.py`：隔离工作树、diff 与路径边界。

## 面试表达

“我没有重写 OpenHarness 的 ReAct 内核，而是在外层实现 Plan–Execute–Verify–Repair 状态机。模型负责非确定性的代码理解与编辑，调度器负责确定性的状态转移、预算、安全边界和 pytest 裁决。每个阶段使用新的 QueryEngine 和不同工具白名单，跨阶段只传 Pydantic 校验后的结构化状态，因此比让一个长对话自行决定是否完成更容易恢复、测试和审计。”

## 首版限制

仅支持受信任的本地 Python 仓库；不使用 Docker；不自动装依赖；不自动提交、合并或推送；不支持 GitHub PR/CI；不做多 Agent；benchmark 当前不包含原生 ReAct baseline。
