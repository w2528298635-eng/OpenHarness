# RepoPilot 评测

## 方法

评测集包含 10 个独立小型 Python Bug：边界、异常输入、分支、字符串归一化、
import wiring、多文件数据流、输入突变、回归、需要重新定位的文本处理和修改范围。
每个案例包括：

- 可提交的 buggy 源码和测试；
- task YAML、问题描述、验证命令和 allowed paths；
- 只对 scripted 策略可见的 golden patch；
- baseline 必须失败、golden patch 后必须通过的完整性检查。

三种策略各运行 1 次：

1. `scripted`：应用 golden patch，验证 fixture、Git、pytest、指标和报告管线；
   它不是模型质量分数。
2. `model_no_retrieval`：DeepSeek `deepseek-chat`，Prompt v2，不注入本地检索。
3. `model_with_retrieval`：同一模型、任务和预算，ANALYZE/REPLAN 注入词法检索上下文。

运行环境为 Windows、Python 3.11。成功必须同时满足：最终状态 `COMPLETE`、真实
pytest 通过、修改文件符合 allowed paths。失败不会从分母中删除。

## 2026-07-30 实测结果

| 策略 | 成功 | 成功率 | 中位耗时 | total tokens | 范围合规 |
|---|---:|---:|---:|---:|---:|
| scripted | 10/10 | 100% | 1.91 s | 0 | 100% |
| model_no_retrieval | 10/10 | 100% | 20.09 s | 253,468 | 100% |
| model_with_retrieval | 10/10 | 100% | 19.98 s | 260,358 | 100% |

另有一次开启检索的 discount 烟测：`COMPLETE`，21.57 秒，3 次模型阶段调用，
27,792 total tokens，只修改目标文件，最终 `2 passed`。

原始运行 ID 和逐案例 token 记录在
[`repopilot-evaluation-2026-07-30.json`](evidence/repopilot-evaluation-2026-07-30.json)。
没有保存或提交 API key。

## 如何解释结果

这些结果证明在当前小型任务集上，端到端链路、权限范围、验证和报告均可复现。
它们不证明 100% 的通用代码修复能力：

- 只有 10 个手工构造的小任务，每种策略只有 1 次重复；
- 两个模型策略都达到 10/10，任务集出现天花板效应，无法体现 RAG 成功率差异；
- 有检索组 total tokens 多 6,890（约 2.7%），中位耗时只少约 0.11 秒，不能据此
  宣称 RAG 更快或更省；
- 本次旧运行只可靠保存 total tokens，未保存所有请求的 input/output/cache 分项；
  代码已补上后续运行的分项累计；
- 未固定 provider 服务端模型快照、随机种子或温度，结果会受服务端变化影响；
- 未进行价格估算，因为没有在报告中绑定当时的官方价格快照和完整 token 分项。

更有意义的下一轮应增加真实仓库任务、至少 3 次重复、不同上下文规模，以及能够
触发 VERIFY→REPAIR 和 REPLAN 的困难案例。

## 全仓验证

- RepoPilot：安装 API extra 后 `118 passed`。
- OpenHarness 关键回归：`19 passed`。
- 上游全量：`1230 passed, 12 skipped, 31 failed`。31 项均在 RepoPilot 之外，
  主要来自 Windows 环境执行 POSIX `printf`、`/tmp`、`fcntl`、符号链接权限，
  以及沙箱禁止写用户主目录。将 OpenHarness 数据目录指向可写位置后，3 个权限类
  用例复核通过。没有修改无关上游模块来掩盖这些失败。
