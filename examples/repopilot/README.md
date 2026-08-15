# RepoPilot examples

## 单任务 smoke

```powershell
cd discount_bug
git init
git add discount.py test_discount.py
git -c user.name=RepoPilot -c user.email=repopilot@example.invalid commit -m "buggy baseline"
python -m pytest -q
cd ..\..\..

openh repopilot run examples\repopilot\task.example.yaml
```

`task.example.yaml` 默认开启本地检索。模型只允许修改 `discount.py`，最终成功必须由
`python -m pytest -q test_discount.py` 验证。

## 10 任务评测

```powershell
openh repopilot evaluate examples\repopilot\evaluation\manifest.yaml `
  --strategy scripted `
  --output .\evaluation-output
```

scripted 策略不调用模型，只验证所有 baseline、golden patch、Git、pytest 和指标
管线。真实对照会产生 provider 费用：

```powershell
openh repopilot evaluate examples\repopilot\evaluation\manifest.yaml `
  --strategy model_no_retrieval `
  --strategy model_with_retrieval `
  --allow-live-matrix `
  --output .\evaluation-output
```

评测器会为每个 case/strategy/repetition 创建独立 Git 仓库和 worktree，保留 run_id
和失败结果，并生成时间戳 JSON/Markdown 报告。
