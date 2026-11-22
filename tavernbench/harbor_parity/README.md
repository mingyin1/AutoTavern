# Harbor parity runs

This branch adds what is needed to compare TavernBench on its original harness with the
[Harbor adapter](https://github.com/mingyin1/adapters/tree/tavernbench-adapter/src/tavernbench)
([harbor-index/adapters#34](https://github.com/harbor-index/adapters/pull/34)) under the same conditions.
The benchmark's tasks, labels and scorer are unchanged: every file under `tavernbench/data/` and
`tavernbench/tavernbench/score.py` is identical to tag `v1.0.0`, the release the adapter pins.

## Why a new runner

The released runners ask a model once per task. `run_claude` already uses the Claude Code CLI, but as a single
call without tools and with the benchmark's system prompt. Harbor runs Claude Code as an agent: it works in a
container, may use its tools, and writes the answer to `/app/answer.json`. Comparing a tool-less call with an
agent would measure the agent setting, not the harness.

`tavernbench/run_agent.py` therefore runs, on the original side, exactly what Harbor's `claude-code` agent runs
(Harbor 0.27.0):

| | Harbor | This runner |
|---|---|---|
| Instruction | the task's `instruction.md` | the same file, read from `--harbor-tasks` |
| Environment | the task image (`python:3.13-slim` plus a few packages) | the same image definition plus Claude Code (`Dockerfile.agent`) |
| Agent | `claude-code` at a pinned version, `disable_web_search=true` | the same version; the same setup and launch scripts, each run with `docker exec ... bash -c` behind Harbor's pipefail prefix, with the same environment and CLI flags (`--cli-flags` takes the flags Harbor builds, e.g. with `reasoning_effort`) |
| Limits | 600 s, 1 CPU, 2 GB | the same |
| Answer | `/app/answer.json` | the same |
| Failed runs | a timeout or a non-zero exit is graded on the answer left; a run that ended in a model-API error (rate limit, overload, lost connection, authentication) gets no valid reward and counts as 0 | the same: before scoring, each failed run is classified with Harbor's own classifier and those rows get an `error` field, which the scorer counts as 0 |
| Scoring | `tests/score.py`, a port of `tavernbench.score` | `tavernbench.score`, unchanged |

Both sides leave out the scorer's per-call checks (token budget, cut-off calls); they describe one model call and
have no meaning for an agent run.

## Reproduce

The parity runs are made by a GitHub Actions workflow on the `parity-ci` branch of
[mingyin1/adapters](https://github.com/mingyin1/adapters/tree/parity-ci) (see `parity/README.md` there). Before
any run it checks, with Harbor's own code, that this runner sends the same scripts and environment as Harbor's
`claude-code` agent and that both sides give the agent the same instruction for every task. It then runs both
sides on GitHub-hosted runners, applies Harbor's error policy to this side, scores it with `tavernbench.score`,
and publishes the results and a summary to the `parity-results` branch.

By hand: generate the Harbor tasks (in a checkout of the adapters repository),

```bash
cd src/tavernbench && uv run tavernbench --output-dir ../../datasets/tavernbench && cd ../..
```

then build the agent image and run one pass over the 368-task evaluation set (in this repository, from
`tavernbench/`):

```bash
export CLAUDE_CODE_VERSION=<pinned version>      # the version Harbor installs with --ak version=...
docker build -f harbor_parity/Dockerfile.agent \
  --build-arg CLAUDE_CODE_VERSION=$CLAUDE_CODE_VERSION \
  -t tavernbench-agent:$CLAUDE_CODE_VERSION harbor_parity

export ANTHROPIC_API_KEY=...                     # and ANTHROPIC_BASE_URL for a custom endpoint
python -m tavernbench.run_agent --set data/eval \
  --harbor-tasks /path/to/adapters/datasets/tavernbench \
  --image tavernbench-agent:$CLAUDE_CODE_VERSION --model <model id> \
  --out answers/original_run1/answers.jsonl --jobs 4 --keep-logs
```

Apply Harbor's error policy (`parity/error_policy.py` on the adapters fork's `parity-ci` branch, run with Harbor's
Python) and score:

```bash
"$(uv tool dir)/harbor/bin/python" /path/to/adapters/parity/error_policy.py \
  --run-agent tavernbench/run_agent.py \
  --answers answers/original_run1/answers.jsonl --out answers/original_run1/graded_answers.jsonl
python -X utf8 -m tavernbench.score --set data/eval --answers answers/original_run1/graded_answers.jsonl
```

Repeat with `original_run2` and `original_run3`. The printed `accuracy` is the score of the run.

The Harbor side, from the adapters repository:

```bash
harbor run -p datasets/tavernbench -a claude-code -m "anthropic/<model id>" \
  --ak version=$CLAUDE_CODE_VERSION --ak disable_web_search=true
```
