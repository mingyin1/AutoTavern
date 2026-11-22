"""Answer a TavernBench set with Claude Code running as an agent in a container.

    python -m tavernbench.run_agent --set data/eval \
        --harbor-tasks /path/to/adapters/datasets/tavernbench \
        --image tavernbench-agent:<claude-code version> --model <model id> \
        --out answers/<run name>/answers.jsonl [--jobs 4]

This runner exists for the Harbor parity experiments (see ../harbor_parity/README.md).
`run_claude` asks Claude once, without tools. Harbor instead runs Claude Code as an
agent that works in a container and writes its answer to /app/answer.json. To compare
the two harnesses under the same conditions, this runner does on the original side what
Harbor's `claude-code` agent does:

* one fresh container per task, from an image that holds the Harbor task image plus
  Claude Code at a pinned version (harbor_parity/Dockerfile.agent);
* the instruction is the text of the Harbor task's instruction.md, read from
  --harbor-tasks, so both sides give the agent the same bytes;
* the same command line and environment as Harbor's claude-code agent with
  `disable_web_search=true` (Harbor 0.27.0, harbor/agents/installed/claude_code.py);
* the reply is the content of /app/answer.json after the agent exits.

The output rows have the fields `tavernbench.score` reads (`id`, `response`), so the
unchanged scorer grades them:

    python -m tavernbench.score --set data/eval --answers answers/<run name>/answers.jsonl

Rows carry no `output_tokens` or `finish_reason`: the per-call checks of the scorer
(token budget, cut-off calls) describe a single model call and do not apply to an agent
run, on either side. A missing answer file gives `response: null`, which scores 0.

Credentials are read from the environment (ANTHROPIC_API_KEY, optional ANTHROPIC_BASE_URL)
and passed to the container by name, so they never appear on a command line. Resumable:
ids already in --out are skipped.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

# What Harbor's claude-code agent runs inside the task container. $TB_INSTRUCTION holds
# the instruction; it is piped to `claude --print` exactly as Harbor pipes it.
AGENT_COMMAND = (
    "mkdir -p $CLAUDE_CONFIG_DIR/debug $CLAUDE_CONFIG_DIR/projects/-app "
    "$CLAUDE_CONFIG_DIR/shell-snapshots $CLAUDE_CONFIG_DIR/statsig "
    "$CLAUDE_CONFIG_DIR/todos $CLAUDE_CONFIG_DIR/skills && "
    'export PATH="$HOME/.local/bin:$PATH"; '
    'printf "%s" "$TB_INSTRUCTION" | '
    "claude --verbose --output-format=stream-json "
    "--disallowedTools WebSearch,WebFetch --permission-mode=bypassPermissions "
    "--print 2>&1 | tee /logs/agent/claude-code.txt"
)

# With a custom endpoint Harbor points every model alias at the requested model.
MODEL_ALIAS_VARS = (
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "ANTHROPIC_DEFAULT_FABLE_MODEL",
    "CLAUDE_CODE_SUBAGENT_MODEL",
)


def container_env(model: str, instruction: str) -> dict[str, str]:
    """Environment of the agent process, as Harbor's claude-code agent sets it."""
    env = {
        "ANTHROPIC_MODEL": model,
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "IS_SANDBOX": "1",
        "CLAUDE_CONFIG_DIR": "/logs/agent/sessions",
        "TB_INSTRUCTION": instruction,
    }
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL"):
        if os.environ.get(name):
            env[name] = os.environ[name]
    if "ANTHROPIC_BASE_URL" in env:
        for name in MODEL_ALIAS_VARS:
            env[name] = os.environ.get(name, model)
    return env


def read_instruction(harbor_tasks: Path, item_id: str) -> str:
    path = harbor_tasks / item_id.lower() / "instruction.md"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found; generate the Harbor tasks first and pass their directory"
        )
    return path.read_text(encoding="utf-8")


def last_result_event(log: Path) -> dict:
    """The final `result` event of Claude Code's stream-json output, if any."""
    found: dict = {}
    if log.is_file():
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("{") and '"type":"result"' in line.replace(" ", ""):
                try:
                    found = json.loads(line)
                except json.JSONDecodeError:
                    pass
    return found


def ask(item: dict, args: argparse.Namespace) -> dict:
    instruction = read_instruction(args.harbor_tasks, item["id"])
    work = Path(tempfile.mkdtemp(prefix=f"tavernbench-{item['id'].lower()}-", dir=args.work_dir))
    app, logs = work / "app", work / "logs"
    app.mkdir()
    logs.mkdir()
    name = f"tavernbench-{item['id'].lower()}-{uuid.uuid4().hex[:8]}"
    env = container_env(args.model, instruction)

    cmd = [args.docker, "run", "--rm", "--name", name, "--workdir", "/app"]
    cmd += ["--cpus", str(args.cpus), "--memory", f"{args.memory_mb}m"]
    cmd += ["--volume", f"{app}:/app", "--volume", f"{logs}:/logs/agent"]
    for key in env:
        cmd += ["--env", key]  # value comes from the environment below, not the command line
    cmd += [args.image, "bash", "-c", AGENT_COMMAND]

    t0 = time.time()
    timed_out = False
    try:
        proc = subprocess.run(
            cmd,
            env={**os.environ, **env},
            capture_output=True,
            text=True,
            timeout=args.timeout,
            check=False,
        )
        exit_code = proc.returncode
    except subprocess.TimeoutExpired:
        timed_out, exit_code = True, None
        subprocess.run([args.docker, "kill", name], capture_output=True, text=True, check=False)

    answer = app / "answer.json"
    response = answer.read_text(encoding="utf-8", errors="replace") if answer.is_file() else None
    result = last_result_event(logs / "claude-code.txt")
    row = {
        "id": item["id"],
        "model": args.model,
        "mode": "agent",
        "response": response,
        "agent_exit_code": exit_code,
        "agent_timed_out": timed_out,
        "num_turns": result.get("num_turns"),
        "cost_usd": result.get("total_cost_usd"),
        "latency_s": round(time.time() - t0, 2),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if args.keep_logs:
        dest = args.out.parent / "logs" / item["id"]
        shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(logs, dest, ignore=shutil.ignore_patterns("sessions"))
    shutil.rmtree(work, ignore_errors=True)
    return row


def main():
    ap = argparse.ArgumentParser(
        description="Answer a TavernBench set with Claude Code as an agent"
    )
    ap.add_argument("--set", required=True, help="TavernBench set directory, e.g. data/eval")
    ap.add_argument(
        "--harbor-tasks", required=True, type=Path, help="generated Harbor task directory"
    )
    ap.add_argument(
        "--image", required=True, help="image built from harbor_parity/Dockerfile.agent"
    )
    ap.add_argument("--model", required=True, help="model id, the same one Harbor is given")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument(
        "--timeout", type=int, default=600, help="agent time limit in seconds (Harbor task: 600)"
    )
    ap.add_argument("--cpus", type=float, default=1, help="container CPUs (Harbor task: 1)")
    ap.add_argument(
        "--memory-mb", type=int, default=2048, help="container memory (Harbor task: 2048)"
    )
    ap.add_argument("--only", default="", help="comma-separated scenarios, e.g. R1,B3")
    ap.add_argument("--ids", default="", help="comma-separated task ids, e.g. R1-000,B3-002")
    ap.add_argument(
        "--limit", type=int, default=None, help="answer only the first N selected tasks"
    )
    ap.add_argument("--keep-logs", action="store_true", help="keep each agent's log next to --out")
    ap.add_argument("--work-dir", default=None, help="parent of the per-task scratch directories")
    ap.add_argument("--docker", default="docker", help="container CLI to use")
    a = ap.parse_args()

    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        ap.error("set ANTHROPIC_API_KEY (and ANTHROPIC_BASE_URL for a custom endpoint) first")

    with open(Path(a.set) / "prompts.jsonl", encoding="utf-8") as source:
        items = [json.loads(line) for line in source]
    only = [s for s in a.only.split(",") if s]
    ids = {s for s in a.ids.split(",") if s}
    if only:
        items = [d for d in items if d["scenario"] in only]
    if ids:
        items = [d for d in items if d["id"] in ids]
    if a.limit is not None:
        items = items[: a.limit]

    a.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if a.out.exists():
        with open(a.out, encoding="utf-8") as previous:
            done = {json.loads(line)["id"] for line in previous if line.strip()}
    todo = [d for d in items if d["id"] not in done]
    for d in todo:  # fail before any container starts if a task is missing
        read_instruction(a.harbor_tasks, d["id"])
    print(f"{len(todo)} items to ask ({len(done)} already answered)", flush=True)

    with cf.ThreadPoolExecutor(a.jobs) as ex, open(a.out, "a", encoding="utf-8") as fo:
        futs = [ex.submit(ask, d, a) for d in todo]
        for f in cf.as_completed(futs):
            r = f.result()
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
            fo.flush()
            print(f"{r['id']}: {str(r['response'])[:60]!r} {r['latency_s']}s", flush=True)


if __name__ == "__main__":
    main()
