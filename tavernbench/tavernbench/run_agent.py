"""Answer a TavernBench set with Claude Code running as an agent in a container.

    python -m tavernbench.run_agent --set data/eval \
        --harbor-tasks /path/to/adapters/datasets/tavernbench \
        --image tavernbench-agent:<claude-code version> --model <model id> \
        --out answers/<run name>/answers.jsonl [--jobs 4]

This runner exists for the Harbor parity experiments (see ../harbor_parity/README.md).
`run_claude` asks Claude once, without tools. Harbor instead runs Claude Code as an
agent that works in a container and writes its answer to /app/answer.json. To compare
the two harnesses under the same conditions, this runner does on the original side what
Harbor's `claude-code` agent does (Harbor 0.27.0, harbor/agents/installed/claude_code.py):

* one fresh container per task, from an image that holds the Harbor task image plus
  Claude Code at a pinned version (harbor_parity/Dockerfile.agent), with the task's
  CPU and memory limits;
* the instruction is the text of the Harbor task's instruction.md, read from
  --harbor-tasks, so both sides give the agent the same bytes;
* the same two commands as Harbor's claude-code agent, run the same way: a setup
  command, then the launch command, each a separate `docker exec ... bash -c` in the
  running container, behind Harbor's pipefail prefix, with the same environment and
  CLI flags. --cli-flags takes the flags Harbor builds; the default is what Harbor
  builds for `disable_web_search=true`;
* the reply is the content of /app/answer.json when the agent stops. As in Harbor, an
  agent that times out or exits with an error is still graded on that file.

The output rows have the fields `tavernbench.score` reads (`id`, `response`), so the
unchanged scorer grades them:

    python -m tavernbench.score --set data/eval --answers answers/<run name>/answers.jsonl

Rows carry no `output_tokens`, `finish_reason`, `error` or `is_error`: the scorer's
per-call checks (token budget, cut-off or failed calls) describe a single model call and
do not apply to an agent run, on either side. Agent outcomes are kept under `agent_*`
names for diagnosis. A missing answer file gives `response: null`, which scores 0.
Harbor keeps no reward for a run that ended in a model-API error (a rate limit, an
overload, a lost connection...); the parity workflow classifies the rows the way Harbor
does and marks those with `error` before scoring, so both sides count them as 0.

Credentials are read from the environment and passed to the container by name, so they
never appear on a command line. As in Harbor, the key is taken from ANTHROPIC_API_KEY,
else ANTHROPIC_AUTH_TOKEN, and keeps that name in the container; ANTHROPIC_BASE_URL is
passed when set. Resumable: ids already in --out are skipped, except rows whose
container could not be run (`runner_error`), which are tried again.
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

# The flags Harbor's claude-code agent builds for `disable_web_search=true`.
DEFAULT_CLI_FLAGS = "--disallowedTools WebSearch,WebFetch --permission-mode=bypassPermissions"

# Harbor puts this in front of every agent command (harbor/agents/installed/base.py), so
# the launch command fails when Claude Code fails, not only when `tee` does.
PIPEFAIL_PREFIX = "(set -o pipefail) 2>/dev/null && set -o pipefail; "

# Harbor's setup command, run before the agent starts.
SETUP_COMMAND = (
    "mkdir -p $CLAUDE_CONFIG_DIR/debug $CLAUDE_CONFIG_DIR/projects/-app "
    "$CLAUDE_CONFIG_DIR/shell-snapshots $CLAUDE_CONFIG_DIR/statsig "
    "$CLAUDE_CONFIG_DIR/todos $CLAUDE_CONFIG_DIR/skills && "
    "if [ -d ~/.claude/skills ]; then "
    "cp -r ~/.claude/skills/. $CLAUDE_CONFIG_DIR/skills/ 2>/dev/null || true; "
    "fi"
)

# With a custom endpoint Harbor points every model alias at the requested model.
MODEL_ALIAS_VARS = (
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "ANTHROPIC_DEFAULT_FABLE_MODEL",
    "CLAUDE_CODE_SUBAGENT_MODEL",
)


def launch_command(cli_flags: str, instruction_var: str) -> str:
    """Harbor's launch command. The instruction arrives in $<instruction_var>, is copied
    to a shell variable and unset, so the agent's own processes do not see it."""
    shell_var = instruction_var.lower()
    flags = (cli_flags.strip() + " ") if cli_flags.strip() else ""
    return (
        'export PATH="$HOME/.local/bin:$PATH"; '
        f'{shell_var}="${instruction_var}"; '
        f"unset {instruction_var}; "
        f'printf "%s" "${shell_var}" | '
        "claude --verbose --output-format=stream-json "
        f"{flags}"
        "--print 2>&1 | tee /logs/agent/claude-code.txt"
    )


def exec_commands(cli_flags: str, instruction_var: str) -> tuple[str, str]:
    """The two scripts Harbor runs with `bash -c` in the task container, in order."""
    return (
        PIPEFAIL_PREFIX + SETUP_COMMAND,
        PIPEFAIL_PREFIX + launch_command(cli_flags, instruction_var),
    )


def container_env(model: str) -> dict[str, str]:
    """Environment of the agent process, as Harbor's claude-code agent sets it."""
    env: dict[str, str] = {}
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    auth_token = os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
    base_url = os.environ.get("ANTHROPIC_BASE_URL", "")
    if api_key:
        env["ANTHROPIC_API_KEY"] = api_key
        env["ANTHROPIC_AUTH_TOKEN"] = ""
    else:
        env["ANTHROPIC_AUTH_TOKEN"] = auth_token
    if base_url:
        env["ANTHROPIC_BASE_URL"] = base_url
    if os.environ.get("CLAUDE_CODE_MAX_OUTPUT_TOKENS"):
        env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = os.environ["CLAUDE_CODE_MAX_OUTPUT_TOKENS"]
    env["ANTHROPIC_MODEL"] = model
    if base_url:
        for name in MODEL_ALIAS_VARS:
            env[name] = os.environ.get(name, model)
    if os.environ.get("CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING", "").strip() == "1":
        env["CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING"] = "1"
    env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
    env["IS_SANDBOX"] = "1"
    if os.environ.get("MAX_THINKING_TOKENS"):
        env["MAX_THINKING_TOKENS"] = os.environ["MAX_THINKING_TOKENS"]
    env["CLAUDE_CONFIG_DIR"] = "/logs/agent/sessions"
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


def docker_exec(
    args: argparse.Namespace, name: str, script: str, env: dict[str, str], timeout=None
) -> subprocess.CompletedProcess:
    """`docker exec` in the task container, as Harbor's docker environment does it.
    Variables are passed by name; their values come from this process's environment."""
    cmd = [args.docker, "exec", "--workdir", "/app"]
    for key in env:
        cmd += ["--env", key]
    cmd += [name, "bash", "-c", script]
    return subprocess.run(
        cmd,
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def ask(item: dict, args: argparse.Namespace) -> dict:
    t0 = time.time()
    row = {"id": item["id"], "model": args.model, "mode": "agent", "response": None}
    work = Path(tempfile.mkdtemp(prefix=f"tavernbench-{item['id'].lower()}-", dir=args.work_dir))
    name = f"tavernbench-{item['id'].lower()}-{uuid.uuid4().hex[:8]}"
    started = False
    try:
        instruction = read_instruction(args.harbor_tasks, item["id"])
        app, logs = work / "app", work / "logs"
        app.mkdir()
        logs.mkdir()
        instruction_var = f"CLAUDE_CODE_INSTRUCTION_{uuid.uuid4().hex.upper()}"
        env = container_env(args.model)
        setup_script, launch_script = exec_commands(args.cli_flags, instruction_var)

        start = [args.docker, "run", "--detach", "--rm", "--name", name, "--workdir", "/app"]
        start += ["--cpus", str(args.cpus), "--memory", f"{args.memory_mb}m"]
        start += ["--volume", f"{app}:/app", "--volume", f"{logs}:/logs/agent"]
        start += [args.image, "sleep", "infinity"]
        subprocess.run(start, capture_output=True, text=True, check=True)
        started = True

        setup = docker_exec(args, name, setup_script, env, timeout=args.timeout)
        if setup.returncode != 0:
            raise RuntimeError(f"setup command failed ({setup.returncode}): {setup.stderr[-200:]}")

        timed_out = False
        try:
            proc = docker_exec(
                args, name, launch_script, {**env, instruction_var: instruction}, args.timeout
            )
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out, exit_code = True, None
        subprocess.run([args.docker, "rm", "--force", name], capture_output=True, check=False)
        started = False

        answer = app / "answer.json"
        if answer.is_file():
            row["response"] = answer.read_text(encoding="utf-8", errors="replace")
        result = last_result_event(logs / "claude-code.txt")
        usage = result.get("usage") or {}
        row.update(
            {
                "agent_exit_code": exit_code,
                "agent_timed_out": timed_out,
                "agent_result_subtype": result.get("subtype"),
                "agent_is_error": result.get("is_error"),
                "agent_error": (str(result.get("result"))[:300] if result.get("is_error") else None),
                "num_turns": result.get("num_turns"),
                "cost_usd": result.get("total_cost_usd"),
                "agent_input_tokens": usage.get("input_tokens"),
                "agent_cache_read_tokens": usage.get("cache_read_input_tokens"),
                "agent_output_tokens": usage.get("output_tokens"),
            }
        )
        if args.keep_logs:
            dest = args.out.parent / "logs" / item["id"]
            shutil.rmtree(dest, ignore_errors=True)
            shutil.copytree(logs, dest, ignore=shutil.ignore_patterns("sessions"))
    except Exception as exc:  # the container could not be run: record it and try again later
        if isinstance(exc, subprocess.CalledProcessError):
            stderr = (exc.stderr or "").strip()
            message = f"`{' '.join(exc.cmd[:2])}` exited with {exc.returncode}: {stderr[-200:]}"
        else:
            message = f"{type(exc).__name__}: {exc}"
        row["runner_error"] = message[:300]
    finally:
        if started:
            subprocess.run([args.docker, "rm", "--force", name], capture_output=True, check=False)
        shutil.rmtree(work, ignore_errors=True)
    row["latency_s"] = round(time.time() - t0, 2)
    row["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
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
    ap.add_argument("--model", required=True, help="model id, the one Harbor passes to the agent")
    ap.add_argument(
        "--cli-flags",
        default=DEFAULT_CLI_FLAGS,
        help="Claude Code flags, as Harbor builds them (default: disable_web_search=true)",
    )
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
        ap.error("set ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN (and ANTHROPIC_BASE_URL) first")

    with open(Path(a.set) / "prompts.jsonl", encoding="utf-8") as source:
        items = [json.loads(line) for line in source]
    only = [s for s in a.only.split(",") if s]
    ids = {s for s in a.ids.split(",") if s}
    if only:
        items = [d for d in items if d["scenario"] in only]
    if ids:
        unknown = sorted(ids - {d["id"] for d in items})
        if unknown:
            ap.error(f"unknown task ids: {', '.join(unknown)}")
        items = [d for d in items if d["id"] in ids]
    if a.limit is not None:
        items = items[: a.limit]

    a.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if a.out.exists():
        with open(a.out, encoding="utf-8") as previous:
            last = {}
            for line in previous:
                if line.strip():
                    row = json.loads(line)
                    last[row["id"]] = row
            done = {i for i, row in last.items() if not row.get("runner_error")}
    todo = [d for d in items if d["id"] not in done]
    for d in todo:  # fail before any container starts if a task is missing
        read_instruction(a.harbor_tasks, d["id"])
    print(f"{len(todo)} items to ask ({len(items) - len(todo)} already answered)", flush=True)

    with cf.ThreadPoolExecutor(a.jobs) as ex, open(a.out, "a", encoding="utf-8") as fo:
        futs = [ex.submit(ask, d, a) for d in todo]
        for f in cf.as_completed(futs):
            r = f.result()
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
            fo.flush()
            status = r.get("runner_error") or str(r["response"])[:60]
            print(f"{r['id']}: {status!r} {r['latency_s']}s", flush=True)


if __name__ == "__main__":
    main()
