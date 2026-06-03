"""Run the τ³ banking_knowledge eval on Modal, in the AllTools config, inside an
isolated container (so the agent's sandboxed shell never touches the local machine).

Canonical config matched to Sierra's leaderboard submissions so scores land on the
same scale:
  retrieval = alltools (BM25 + dense text-embedding-3-large + sandboxed shell)
  user simulator = gpt-5.2 (reasoning_effort low), seed 300, banking_knowledge only.

Each rollout (full trajectory + DB-state reward) is written to Supabase
tau2_rollouts_<suffix> by scripts/rollouts_supabase.py (reads creds from the
tau3-eval secret env vars).

Smoke first, then scale:
  modal run scripts/modal_tau3_eval.py --num-tasks 1            # 1-task smoke (M3 only)
  modal run scripts/modal_tau3_eval.py --num-tasks 10 --models m3,opus48
"""
import modal

REPO = "/Users/lilyzhang/Documents/lily-memory/Build/Autoresearch/tau2-bench"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ripgrep", "bubblewrap", "socat", "git", "curl")
    # Node + Anthropic sandbox-runtime (provides `srt`, the agentic-shell sandbox)
    .run_commands(
        "curl -fsSL https://deb.nodesource.com/setup_20.x | bash -",
        "apt-get install -y nodejs",
        "npm install -g @anthropic-ai/sandbox-runtime@0.0.23",
    )
    .pip_install("psycopg2-binary")
    .add_local_dir(
        REPO, "/tau2", copy=True,
        ignore=["**/.venv", "**/.git", "**/node_modules", "**/__pycache__",
                "data/simulations/**", "web/**", "**/*.pyc"],
    )
    .run_commands("cd /tau2 && pip install -e '.[knowledge]'")
    # bwrap can't create user namespaces under Modal's gVisor, so run the agentic
    # shell directly in the (already isolated) container instead of via srt/bwrap.
    .env({"TAU2_NO_SRT_SANDBOX": "1"})
)

app = modal.App("tau3-banking-eval")

# model registry: suffix -> (litellm model id, agent-llm-args, the model's OWN board retrieval config)
MODELS = {
    "m3":     ("openrouter/minimax/minimax-m3", '{"temperature":0,"max_tokens":16000}', "alltools"),
    "opus48": ("claude-opus-4-8",               '{"max_tokens":16000}', "alltools"),  # Opus rejects reasoning_effort/thinking via litellm; plain max_tokens runs
    "opus47": ("claude-opus-4-7",               '{"max_tokens":16000}', "alltools"),  # leaderboard control (official 25.3%)
    # GLM-5 board entry: text-embedding-3-large (= openai_embeddings), thinking on, temp 1.0 / top_p 0.95. Target 9.79%.
    "glm5":   ("openrouter/z-ai/glm-5",         '{"temperature":1.0,"top_p":0.95,"max_tokens":16000}', "openai_embeddings"),
}


@app.function(image=image, secrets=[modal.Secret.from_name("tau3-eval")],
              timeout=14400, cpu=4.0, memory=8192)
def verify_shell():
    """Cheap proof the agentic shell runs in Modal (no LLM cost): build a sandbox
    and run a real shell command. Pre-fix this returned the bwrap namespace error;
    post-fix it should return the command output."""
    import os
    os.chdir("/tau2")
    from tau2.knowledge.sandbox_manager import SandboxManager
    sm = SandboxManager(allow_writes=False)
    rc, out, err = sm.run_command("echo shell-ok && uname -a")
    passed = rc == 0 and "shell-ok" in out and "operation not permitted" not in err.lower()
    print(f"SHELL_VERIFY result: rc={rc!r} no_srt_sandbox={os.environ.get('TAU2_NO_SRT_SANDBOX')!r}", flush=True)
    print(f"SHELL_VERIFY stdout: {out.strip()!r}", flush=True)
    print(f"SHELL_VERIFY stderr: {err.strip()!r}", flush=True)
    print(f"SHELL_VERIFY: {'PASS — shell runs in Modal' if passed else 'FAIL'}", flush=True)
    return {"passed": passed, "returncode": rc, "stdout": out.strip(), "stderr": err.strip()}


@app.function(image=image, secrets=[modal.Secret.from_name("tau3-eval")],
              timeout=14400, cpu=4.0, memory=8192)
def run_eval(suffix: str, num_tasks: int):
    """Run the canonical AllTools config; if the shell sandbox can't run in the
    container, fall back to openai_embeddings so we still produce data overnight."""
    import os, subprocess
    model, args, primary_cfg = MODELS[suffix]
    os.chdir("/tau2")
    # run the model's OWN board config first; fall back to openai_embeddings (no shell) if it can't run
    configs = [primary_cfg] + (["openai_embeddings"] if primary_cfg != "openai_embeddings" else [])
    for rcfg in configs:
        save = f"eval_{suffix}_{rcfg}_n{num_tasks}"
        cmd = [
            "tau2", "run", "--domain", "banking_knowledge", "--retrieval-config", rcfg,
            "--agent-llm", model, "--agent-llm-args", args,
            "--user-llm", "gpt-5.2", "--user-llm-args", '{"reasoning_effort":"low"}',
            "--num-tasks", str(num_tasks), "--num-trials", "1",
            "--max-concurrency", "8", "--max-steps", "100", "--seed", "300",
            "--save-to", save,
        ]
        print(f"RUN [{rcfg}]:", " ".join(cmd), flush=True)
        subprocess.run(cmd, text=True)
        results = f"/tau2/data/simulations/{save}/results.json"
        if not os.path.exists(results):
            print(f"[{rcfg}] no results.json — trying next config", flush=True)
            continue
        subprocess.run(["python", "scripts/rollouts_supabase.py", "create", "--suffix", suffix])
        subprocess.run(["python", "scripts/rollouts_supabase.py", "push", "--suffix", suffix,
                        "--results", results, "--retrieval-config", rcfg,
                        "--domain", "banking_knowledge"])
        out = subprocess.run(["python", "scripts/rollouts_supabase.py", "stats", "--suffix", suffix],
                             capture_output=True, text=True)
        return f"{suffix} [{rcfg}]: {out.stdout.strip()}"
    return f"{suffix}: FAILED (both alltools and openai_embeddings produced no results)"


@app.local_entrypoint()
def main(num_tasks: int = 1, models: str = "m3"):
    suffixes = [s.strip() for s in models.split(",") if s.strip()]
    print(f"== τ³ banking eval on Modal: models={suffixes}, num_tasks={num_tasks} ==")
    # run models in parallel containers
    results = list(run_eval.starmap([(s, num_tasks) for s in suffixes]))
    print("\n==== RESULTS ====")
    for s, res in zip(suffixes, results):
        print(f"[{s}] {res}")
