"""
Modal app to run tau2 benchmarks remotely.

Setup:
1. Install Modal: pip install modal
2. Authenticate: modal setup
3. Set secrets: modal secret create openrouter OPENROUTER_API_KEY=your-key-here
4. Run: modal run modal_run.py
"""

import modal

# Create Modal app
app = modal.App("tau2-pctx-eval")

# Create a volume to persist simulation results
volume = modal.Volume.from_name("tau2-results", create_if_missing=True)

# Create a custom image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    # Install curl and other system dependencies
    .apt_install("curl")
    # Install uv
    .pip_install("uv")
    # Install pctx server
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -LsSf https://github.com/portofcontext/pctx/releases/download/v0.6.0-beta.1/pctx-installer.sh | sh"
    )
    # Add tau2-bench directory (exclude .git to avoid build errors)
    .add_local_dir(".", "/root/tau2-bench", ignore=[".git"])
)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("openrouter")],
    volumes={"/results": volume},  # Mount volume to persist results
    timeout=3600 * 6,  # 6 hour timeout
    cpu=2.0,
    memory=4096,
)
def run_tau2_eval():
    """Run the tau2 evaluation benchmark."""
    import os
    import sys
    import subprocess
    import time

    # Set environment variables
    os.environ["PCTX_MODE"] = "fs"

    # Change to tau2-bench directory
    os.chdir("/root/tau2-bench")

    # Start pctx server in the background
    print("Starting pctx server...")
    # Add pctx to PATH (installed in ~/.local/bin by default)
    pctx_path = os.path.expanduser("~/.local/bin/pctx")

    # Start pctx server as a background process
    pctx_process = subprocess.Popen(
        [pctx_path, "start"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Combine stderr with stdout
        text=True,
    )

    # Give the server a moment to start up
    print("Waiting for pctx server to start...")
    time.sleep(5)

    # Check if pctx process is still running
    if pctx_process.poll() is not None:
        # Process exited, capture output
        output, _ = pctx_process.communicate()
        print(f"ERROR: pctx server failed to start!")
        print(f"pctx output:\n{output}")
        raise RuntimeError("pctx server failed to start")

    print("pctx server started successfully")

    # Create a fresh virtual environment
    print("Creating virtual environment...")
    subprocess.run(["uv", "venv"], check=True, capture_output=False)

    # Install tau2 package and its dependencies (including pctx-client from PyPI)
    print("Installing tau2...")
    subprocess.run(
        ["uv", "pip", "install", "-e", "."], check=True, capture_output=False
    )

    # Run tau2 command using uv run (to use the venv we created)
    print("Running tau2 evaluation...")

    # Create a thread to monitor pctx process output
    import threading

    def monitor_pctx():
        """Monitor and print pctx server output."""
        while True:
            line = pctx_process.stdout.readline()
            if not line:
                break
            print(f"[PCTX] {line.rstrip()}")

    monitor_thread = threading.Thread(target=monitor_pctx, daemon=True)
    monitor_thread.start()

    try:
        result = subprocess.run(
            [
                "uv",
                "run",
                "tau2",
                "run",
                "--domain",
                "airline",
                "--agent",
                "llm_agent_pctx",
                "--agent-llm",
                "openrouter/openai/gpt-5",
                "--user-llm",
                "openrouter/openai/gpt-4o-2024-05-13",
                "--log-level",
                "INFO",
                "--max-concurrency",
                "1",
            ],
            check=False,  # Don't raise on error, we want to save results regardless
            capture_output=False,
        )
        return_code = result.returncode
    finally:
        # Copy simulation results to persistent volume
        print("Saving simulation results to volume...")
        subprocess.run(
            ["cp", "-r", "/root/tau2-bench/data", "/results/"], capture_output=False
        )
        volume.commit()  # Persist the volume changes
        print("Results saved to Modal volume 'tau2-results'")

    print("Evaluation complete!")
    return return_code


@app.function(volumes={"/results": volume})
def list_results():
    """List all simulation results in the volume."""
    import os

    results_dir = "/results/data/simulations"
    if os.path.exists(results_dir):
        files = os.listdir(results_dir)
        print(f"Found {len(files)} result files:")
        for f in files:
            print(f"  - {f}")
        return files
    else:
        print("No results found yet.")
        return []


@app.local_entrypoint()
def main():
    """Entry point for modal run."""
    print("Starting tau2 evaluation on Modal...")
    result = run_tau2_eval.remote()
    print(f"Evaluation finished with return code: {result}")
    print("\nTo list saved results, run: modal run modal_run.py::list_results")
    print(
        "To download results, use: modal volume get tau2-results data/simulations <local-path>"
    )
