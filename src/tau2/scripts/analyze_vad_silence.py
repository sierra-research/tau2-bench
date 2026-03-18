#!/usr/bin/env python3
import re
from datetime import datetime, timedelta
from pathlib import Path

from loguru import logger


def parse_timestamp(ts_str):
    return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")


def analyze_task(log_file):
    try:
        with open(log_file, "r") as f:
            lines = f.readlines()
        content = "".join(lines)
    except Exception as e:
        logger.error(f"Error analyzing task {log_file}: {e}")
        return None

    # Find last agent audio time
    last_audio_time = None
    for i, line in enumerate(lines):
        tick_match = re.search(
            r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+).*Tick (\d+) completed", line
        )
        if tick_match:
            tick_time = tick_match.group(1)
            for j in range(i + 1, min(i + 10, len(lines))):
                audio_match = re.search(r"Agent audio: (\d+) bytes", lines[j])
                if audio_match and int(audio_match.group(1)) > 0:
                    last_audio_time = parse_timestamp(tick_time)
                    break

    if not last_audio_time:
        return None

    # Find simulation end time and reward
    sim_end_time = None
    reward = 0.0
    for line in reversed(lines):
        match = re.search(
            r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+).*FINISHED SIMULATION", line
        )
        if match:
            sim_end_time = parse_timestamp(match.group(1))

        # Look for reward separately
        reward_match = re.search(r"Reward:\s*([\d.]+)", line)
        if reward_match:
            reward = float(reward_match.group(1))

        if sim_end_time and reward is not None:
            break

    if not sim_end_time:
        return None

    silence_seconds = (sim_end_time - last_audio_time).total_seconds()

    # Find all speech_started events with timestamps (from provider VAD)
    # xAI uses: "speech_started"
    # OpenAI uses: "Speech started detected at"
    speech_started_times = []
    for match in re.finditer(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+).*(speech_started|Speech started detected)",
        content,
    ):
        speech_started_times.append(parse_timestamp(match.group(1)))

    # Find user speech chunks (from user simulator, speech=True)
    user_speech_times = []
    for match in re.finditer(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+).*\[USER\].*speech=True", content
    ):
        user_speech_times.append(parse_timestamp(match.group(1)))

    # Count speech_started during silence period (after last audio)
    speech_during_silence = sum(1 for t in speech_started_times if t > last_audio_time)

    # Count speech_started in last 20 seconds of call
    last_20s_start = sim_end_time - timedelta(seconds=20)
    speech_last_20s = sum(1 for t in speech_started_times if t > last_20s_start)

    # Count user speech chunks during silence period (what user simulator sent)
    user_speech_during_silence = sum(
        1 for t in user_speech_times if t > last_audio_time
    )
    user_speech_last_20s = sum(1 for t in user_speech_times if t > last_20s_start)

    return {
        "silence_seconds": silence_seconds,
        "reward": reward,
        "total_speech_started": len(speech_started_times),
        "speech_during_silence": speech_during_silence,
        "speech_last_20s": speech_last_20s,
        "user_speech_during_silence": user_speech_during_silence,
        "user_speech_last_20s": user_speech_last_20s,
    }


def main():
    import sys

    provider = sys.argv[1] if len(sys.argv) > 1 else "xai"

    # Analyze all tasks for the specified provider
    dirs = [
        (
            "XML",
            f"tmp/qa_results/exp_20250116_xml_tags/retail_control_{provider}/tasks",
        ),
        (
            "NO_XML",
            f"tmp/qa_results/exp_20250116_no_xml_tags/retail_control_{provider}/tasks",
        ),
    ]

    all_results = []

    for exp_name, tasks_path in dirs:
        tasks_dir = Path(tasks_path)
        if not tasks_dir.exists():
            continue

        for task_dir in sorted(
            tasks_dir.iterdir(),
            key=lambda x: (
                int(x.name.split("_")[1]) if x.name.startswith("task_") else -1
            ),
        ):
            if not task_dir.is_dir() or task_dir.name == "Icon":
                continue

            try:
                task_num = int(task_dir.name.split("_")[1])
            except Exception as e:
                logger.error(f"Error parsing task {task_dir.name}: {e}")
                continue

            for sim_dir in task_dir.iterdir():
                if not sim_dir.is_dir() or sim_dir.name == "Icon":
                    continue

                log_file = sim_dir / "task.log"
                if not log_file.exists():
                    continue

                result = analyze_task(log_file)
                if result:
                    result["exp"] = exp_name
                    result["task"] = task_dir.name
                    result["task_num"] = task_num
                    all_results.append(result)
                break

    # Sort by exp, then task_num
    all_results.sort(key=lambda x: (x["exp"], x["task_num"]))

    print("=" * 140)
    print(f"{provider.upper()} SILENCE ANALYSIS - All 60 Tasks (30 XML + 30 NO_XML)")
    print("=" * 140)
    print()
    print(
        f"{'Exp':<7} {'Task':<10} {'Silence':>8} {'VAD':>8} {'VAD':>8} {'User Sim':>8} {'User Sim':>8} {'Reward':>7} {'Issue?':<12}"
    )
    print(
        f"{'':>7} {'':>10} {'(sec)':>8} {'During':>8} {'Last20s':>8} {'During':>8} {'Last20s':>8} {'':>7}"
    )
    print(
        f"{'-' * 7} {'-' * 10} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 7} {'-' * 12}"
    )

    issues_found = 0
    vad_misses = 0
    no_user = 0
    for r in all_results:
        # Flag as issue if: silence > 20s
        # Further categorize:
        # - VAD MISS: User sim sent speech but xAI VAD didn't detect (user_speech_during_silence > 0 but speech_during_silence == 0)
        # - NO USER: User sim didn't send any speech in last 20s
        issue = ""
        if r["silence_seconds"] > 20:
            if r["user_speech_during_silence"] > 0 and r["speech_during_silence"] == 0:
                issue = "VAD MISS!"
                vad_misses += 1
                issues_found += 1
            elif r["user_speech_last_20s"] == 0:
                issue = "NO USER"
                no_user += 1
                issues_found += 1
            else:
                issue = "OTHER"
                issues_found += 1

        print(
            f"{r['exp']:<7} {r['task']:<10} {r['silence_seconds']:>8.1f} {r['speech_during_silence']:>8} {r['speech_last_20s']:>8} {r['user_speech_during_silence']:>8} {r['user_speech_last_20s']:>8} {r['reward']:>7.1f} {issue:<12}"
        )

    print()
    print("=" * 140)
    print("LEGEND:")
    print("  - Silence: Seconds from last agent audio to simulation end")
    print("  - VAD During: speech_started events from provider AFTER agent went silent")
    print("  - VAD Last20s: speech_started events from provider in final 20 seconds")
    print(
        "  - User Sim During: User simulator speech chunks sent AFTER agent went silent"
    )
    print("  - User Sim Last20s: User simulator speech chunks sent in final 20 seconds")
    print(
        "  - VAD MISS!: User simulator sent speech during silence, but provider VAD didn't detect it"
    )
    print("  - NO USER: User simulator didn't send any speech in last 20 seconds")
    print()
    print(f"Tasks with issues (>20s silence): {issues_found}")
    print(f"  - VAD MISS (user sent speech but xAI didn't detect): {vad_misses}")
    print(f"  - NO USER (user sim didn't send speech): {no_user}")


if __name__ == "__main__":
    main()
