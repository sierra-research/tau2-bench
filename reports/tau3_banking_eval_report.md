# τ³-Bench Banking: MiniMax M3 vs Claude Opus 4.8 vs Opus 4.7

*Run overnight 2026-06-02 on Modal. 10 tasks, canonical AllTools config, gpt-5.2 user-sim, seed 300, pass@1.*

**Read first.** On 10 τ³ `banking_knowledge` tasks, the open-weights **MiniMax M3 edged both Claude Opus models, 5/10 vs 4/10.** Two caveats keep this preliminary: the 10-task subset runs ~1.5x easier than the full benchmark, and **Modal's container blocked the AllTools shell tool** (so this was BM25 + dense retrieval, not the full shell-enabled config).

## Results

| Model | Our pass@1 (10 tasks) | Official pass@1 (97 tasks) |
|---|---|---|
| **MiniMax M3** (open) | **50%** (5/10) | not on leaderboard |
| **Claude Opus 4.8** | **40%** (4/10) | not on leaderboard |
| Opus 4.7 (control) | 40% (4/10) | **25.3%** |
| GPT-5.5 (reference) | — | 37.4% |
| GPT-5.2 (reference) | — | 24.7% |

**M3 and Opus are not strictly ranked.** M3 uniquely solved tasks 007 and 010; Opus uniquely solved 002 and 012. Complementary strengths, not domination.

Per-task (1 = pass):

| task | M3 | Opus 4.8 | Opus 4.7 |
|---|---|---|---|
| 001 | 1 | 1 | 1 |
| 002 | 0 | 0 | 1 |
| 003 | 0 | 0 | 0 |
| 004 | 1 | 1 | 0 |
| 005 | 0 | 0 | 0 |
| 006 | 1 | 1 | 1 |
| 007 | **1** | 0 | 0 |
| 008 | 0 | 0 | 0 |
| 010 | **1** | 0 | 0 |
| 012 | 0 | 1 | 1 |

## Two caveats that gate the numbers

**1. The 10-task subset is easy.** The Opus-4.7 control scored **40% here vs its official 25.3%** on the full 97-task set, so the first 10 tasks (by seed) run ~1.5x easier than average. Absolute scores are **not** leaderboard-comparable; only the relative M3-vs-Opus comparison on identical tasks is clean. A full 97-task run is needed for a real number.

**2. The AllTools shell tool did not run in Modal.** Every `shell(...)` call returned `bwrap: Creating new namespace failed: Operation not permitted`, Modal's gVisor sandbox blocks the bubblewrap user-namespace that the agentic-shell needs. The models fell back to BM25 + dense KB search (both work fine), so this was effectively **AllTools-minus-shell**. The official submissions ran a working shell. To get the true config, the shell needs a privileged container (or run locally where bwrap works).

## Replication check

With the config matched to Sierra's methodology (AllTools, gpt-5.2 user-sim, seed 300) the Opus-4.7 control lands in the right zone once you account for the easy subset (40% on 10 easy tasks ≈ 25.3% over all 97). The pipeline is faithful **except for the dead shell**. Verdict: directionally replicating, not yet a hard match. Needs the full set + working shell to confirm.

## Trajectory findings (the interesting part)

- **M3 is exhaustive in retrieval.** On task 007 it fired 12+ KB searches (BM25 and dense, varying the query and `k`), kept probing until it surfaced the right cards, and passed (45 messages). It also kept retrying the dead shell tool, it never "learned" the sandbox was broken.
- **Opus 4.8 is terse.** Same task: ~2 searches, a time check, then a fast answer, and it missed (10 messages vs M3's 45). Its efficiency costs it recall on hard retrieval tasks. This is the crux of τ-Knowledge: the bottleneck is *finding and reasoning over* the right policy, and M3's brute-force search wins some of those.
- **Opus 4.7 sits in between** (9 tool calls), tried the shell, fell back to search.
- **All three open with identity/context gathering and ground answers in retrieved docs**, no fabricated policies. That is exactly the behavior AfterQuery had to fine-tune *into* Llama-3.1-8B; it's native in all three frontier models here.

## Cost

~**$0.70 per conversation** for the Opus models. M3 is cheaper per token but used the most (it searches more): ~208K input tokens/rollout vs Opus 4.8's ~141K and Opus 4.7's ~97K. All 30 rollouts ≈ a few dollars of API.

## Data

All 30 rollouts, full trajectories (every KB search, tool call, message) plus DB-state reward, are in Supabase: `tau2_rollouts_m3`, `tau2_rollouts_opus48`, `tau2_rollouts_opus47`.

## Next steps for a leaderboard-grade result

1. **Full 97 tasks × 4 trials** (pass^1-4) for real comparability with the official board.
2. **Fix the shell**: run AllTools in a privileged container (or locally) so bwrap works, or accept BM25+dense and label it explicitly.
3. Then **M3 and Opus 4.8 become genuinely new leaderboard entries**, neither exists on Sierra's board today, which is the publishable angle.
