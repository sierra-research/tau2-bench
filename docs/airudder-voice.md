# AI Rudder Voice: Realtime-compatible endpoint

AI Rudder Voice reuses the OpenAI audio-native adapter with a separate endpoint
and credential selected by the `airudder-` model prefix. The hook changes only
endpoint/key selection; the benchmark's policy, tools, user simulator and scoring
remain in the stock harness. Existing OpenAI and Pine routes are unchanged.

## Access

- Provider argument: `openai` (the reused protocol adapter, not the agent vendor).
- Model: `airudder-voice-v1`.
- `AIRUDDER_REALTIME_BASE_URL`: `wss://taubench.airudder.com/v1/realtime`.
- `AIRUDDER_REALTIME_API_KEY`: obtain privately from the evaluation contact.

Set the base URL without a query string; the adapter appends the model parameter.
Credentials are sent in the Authorization Bearer header. Do not put the Token in
Git, a public issue/PR, URLs, logs or command-line arguments. Request private
credential delivery and coordinate an evaluation window with Yechang Hu at
yechang.hu@airudder.com. No agent-side Gemini, ASR or TTS keys are required by the
benchmark client. Evaluator-side simulator/judge/TTS credentials remain separate.

Missing endpoint or credential fails explicitly instead of falling back to an
OpenAI key. At the current service, an absent/invalid Token is rejected after WS
Upgrade with close code 1008; a valid connection receives `session.created`.
Use normal TLS verification. Sending `session.update` can start a billable call.

One model and one endpoint serve retail, airline and telecom. Internal routing
uses the policy and tools supplied through the normal session interface, not
task IDs, hidden goals or reference answers.

## Reproduction

See [voice setup](leaderboard-submission.md#voice-persona-setup-required-for-local-runs)
for local voices and evaluator credentials. The scored run used upstream
`b7ea9074c1cba482b30687fecdb5c8425fd6f619` with the transport hook in this PR.
It used Python 3.12.13 and LiteLLM 1.82.6 rather than the upstream 1.81.11 lock;
the latter difference matters for Custom xhigh serialization. Full configuration,
dependency deviations and retry counts are in the
[evaluation methodology](submissions/ai-rudder-voice-2026-09-26.md).

After configuring the endpoint and private credential in the environment:

```bash
tau2 run --domain retail --audio-native \
  --audio-native-provider openai --audio-native-model airudder-voice-v1 \
  --speech-complexity regular --num-trials 1 --max-concurrency 6 \
  --hallucination-retries 3 --review-model claude-opus-4-5 \
  --verbose-logs --save-to <fresh_standard_retail_name>
```

Repeat for airline and telecom, using separate save names. Self-run concurrency
was 6 / 3 / 12; coordinate capacity for an official rerun rather than assuming
concurrency cannot affect timeouts. Do not apply task filters for full evaluation.
For Custom, add:

```bash
--user-llm gpt-5.5-2026-04-23 --user-llm-args '{"reasoning_effort":"xhigh"}'
```

Verify the installed client's xhigh serialization before the Custom run. The
agent is identical across the two tracks. Local external-account voices are not
Sierra's internal personas; Sierra performs its final evaluation with its own
voices. Running these commands incurs usage charges and requires coordination.

## Evidence and boundaries

The self-run campaign contains 278 retained tasks per track. It used same-host
loopback; it is not a Sierra-certified result. Separate public-WSS smoke tests
completed one Standard/regular task per domain with audio, matching tool-call
results and reward=1. Smoke disabled hallucination review and is not capacity or
long-duration certification. The public deployment additionally changes the
endpoint's bind address for gateway access, disclosed separately from the scored
system revision. No benchmark-side prompt, tool or evaluator change is proposed.

Canonical trajectory/audio packages and hashes:

- [Standard](https://taubench.oss-us-east-1.aliyuncs.com/releases/va20260926a-r2/standard.tar.gz)
- [Custom](https://taubench.oss-us-east-1.aliyuncs.com/releases/va20260926a-r2/custom.tar.gz)
- [SHA256SUMS](https://taubench.oss-us-east-1.aliyuncs.com/releases/va20260926a-r2/SHA256SUMS)

The r2 package revision adds the owner declaration and public blog reference;
trajectory/audio bytes and scores are unchanged. Only retained canonical data
is public; internal debug logs and credentials are not included.
