import asyncio
import base64
import json
import random
import shlex
import struct
import uuid
from typing import Any

MAX_STEPS = 200
MAX_ERRORS = 10
BASE_SEED = 300
NUM_TRIALS = 5
MAX_API_ATTEMPTS = 30
BRIDGE_INLINE_LIMIT = 48 * 1024
BRIDGE_CHUNK_SIZE = 32 * 1024
STOP_MARKERS = ("###STOP###", "###TRANSFER###", "###OUT-OF-SCOPE###")

AGENT_INSTRUCTION = """You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only."""

NL_SYSTEM_PROMPT = """
        TASK
        - You will be given a list of expected outcomes and a conversation that was collected during a test case run.
        - The conversation is between an agent and a customer.
        - Your job is to evaluate whether the agent satisfies each of the expected outcomes.
        - Grade each expected outcome individually.

        FORMAT
        - Your response should be a JSON object with the following fields:
        - `reasoning`: a short explanation for your classification
        - `metExpectation`: `true` if the agent satisfies the expected outcomes, `false` otherwise
        - `expectedOutcome`: repeat the expectation from the input that you are grading

        Example response structure:
        {
            "results": [
                {
                    "expectedOutcome": "<one of the expected outcomes from the input>",
                    "reasoning": "<reasoning trace>",
                    "metExpectation": <false or true>,
                }
            ]
        }
        """


def _record(args):
    md = getattr(args, "metadata", None) or {}
    return md, md.get("metadata", md)


def _mode(args):
    _, extra = _record(args)
    return extra.get("mode") or "formal"


def _run_config(args):
    """Read run-specific knobs from the DataSpec metadata record."""
    _, extra = _record(args)
    params = getattr(args, "params", None) or {}
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except (TypeError, ValueError):
            params = {}
    # Task-level params are the explicit override path for template profiles;
    # record metadata remains the source for frozen per-case values.
    if isinstance(params, dict):
        extra = {**extra, **{k: params[k] for k in ("candidate_reasoning_effort", "retrieval_variant") if k in params}}
    effort = extra.get("candidate_reasoning_effort")
    if effort not in {"medium", "high", "max"}:
        raise ValueError("candidate_reasoning_effort must be medium, high, or max")
    retrieval = extra.get("retrieval_variant")
    if retrieval != "alltools":
        raise ValueError("retrieval_variant must be alltools for the Sierra-aligned profile")
    return effort, retrieval


async def _ensure_bridge(sh, asset_url):
    bootstrap = "import hashlib\nimport pathlib\nimport shutil\nimport subprocess\nimport tarfile\nimport urllib.request\nimport sys\n\nEXPECTED_SHA256 = 'ec6f44410985bbecef7fc933e2b43bfcb54a6f5aedaad7721565db49ba163d45'\nURL = sys.argv[1]\nmarker = pathlib.Path('/tmp/tau3-runtime-layer.sha256')\nif not marker.exists() or marker.read_text().strip() != EXPECTED_SHA256:\n    destination = pathlib.Path('/tmp/tau3-runtime-delta.tar.gz')\n    try:\n        response = urllib.request.urlopen(URL, timeout=90)\n    except Exception:\n        response = urllib.request.build_opener(urllib.request.ProxyHandler({})).open(URL, timeout=90)\n    with response, destination.open('wb') as target:\n        shutil.copyfileobj(response, target)\n    if hashlib.sha256(destination.read_bytes()).hexdigest() != EXPECTED_SHA256:\n        raise RuntimeError('runtime_asset_hash_mismatch')\n    layers = pathlib.Path('/tmp/tau3-runtime-layers')\n    layers.mkdir(exist_ok=True)\n    with tarfile.open(destination) as bundle:\n        for member in bundle.getmembers():\n            if not member.name.startswith('layers/') or not member.isfile() or '..' in member.name:\n                raise RuntimeError('invalid_runtime_bundle')\n        bundle.extractall(layers, filter='data')\n    for layer in sorted((layers / 'layers').glob('*.tar')):\n        subprocess.run(['tar', '--exclude=.wh.*', '-xf', str(layer), '-C', '/'], check=True, capture_output=True)\n    for name in ['srt', 'rg', 'bwrap', 'socat']:\n        if not shutil.which(name):\n            raise RuntimeError('runtime_binary_missing:' + name)\n    marker.write_text(EXPECTED_SHA256 + '\\n')\nprint('runtime_asset_verified:' + EXPECTED_SHA256)\n"
    prepared = await sh("python3 -c " + shlex.quote(bootstrap) + " " + shlex.quote(asset_url), timeout=300)
    if prepared.exit_code != 0:
        raise RuntimeError("runtime_asset_prepare_failed:" + prepared.stderr[-1500:])
    command = """if python3 -c 'import urllib.request; urllib.request.urlopen("http://127.0.0.1:8765/health", timeout=2).read()' >/dev/null 2>&1; then exit 0; fi
rm -f /tmp/tau3-bridge.log /tmp/tau3-bridge.pid
cd /workspace
nohup python3 /opt/tau3_bridge.py --port 8765 </dev/null >/tmp/tau3-bridge.log 2>&1 &
echo $! >/tmp/tau3-bridge.pid
for i in $(seq 1 60); do
  python3 -c 'import urllib.request; urllib.request.urlopen("http://127.0.0.1:8765/health", timeout=2).read()' >/dev/null 2>&1 && exit 0
  sleep 1
done
tail -100 /tmp/tau3-bridge.log >&2 2>/dev/null || true
exit 1"""
    result = await sh(command, timeout=75)
    if result.exit_code != 0:
        raise RuntimeError(f"tau3_bridge_start_failed:{result.stderr[-3000:]}")


async def _post(sh, path, payload, timeout=180):
    encoded = base64.b64encode(json.dumps(payload, ensure_ascii=False).encode()).decode()
    url = "http://127.0.0.1:8765" + path
    if len(encoded) <= BRIDGE_INLINE_LIMIT:
        script = (
            "import base64,sys,urllib.request; "
            "data=base64.b64decode(sys.argv[1]); "
            "request=urllib.request.Request(sys.argv[2],data=data,headers={'Content-Type':'application/json'}); "
            f"\ntry:\n print(urllib.request.urlopen(request,timeout={int(timeout)}).read().decode())\nexcept urllib.error.HTTPError as exc:\n print(exc.read().decode(),file=sys.stderr)\n sys.exit(1)"
        )
        command = f"python3 -c {shlex.quote(script)} {shlex.quote(encoded)} {shlex.quote(url)}"
        result = await sh(command, timeout=timeout + 10)
    else:
        # Linux limits each argv entry to roughly 128 KiB. Long knowledge
        # trajectories can exceed that after base64 expansion, so stage large
        # payloads in bounded chunks before posting them to the local bridge.
        payload_path = f"/tmp/tau3-payload-{uuid.uuid4().hex}.b64"
        init = await sh(f": > {shlex.quote(payload_path)}", timeout=30)
        if init.exit_code != 0:
            raise RuntimeError(f"bridge_payload_init_failed:{init.stderr[-3000:]}")
        try:
            for start in range(0, len(encoded), BRIDGE_CHUNK_SIZE):
                chunk = encoded[start : start + BRIDGE_CHUNK_SIZE]
                append = await sh(
                    f"printf %s {shlex.quote(chunk)} >> {shlex.quote(payload_path)}",
                    timeout=30,
                )
                if append.exit_code != 0:
                    raise RuntimeError(
                        f"bridge_payload_append_failed:{append.stderr[-3000:]}"
                    )
            script = (
                "import base64,sys,urllib.request; "
                "data=base64.b64decode(open(sys.argv[1],'rb').read()); "
                "request=urllib.request.Request(sys.argv[2],data=data,headers={'Content-Type':'application/json'}); "
                f"\ntry:\n print(urllib.request.urlopen(request,timeout={int(timeout)}).read().decode())\nexcept urllib.error.HTTPError as exc:\n print(exc.read().decode(),file=sys.stderr)\n sys.exit(1)"
            )
            command = f"python3 -c {shlex.quote(script)} {shlex.quote(payload_path)} {shlex.quote(url)}"
            result = await sh(command, timeout=timeout + 10)
        finally:
            await sh(f"rm -f {shlex.quote(payload_path)}", timeout=30)
    if result.exit_code != 0:
        raise RuntimeError(f"bridge_{path}_failed:{(result.stderr or result.stdout)[-3000:]}")
    return json.loads(result.stdout)


async def _bridge_call(args, sh, path, payload, timeout=600):
    """Relay exact official embedding inputs via a private Vela model client."""
    operation = asyncio.create_task(_post(sh, path, payload, timeout=timeout))
    try:
        while not operation.done():
            polled = await _post(sh, "/embedding/poll", {}, timeout=30)
            for item in polled["requests"]:
                if item["model"] != "text-embedding-3-large":
                    raise ValueError("unexpected_embedding_model")
                result = {"id": item["id"]}
                try:
                    import inspect
                    create = args.llms["embedding"].client.embeddings.create
                    try:
                        response = create(
                            model=item["model"], input=item["input"],
                            encoding_format="float", timeout=180,
                        )
                    except TypeError:
                        response = create(model=item["model"], input=item["input"])
                    if inspect.isawaitable(response):
                        response = await response
                    rows = sorted(response.data, key=lambda row: row.index)
                    if len(rows) != len(item["input"]) or any(len(row.embedding) != 3072 for row in rows):
                        raise ValueError("embedding_shape_mismatch")
                    values = [value for row in rows for value in row.embedding]
                    result["vectors"] = base64.b64encode(struct.pack("<" + "d" * len(values), *values)).decode()
                except Exception as exc:
                    # Never copy SDK exception bodies/headers into tool logs.
                    result["error"] = type(exc).__name__ + ":" + str(exc)[:500].replace("\n", " ")
                await _post(sh, "/embedding/result", result, timeout=60)
            if not operation.done():
                await asyncio.sleep(0.5)
        return await operation
    finally:
        if not operation.done():
            operation.cancel()
            await asyncio.gather(operation, return_exceptions=True)


def _assistant(response):
    if response.get("choices"):
        return response["choices"][0]["message"]
    if response.get("message"):
        return response["message"]
    if response.get("role") == "assistant":
        return response
    raise ValueError(f"unexpected model response keys: {sorted(response)}")


def _normalize_calls(message, requestor):
    calls = []
    for raw in message.get("tool_calls") or []:
        function = raw.get("function") or {}
        arguments = function.get("arguments") or "{}"
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        calls.append({
            "id": raw.get("id") or f"call_{uuid.uuid4().hex}",
            "name": function.get("name", ""),
            "arguments": arguments,
            "requestor": requestor,
        })
    content = message.get("content")
    return {"role": requestor, "content": content, "tool_calls": calls or None}


def _trial_seed(args):
    rng = random.Random(BASE_SEED)
    seeds = [rng.randint(0, 1000000) for _ in range(NUM_TRIALS)]
    epoch = int(getattr(args, "epoch", 0) or 0)
    if not 0 <= epoch < NUM_TRIALS:
        raise ValueError(f"epoch {epoch} is outside the official {NUM_TRIALS}-trial range")
    return epoch, seeds[epoch]


def _compatibility_value(llm):
    compatibility = getattr(llm, "compatibility", None)
    return getattr(compatibility, "value", compatibility)


def _model_runtime(llm):
    """Report configured reasoning only; leave its application to the model.

    Gateway overrides must not be copied into the cross-protocol request.
    They are applied downstream by the gateway using the existing client headers.
    """
    protocols = {1: "chat_completions", 2: "anthropic", 4: "oai_responses"}
    compatibility = _compatibility_value(llm)
    if compatibility not in protocols:
        raise ValueError(f"unsupported tau3 model compatibility: {compatibility}")
    client = getattr(llm, "client", None)
    kwargs = getattr(llm, "kwargs", {}) or {}
    headers = {}
    for value in (
        kwargs.get("default_headers"), getattr(llm, "headers", None),
        getattr(llm, "default_headers", None), getattr(client, "default_headers", None),
    ):
        if hasattr(value, "items"):
            headers.update({str(k).lower(): v for k, v in value.items()})
    configured = {}
    status = "provider_default_unknown"
    features = headers.get("x-gateway-features")
    if features:
        try:
            body = json.loads(base64.b64decode(features)).get("set_req_body", {})
            for name, keys in (
                ("thinking", ("type", "budget_tokens")),
                ("output_config", ("effort",)), ("reasoning", ("effort",)),
            ):
                value = body.get(name)
                if isinstance(value, dict):
                    configured[name] = {
                        k: value[k] for k in keys
                        if k in value and isinstance(value[k], (str, int, bool))
                    }
            if isinstance(body.get("reasoning_effort"), str):
                configured["reasoning_effort"] = body["reasoning_effort"]
            status = "gateway_configured" if configured else status
        except (ValueError, TypeError, AttributeError):
            status = "gateway_config_unreadable"
    return {
        "model_id": getattr(llm, "model_id", getattr(llm, "id", None)),
        "protocol": protocols[compatibility],
        "reasoning_policy": "inherit_selected_model",
        "reasoning_status": status,
        "configured_reasoning": configured,
    }


def _seed_transport(llm, seed):
    """Mirror upstream LiteLLM drop_params=True for unsupported protocols."""
    compatibility = _compatibility_value(llm)
    if seed is None:
        return {"requested": None, "status": "not_requested", "compatibility": compatibility}
    if compatibility == 1:
        return {"requested": seed, "status": "sent", "compatibility": compatibility}
    if compatibility in {2, 4}:
        protocol = "anthropic" if compatibility == 2 else "oai_responses"
        return {
            "requested": seed,
            "status": "dropped_unsupported",
            "compatibility": compatibility,
            "reason": f"seed cannot be represented by {protocol}",
        }
    return {
        "requested": seed,
        "status": "dropped_unknown_compatibility",
        "compatibility": compatibility,
    }


async def _raw_completion(llm, request):
    # Ludus renders messages/tools for llm.compatibility. Model reasoning is
    # inherited via its client/gateway configuration, not forced by this adapter.
    _model_runtime(llm)
    # Artificial Analysis retries API failures up to 30 attempts. Disable
    # Ludus's nested four-attempt retry so this limit is exact and observable.
    if hasattr(llm, "retry_count"):
        llm.retry_count = 0
    for attempt in range(1, MAX_API_ATTEMPTS + 1):
        try:
            response = await llm.run_with_raw_request(
                request, response_format="chat_completions"
            )
            message = _assistant(response)
            if (
                not bool((message.get("content") or "").strip())
                and not message.get("tool_calls")
            ):
                # A completed API response with no usable participant message
                # is a malformed completion, not a model-under-test outcome.
                # Retry it inside AA's existing 30-attempt API failure budget.
                raise RuntimeError("empty_model_response")
            return response
        except (TypeError, ValueError):
            # Request-rendering/schema errors are deterministic, not API
            # failures, and must surface immediately.
            raise
        except Exception:
            if attempt >= MAX_API_ATTEMPTS:
                raise
            await asyncio.sleep(min(2 ** (attempt - 1), 10))


async def _completion(
    llm,
    messages,
    tools,
    *,
    seed=None,
    temperature=0.6,
    max_tokens=None,
    reasoning_effort=None,
):
    request = {
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto" if tools else None,
        "temperature": temperature,
        "timeout": 600,
    }
    if max_tokens is not None:
        request["max_tokens"] = max_tokens
    if reasoning_effort is not None:
        request["reasoning_effort"] = reasoning_effort
    if seed is not None and _compatibility_value(llm) == 1:
        request["seed"] = seed
    if request["tool_choice"] is None:
        request.pop("tool_choice")
        request.pop("tools")
    return await _raw_completion(llm, request)


def _chat_calls(item):
    return [
        {
            "id": call["id"],
            "type": "function",
            "function": {"name": call["name"], "arguments": json.dumps(call["arguments"], ensure_ascii=False)},
        }
        for call in item.get("tool_calls") or []
    ]


def _flip_for_user(history):
    flipped = []
    for item in history:
        if item["role"] == "user":
            value = {"role": "assistant", "content": item.get("content")}
            calls = _chat_calls(item)
            if calls:
                value["tool_calls"] = calls
            flipped.append(value)
        elif item["role"] == "assistant":
            if item.get("tool_calls"):
                raise ValueError("agent tool calls are not visible to user simulator")
            flipped.append({"role": "user", "content": item.get("content")})
        elif item["role"] == "tool":
            flipped.append({"role": "tool", "tool_call_id": item["id"], "content": item.get("content") or ""})
    return flipped


def _for_chat(item):
    if item["role"] == "tool":
        return {"role": "tool", "tool_call_id": item["id"], "content": item.get("content") or ""}
    result = {"role": item["role"], "content": item.get("content")}
    calls = _chat_calls(item)
    if calls:
        result["tool_calls"] = calls
    return result


async def _run_formal(args, sh, info):
    candidate = args.llms["candidate"]
    user_llm = args.llms["user"]
    agent_system = f"<instructions>\n{AGENT_INSTRUCTION}\n</instructions>\n<policy>\n{info['policy']}\n</policy>"
    user_system = f"{info['user_guidelines']}\n\n<scenario>\n{info['scenario']}\n</scenario>"
    greeting = {"role": "assistant", "content": "Hi! How can I help you today?"}
    trajectory = [greeting]
    agent_history = [greeting]
    user_history = []
    from_role = "assistant"
    to_role = "user"
    current = greeting
    errors = 0
    termination = None
    step_count = 0
    epoch, seed = _trial_seed(args)
    candidate_effort, _ = _run_config(args)

    while termination is None:
        # One official orchestrator step is exactly one participant generation or
        # one environment execution. The initial greeting is not a step.
        if to_role == "tool":
            for call in current["tool_calls"]:
                # Official Environment.get_response returns ordinary tool errors
                # as ToolMessage(error=True); bridge/transport failures propagate
                # as infrastructure failures and must not be converted to tool output.
                tool_result = await _bridge_call(args, sh, "/tool", call)
                if tool_result.get("error"):
                    errors += 1
                trajectory.append(tool_result)
                if from_role == "assistant":
                    agent_history.append(tool_result)
                else:
                    user_history.append(tool_result)
            to_role, from_role = from_role, "tool"
        elif to_role == "assistant":
            if from_role == "user" and not current.get("tool_calls"):
                agent_history.append(current)
            raw = await _completion(
                candidate,
                [{"role": "system", "content": agent_system}]
                + [_for_chat(x) for x in agent_history],
                info["agent_tools"],
                temperature=1.0,
                max_tokens=128000,
                seed=seed,
                reasoning_effort=candidate_effort,
            )
            current = _normalize_calls(_assistant(raw), "assistant")
            if (
                not bool((current.get("content") or "").strip())
                and not current.get("tool_calls")
            ):
                raise ValueError(
                    f"AssistantMessage must have either content or tool_calls: {current}"
                )
            agent_history.append(current)
            trajectory.append(current)
            from_role = "assistant"
            to_role = "tool" if current.get("tool_calls") else "user"
        elif to_role == "user":
            if from_role == "assistant" and not current.get("tool_calls"):
                user_history.append(current)
            raw = await _completion(
                user_llm,
                [{"role": "system", "content": user_system}]
                + _flip_for_user(user_history),
                info["user_tools"],
                seed=seed,
            )
            current = _normalize_calls(_assistant(raw), "user")
            if (
                not bool((current.get("content") or "").strip())
                and not current.get("tool_calls")
            ):
                raise ValueError(
                    f"UserMessage must have either content or tool_calls: {current}"
                )
            user_history.append(current)
            trajectory.append(current)
            from_role = "user"
            to_role = "tool" if current.get("tool_calls") else "assistant"
            # UserSimulator.is_stop returns False for every tool-call message,
            # including mixed text+tool messages. The ordinary LLMAgent never
            # stops on content markers.
            if not current.get("tool_calls") and any(
                marker in (current.get("content") or "") for marker in STOP_MARKERS
            ):
                termination = "user_stop"
        else:
            raise ValueError(f"invalid official route: {from_role} -> {to_role}")

        step_count += 1
        # Official order: skip limits while waiting for ENV; otherwise max
        # steps, then errors (errors override max_steps when both are reached).
        if to_role != "tool":
            if step_count >= MAX_STEPS:
                termination = "max_steps"
            if errors >= MAX_ERRORS:
                termination = "too_many_errors"

    seed_transport = {
        "candidate": _seed_transport(candidate, seed),
        "user": _seed_transport(user_llm, seed),
    }
    return trajectory, termination, errors, step_count, epoch, seed, seed_transport


def _judge_json(response):
    # Upstream v1.0.1 directly parses assistant_message.content. Do not repair
    # fenced JSON or alternate content shapes, because that changes evaluator
    # failure semantics.
    content = _assistant(response).get("content")
    if not isinstance(content, str):
        raise TypeError(
            f"NL judge content must be a string, got {type(content).__name__}"
        )
    return json.loads(content)


async def _nl_reward(args, trajectory, assertions):
    if not assertions:
        return 1.0, []
    trajectory_str = "\n".join(f"{x['role']}: {x.get('content')}" for x in trajectory)
    user_prompt = f"""
        conversation:
        {trajectory_str}

        expectedOutcomes:
        {assertions}
        """
    request = {
        "messages": [{"role": "system", "content": NL_SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}],
        "temperature": 0.6,
        "timeout": 600,
    }
    raw = await _raw_completion(args.llms["judge"], request)
    results = _judge_json(raw).get("results", [])
    # Match upstream v1.0.1 exactly: it scores the returned checks and does not
    # separately require the judge to echo every input assertion.
    return float(all(bool(x.get("metExpectation")) for x in results)), results


async def _grade(args, sh, trajectory, termination, errors):
    if termination not in {"agent_stop", "user_stop"}:
        return {"task_result": None, "reward": 0.0, "score": 0.0, "termination_reason": termination, "num_errors": errors, "messages": trajectory}
    official = await _bridge_call(args, sh, "/grade", {"messages": trajectory})
    _, extra = _record(args)
    nl_score, nl_checks = await _nl_reward(args, trajectory, extra.get("nl_assertions") or [])
    reward = float(official["reward_without_nl"]) * nl_score
    return {
        "task_result": None,
        "reward": reward,
        "score": reward,
        "termination_reason": termination,
        "num_errors": errors,
        "reward_basis": official["reward_basis"],
        "reward_components": {**official["components"], "nl_assertion": nl_score} if extra.get("nl_assertions") else official["components"],
        "db_check": official["db_check"],
        "action_checks": official["action_checks"],
        "nl_assertion_checks": nl_checks,
        "messages": trajectory,
    }


async def run(args) -> dict[str, Any]:
    if _mode(args) == "embedding_transport_verify":
        response = await args.llms["embedding"].client.embeddings.create(
            model="text-embedding-3-large", input=["bank account"],
            encoding_format="float", timeout=30,
        )
        if not response.data:
            raise RuntimeError("embedding_transport_no_vectors")
        dimensions = len(response.data[0].embedding)
        if dimensions != 3072:
            raise RuntimeError("embedding_transport_wrong_dimensions")
        return {"reward": 1.0, "control_only": True, "dimensions": dimensions,
                "process_revision": "alltools-v11-direct-transport-probe"}
    _, retrieval_variant = _run_config(args)
    model_runtime = {name: _model_runtime(llm) for name, llm in args.llms.items()}
    _, extra = _record(args)
    sh = args.sandbox.shell
    await _ensure_bridge(sh, extra["runtime_asset_url"])
    info = await _bridge_call(args, sh, "/init", {
        "task_id": extra["upstream_task_id"],
        "retrieval_variant": retrieval_variant,
        "retrieval_config": retrieval_variant,
    })
    # Sierra-aligned AllTools contract: fail the record before any model call
    # if the runtime did not resolve the expected retrieval tool surface.
    expected = {"KB_search_bm25", "KB_search_dense", "shell"}
    actual = {str(t.get("name") or (t.get("function") or {}).get("name")) for t in (info.get("agent_tools") or []) if isinstance(t, dict)}
    if retrieval_variant == "alltools" and not expected.issubset(actual):
        raise RuntimeError(f"retrieval_tool_contract_failed:expected={sorted(expected)} actual={sorted(actual)}")
    mode = _mode(args)
    if mode == "runtime_verify":
        probes = {}
        for name, arguments in (
            ("KB_search_bm25", {"query": "credit card refund", "k": 2}),
            ("KB_search_dense", {"query": "credit card refund", "k": 2}),
            ("shell", {"command": "printf alltools_shell_probe"}),
        ):
            response = await _bridge_call(args, sh, "/tool", {
                "requestor": "assistant", "id": "probe_" + name,
                "name": name, "arguments": arguments,
            })
            if response.get("error") or not response.get("content"):
                raise RuntimeError("runtime_probe_failed:" + name)
            if name == "shell" and "alltools_shell_probe" not in str(response["content"]):
                raise RuntimeError("runtime_shell_probe_failed")
            if name == "shell" and "Operation not permitted" in str(response["content"]):
                raise RuntimeError("runtime_shell_namespace_failed")
            probes[name] = response
        return {"reward": 1.0, "score": 1.0, "control_only": True,
                "probes": probes, "effective_retrieval_tools": sorted(actual),
                "process_revision": "alltools-v10-embedding-diagnostics"}
    if mode == "formal":
        trajectory, termination, errors, step_count, epoch, seed, seed_transport = await _run_formal(args, sh, info)
    elif mode == "gold_verify":
        gold = await _post(sh, "/gold", {})
        trajectory, termination, errors = gold["messages"], gold["termination_reason"], 0
        step_count, epoch, seed, seed_transport = 0, int(getattr(args, "epoch", 0) or 0), None, {}
    elif mode in {"dummy_verify", "dummy"}:
        trajectory = [
            {"role": "assistant", "content": "Hi! How can I help you today?"},
            {"role": "user", "content": "I need help."},
            {"role": "assistant", "content": "I cannot determine the answer."},
        ]
        termination, errors = "max_steps", 0
        step_count, epoch, seed, seed_transport = 0, int(getattr(args, "epoch", 0) or 0), None, {}
    else:
        return {"task_result": None, "reward": 0.0, "score": 0.0, "error": f"unknown_mode:{mode}"}
    result = await _grade(args, sh, trajectory, termination, errors)
    result.update(upstream_task_id=extra["upstream_task_id"], expected_trials=int(extra.get("num_trials", 4)),
                  candidate_reasoning_effort=_run_config(args)[0],
                  step_count=step_count, trial=epoch, seed=seed,
                  seed_transport=seed_transport, model_runtime=model_runtime,
                  effective_retrieval_tools=sorted(actual), retrieval_variant=retrieval_variant,
                  process_revision="alltools-v10-embedding-diagnostics")
    return result


async def calculate(results: list[list[dict[str, Any]]]) -> dict[str, float]:
    import math
    items = [item for epoch in results for item in epoch]
    scores = [float(item.get("score", item.get("reward", 0.0))) for item in items]
    metrics = {"total": float(len(scores)), "avg_score": sum(scores) / len(scores) if scores else 0.0}
    grouped = {}
    for item in items:
        task_id = item.get("upstream_task_id")
        if task_id is None:
            continue
        trials = grouped.setdefault(task_id, {})
        trial = int(item["trial"])
        if trial in trials:
            raise ValueError("duplicate_task_trial")
        trials[trial] = float(item.get("score", item.get("reward", 0.0)))
    if grouped:
        n = len(next(iter(grouped.values())))
        if all(set(t) == set(range(n)) for t in grouped.values()):
            for k in range(1, n + 1):
                metrics[f"pass^{k}"] = sum(
                    math.comb(sum(v == 1.0 for v in trials.values()), k) / math.comb(n, k)
                    for trials in grouped.values()
                ) / len(grouped)
    return metrics
