"""Green agent implementation - manages assessment and evaluation."""

import asyncio
import json
import os
import time
import tomllib
from typing import Optional

import dotenv
import gymnasium as gym
import uvicorn
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, Message, SendMessageSuccessResponse
from a2a.utils import get_text_parts, new_agent_text_message
from agentify_tau_bench.utils import a2a_send_message, parse_tags
from loguru import logger
from pydantic import BaseModel, Field

from tau2.data_model.simulation import RewardInfo, SimulationRun
from tau2.environment.tool import Tool
from tau2.gym import TAU_BENCH_ENV_ID, register_gym_agent
from tau2.run import get_tasks

dotenv.load_dotenv()

RESPOND_ACTION_NAME = "respond"

# Register the environments (only needed once)
register_gym_agent()


def load_agent_card_toml(agent_name):
    current_dir = __file__.rsplit("/", 1)[0]
    with open(f"{current_dir}/{agent_name}.toml", "rb") as f:
        return tomllib.load(f)


def tools_to_str(tools: list[Tool]) -> str:
    return json.dumps([tool.openai_schema for tool in tools], indent=2)


def get_task_ids(domain: str, task_ids: Optional[list[str]]) -> list[str]:
    """
    Get the task IDs for the domain.
    If task_ids are not specified, it will get all tasks for the domain.
    Else, it will validate the task IDs against the domain
    """
    task_set_name = domain
    task_split_name = "base"
    if task_ids is None:
        tasks = get_tasks(task_set_name=task_set_name, task_split_name=task_split_name)
    else:
        tasks = get_tasks(
            task_set_name=task_set_name,
            task_split_name=task_split_name,
            task_ids=task_ids,
        )

    return [task.id for task in tasks]


class EnvConfig(BaseModel):
    domain: str = Field(description="The domain to run the simulation on")
    task_ids: Optional[list[str]] = Field(
        description="The task IDs to run the simulation. If None, will run all tasks for the domain.",
        default=None,
    )
    max_steps: Optional[int] = Field(
        description="The maximum number of steps to run the simulation. Default is 100.",
        default=100,
    )
    user_llm: Optional[str] = Field(
        description="The LLM to use for the user simulator. If None, will use the default LLM for the domain.",
        default=None,
    )
    user_llm_args: Optional[dict] = Field(
        description="The arguments to pass to the user simulator LLM. If None, will use the default arguments for the domain.",
        default=None,
    )


def get_green_agent_config(user_input: str) -> tuple[str, EnvConfig]:
    """
    Get the green agent configuration from the user input.

    Args:
        user_input: The user input string.

    Returns:
        A tuple containing the white agent URL and the environment configuration.
    """
    tags = parse_tags(user_input)
    try:
        white_agent_url = tags["white_agent_url"]
    except Exception as e:
        raise ValueError(f"Error parsing white agent URL: {e}")

    try:
        env_config_json = tags["env_config"]
    except Exception as e:
        raise ValueError(f"Error parsing env config: {e}")
    try:
        env_config = EnvConfig.model_validate_json(env_config_json)
    except Exception as e:
        raise ValueError(f"Error parsing environment config: {e}")

    if env_config.task_ids is None:
        env_config.task_ids = get_task_ids(domain=env_config.domain, task_ids=None)
    else:
        env_config.task_ids = get_task_ids(
            domain=env_config.domain, task_ids=env_config.task_ids
        )

    return white_agent_url, env_config


async def ask_agent_to_solve(
    white_agent_url: str,
    env: gym.Env,
    max_retries: int = 3,
) -> Optional[SimulationRun]:
    terminated = False
    context_id = None
    observation, info = env.reset()
    # Access available tools and policy from info

    # Here, instead of calling white agent like calling an LLM, we need to present
    #   the assessment scenario to the white agent as if it is a independent task
    # Specifically, here we provide the tool information for the agent to reply with
    task_description = f"""
{info["policy"]}
Here's a list of tools you can use (you can use at most one tool at a time):
{tools_to_str(info["tools"])}
Please response in the JSON format. Please wrap the JSON part with <json>...</json> tags.
The JSON should contain:
- "name": the tool call function name, or "{
        RESPOND_ACTION_NAME
    }" if you want to respond directly.
- "arguments": the arguments for the tool call, or {{"content": "your message here"}} if you want to respond directly.
You should only use one tool at a time!!
You cannot respond to user and use a tool at the same time!!

Examples of responses:
<json>
{
        json.dumps(
            {
                "name": "find_user_id_by_name_zip",
                "arguments": {
                    "first_name": "Yusuf",
                    "last_name": "Rossi",
                    "zip_code": "19122",
                },
            },
            indent=2,
        )
    }
</json>

<json>
{
        json.dumps(
            {
                "name": "{RESPOND_ACTION_NAME}",
                "arguments": {"content": "Hello, how can I help you today?"},
            },
            indent=2,
        )
    }
</json>

Next, I'll provide you with the user message and tool call results.
User message: {json.dumps(observation, indent=2)}
    """
    next_green_message = task_description
    while not terminated:
        logger.info(
            f"@@@ Green agent: Sending message to white agent{'ctx_id=' + str(context_id) if context_id else ''}... -->\n{next_green_message}"
        )

        # Retry logic for white agent communication
        white_agent_response = None
        for attempt in range(max_retries):
            try:
                white_agent_response = await a2a_send_message(
                    white_agent_url, next_green_message, context_id=context_id
                )
                break  # Success, exit retry loop
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2**attempt)  # Exponential backoff
                else:
                    logger.error(
                        f"All {max_retries} attempts failed to reach white agent"
                    )
                    raise

        if white_agent_response is None:
            logger.error("Failed to get response from white agent after retries")
            return None

        res_root = white_agent_response.root
        assert isinstance(res_root, SendMessageSuccessResponse), (
            f"Expected SendMessageSuccessResponse, got {type(res_root)}"
        )
        res_result = res_root.result
        assert isinstance(res_result, Message), (
            f"Expected Message, got {type(res_result)}"
        )
        if context_id is None:
            context_id = res_result.context_id
        else:
            assert context_id == res_result.context_id, (
                "Context ID should remain the same in a conversation"
            )

        text_parts = get_text_parts(res_result.parts)
        assert len(text_parts) == 1, (
            "Expecting exactly one text part from the white agent"
        )
        white_text = text_parts[0]
        logger.info(f"@@@ White agent response:\n{white_text}")
        # parse the action out
        white_tags = parse_tags(white_text)
        logger.info(f"@@@ White agent tags: {white_tags}")
        action_json = white_tags["json"]
        action_dict = json.loads(action_json)
        is_tool_call = action_dict["name"] != RESPOND_ACTION_NAME
        if not is_tool_call:
            action = action_dict["arguments"]["content"]
        else:
            action = json.dumps(action_dict)

        observation, reward, terminated, truncated, info = env.step(action)
        logger.info(f"@@@ Green agent: Observation: {observation}")
        logger.info(f"@@@ Green agent: Reward: {reward}")
        next_green_message = observation

        # instead of maintain history, just prepare the next message with the latest observation
        if terminated:
            break
        # TODO: We need to reset the green agent!!!

    if info["simulation_run"] is not None:
        simulation_run = SimulationRun.model_validate_json(info["simulation_run"])
    else:
        simulation_run = None
    if info["reward_info"] is not None:
        reward_info = RewardInfo.model_validate_json(info["reward_info"])
        simulation_run.reward_info = reward_info
    return simulation_run


CONCURRENCY_LIMIT = 2  # Reduced from 10 to avoid overwhelming the white agent


class TauGreenAgentExecutor(AgentExecutor):
    def __init__(self):
        self.semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # parse the task
        logger.info("Green agent: Received a task, parsing...")
        user_input = context.get_user_input()
        white_agent_url, env_config = get_green_agent_config(user_input)

        # set up the environment
        logger.info("Green agent: Setting up the environment...")
        logger.info(f"Green agent: White agent URL: {white_agent_url}")
        logger.info(
            f"Green agent: Environment configuration: {env_config.model_dump_json(indent=2)}"
        )
        logger.info("Green agent: Starting evaluation...")
        metrics = {"info": {}, "tasks": {}}
        timestamp_started = time.time()

        async def run_one_task(task_id: str) -> None:
            async with self.semaphore:
                try:
                    logger.info(f"Green agent: Running task {task_id}...")
                    task_env_config = {
                        "domain": env_config.domain,
                        "task_id": task_id,
                        "max_steps": env_config.max_steps,
                        "user_llm": env_config.user_llm,
                        "user_llm_args": env_config.user_llm_args,
                        "all_messages_as_observation": False,
                    }
                    env = gym.make(TAU_BENCH_ENV_ID, **task_env_config)
                    res = await ask_agent_to_solve(
                        white_agent_url,
                        env,
                    )
                    if res is not None and res.reward_info is not None:
                        metrics["tasks"][task_id] = res.reward_info.reward
                    else:
                        logger.error(
                            f"Green agent: Task {task_id} returned None or missing reward_info"
                        )
                        metrics["tasks"][task_id] = 0
                except Exception as e:
                    logger.error(f"Green agent: Error running task {task_id}: {e}")
                    metrics["tasks"][
                        task_id
                    ] = -1  # TODO: This is just a placeholder, we need to handle this properly

        tasks = [run_one_task(task_id) for task_id in env_config.task_ids]
        logger.info(f"Green agent: Running {len(tasks)} tasks...")
        await asyncio.gather(*tasks)

        metrics["info"]["time_used"] = time.time() - timestamp_started
        metrics["info"]["success"] = all(
            metrics["tasks"][task_id] == 1 for task_id in env_config.task_ids
        )

        logger.info("Green agent: Evaluation complete.")
        await event_queue.enqueue_event(
            new_agent_text_message(
                f"Finished. White agent success: {metrics['info']['success']}\nMetrics: {json.dumps(metrics, indent=2)}\n"
            )
        )  # alternative, impl as a task-generating agent

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError


def start_green_agent(agent_name="tau2_green_agent", host="localhost", port=9001):
    """
    Start the green agent.

    Args:
        agent_name: The name of the green agent.
        host: The host to run the green agent on.
        port: The port to run the green agent on.
    """
    logger.info("Starting green agent...")
    agent_card_dict = load_agent_card_toml(agent_name)

    # url = f"http://{host}:{port}"
    # agent_card_dict["url"] = url  # complete all required card fields
    agent_card_dict["url"] = os.getenv("AGENT_URL")

    request_handler = DefaultRequestHandler(
        agent_executor=TauGreenAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=AgentCard(**agent_card_dict),
        http_handler=request_handler,
    )

    uvicorn.run(app.build(), host=host, port=port)
