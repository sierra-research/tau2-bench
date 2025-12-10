"""Launcher module - initiates and coordinates the evaluation process."""

import json
import multiprocessing

from agentify_tau_bench.green_agent.agent import start_green_agent
from agentify_tau_bench.utils import a2a_send_message, wait_agent_ready
from agentify_tau_bench.white_agent.agent import start_white_agent
from loguru import logger


async def launch_evaluation():
    # start green agent
    logger.info("Launching green agent...")
    green_address = ("localhost", 9001)
    green_url = f"http://{green_address[0]}:{green_address[1]}"
    p_green = multiprocessing.Process(
        target=start_green_agent, args=("tau_green_agent", *green_address)
    )
    p_green.start()
    assert await wait_agent_ready(green_url), "Green agent not ready in time"
    logger.info("Green agent is ready.")

    # start white agent
    logger.info("Launching white agent...")
    white_address = ("localhost", 9002)
    white_url = f"http://{white_address[0]}:{white_address[1]}"
    p_white = multiprocessing.Process(
        target=start_white_agent, args=("general_white_agent", *white_address)
    )
    p_white.start()
    assert await wait_agent_ready(white_url), "White agent not ready in time"
    logger.info("White agent is ready.")

    # send the task description
    logger.info("Sending task description to green agent...")
    task_config = {
        "domain": "mock",
        "task_ids": None,
        "max_steps": 20,
        "user_llm": "openrouter/openai/gpt-4o",
        "user_llm_args": {"temperature": 0.0, "custom_llm_provider": "litellm_proxy"},
    }
    task_text = f"""
Your task is to instantiate tau-bench to test the agent located at:
<white_agent_url>
http://{white_address[0]}:{white_address[1]}/
</white_agent_url>
You should use the following env configuration:
<env_config>
{json.dumps(task_config, indent=2)}
</env_config>
    """
    print("Task description:")
    print(task_text)
    print("Sending...")
    response = await a2a_send_message(green_url, task_text)
    print("Response from green agent:")
    print(response.model_dump_json(indent=2))

    print("Evaluation complete. Terminating agents...")
    p_green.terminate()
    p_green.join()
    p_white.terminate()
    p_white.join()
    print("Agents terminated.")
