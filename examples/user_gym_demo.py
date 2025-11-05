"""
Demo script for UserGymEnv - Playing as the user against an automated agent.

This script demonstrates how to use the UserGymEnv to control the user's actions
while an LLMAgent responds automatically.
"""
import gymnasium as gym
from tau2.gym import register_gym_agent, TAU_BENCH_USER_ENV_ID


def main():
    """Run a simple user gym environment demo."""
    # Register the gym environments
    register_gym_agent()

    # Create a UserGymEnv - you control the user, agent is automated
    print("Creating UserGymEnv for mock/create_task_1...")
    env = gym.make(
        TAU_BENCH_USER_ENV_ID,
        domain="mock",
        task_id="create_task_1",
        agent_llm="gpt-4o-mini",  # Use a fast model for demo
        agent_llm_args={"temperature": 0.7},
    )

    # Reset the environment
    print("\n" + "=" * 80)
    print("RESET: Starting new conversation")
    print("=" * 80)
    observation, info = env.reset()

    print(f"\nAgent says: {observation}")
    print(f"\nTask: {info['task'].id}")
    print(f"Agent has {len(info['agent_tools'])} tools available")
    print(f"User has {len(info['user_tools'])} tools available")

    # Simulate a conversation
    conversation = [
        "I need to create a new task",
        "Call it 'Important Meeting' for user_1",
    ]

    for i, user_action in enumerate(conversation, 1):
        print("\n" + "-" * 80)
        print(f"STEP {i}")
        print("-" * 80)
        print(f"You (user): {user_action}")

        # Take a step
        observation, reward, terminated, truncated, info = env.step(user_action)

        print(f"\nAgent says: {observation}")
        print(f"Reward: {reward}")
        print(f"Terminated: {terminated}")

        if terminated:
            print("\n" + "=" * 80)
            print("CONVERSATION ENDED")
            print("=" * 80)
            break

    # If not terminated, close properly
    if not terminated:
        print("\n" + "=" * 80)
        print("Demo complete - conversation ongoing")
        print("=" * 80)


if __name__ == "__main__":
    main()

