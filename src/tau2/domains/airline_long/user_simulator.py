"""User simulator for the airline_long domain with noisy user messages."""

import hashlib
import random
from typing import Optional, Tuple

from tau2.data_model.message import Message, UserMessage
from tau2.environment.tool import Tool
from tau2.user.base import UserState, ValidUserInputMessage
from tau2.user.user_simulator import UserSimulator


# ============================================================================
# USER MESSAGE NOISE GENERATION UTILITIES
# These functions add irrelevant noise to user messages, simulating the kind of
# distracting but harmless content that doesn't affect task completion.
# All functions take a Random instance (rng) for deterministic output.
# ============================================================================


def _generate_weather_comment(rng: random.Random) -> str:
    """Generate irrelevant weather-related small talk."""
    weather_types = [
        "sunny", "cloudy", "rainy", "windy", "humid", "chilly", "warm", "foggy"
    ]
    activities = [
        "went for a walk", "stayed inside", "had coffee on the porch",
        "watched the clouds", "enjoyed the breeze", "read a book by the window"
    ]
    weather = rng.choice(weather_types)
    activity = rng.choice(activities)
    return f"By the way, it's so {weather} here today - I {activity} earlier."


def _generate_personal_tangent(rng: random.Random) -> str:
    """Generate irrelevant personal anecdotes."""
    tangents = [
        "My cat just jumped on my keyboard, sorry if there are any typos!",
        "I'm multitasking while making dinner, hope that's okay.",
        "Just had to let my dog out, back now.",
        "Sorry, my phone battery is at 15%, charging it now.",
        "I'm sitting in my favorite coffee shop right now, love their lattes.",
        "My neighbor's lawn mower is so loud, can barely think!",
        "Just realized I forgot to water my plants this morning.",
        "My kid is watching cartoons in the background, excuse any noise.",
        "I've been meaning to reorganize my bookshelf all week.",
        "Just finished my second cup of tea today.",
        "My coworker keeps pinging me on Slack, one sec.",
        "I should really clean my desk, it's getting cluttered.",
        "The traffic outside my window has been crazy today.",
        "I've been binge-watching that new show everyone's talking about.",
        "My subscription to that magazine finally arrived!",
    ]
    return rng.choice(tangents)


def _generate_filler_phrase(rng: random.Random) -> str:
    """Generate conversational filler phrases."""
    fillers = [
        "Anyway, back to what I was saying...",
        "So yeah, where were we?",
        "Right, so...",
        "Okay, moving on...",
        "Anyway...",
        "So, um...",
        "Let me think...",
        "Hmm, let's see...",
        "Oh, right!",
        "Actually, wait...",
    ]
    return rng.choice(fillers)


def _generate_gratitude_noise(rng: random.Random) -> str:
    """Generate excessive gratitude or politeness."""
    phrases = [
        "Thanks so much for your patience with me!",
        "I really appreciate your help with this!",
        "You've been super helpful, thank you!",
        "Sorry for all the questions!",
        "Hope I'm not being too much trouble!",
        "Thanks for bearing with me here!",
        "I know this might be a lot to ask!",
        "Really grateful for your assistance!",
    ]
    return rng.choice(phrases)


def _generate_time_comment(rng: random.Random) -> str:
    """Generate irrelevant comments about time."""
    comments = [
        f"Can't believe it's already {rng.randint(1, 12)}pm!",
        "Time really flies, doesn't it?",
        "I've been meaning to do this all week.",
        "Finally got around to sorting this out!",
        "Been putting this off for days.",
        f"I've been on hold for like {rng.randint(5, 30)} minutes before calling you.",
        "Just woke up from a nap, still a bit groggy.",
        "Trying to get this done before my meeting at " + f"{rng.randint(1, 5)}.",
    ]
    return rng.choice(comments)


def _generate_tech_complaint(rng: random.Random) -> str:
    """Generate irrelevant tech complaints."""
    complaints = [
        "My wifi has been so spotty lately!",
        "Sorry, my browser froze for a second there.",
        "Had to restart my computer earlier, so annoying.",
        "The app keeps logging me out for some reason.",
        "My phone's autocorrect is driving me crazy.",
        "Just updated my phone and everything looks different now.",
        "My laptop fan is making weird noises today.",
    ]
    return rng.choice(complaints)


def _generate_life_update(rng: random.Random) -> str:
    """Generate irrelevant life updates."""
    updates = [
        "Just got back from the gym, feeling energized!",
        "I've been trying to eat healthier this month.",
        "Started learning guitar last week, fingers hurt!",
        "My friend recommended this great podcast recently.",
        "Been trying to read more books this year.",
        "Just finished redecorating my living room!",
        "I've been getting into gardening lately.",
        "Started a new workout routine, pretty sore today.",
        "Been meaning to learn how to cook Thai food.",
        "Just adopted a new houseplant, hope I don't kill it!",
    ]
    return rng.choice(updates)


def _generate_random_aside(rng: random.Random) -> str:
    """Generate random asides and parenthetical comments."""
    asides = [
        "(hope that makes sense)",
        "(if that's even possible)",
        "(not sure if I'm explaining this right)",
        "(you probably get this all the time)",
        "(sorry for rambling)",
        "(is that the right term?)",
        "(I think I'm remembering that correctly)",
        "(pardon my typing)",
    ]
    return rng.choice(asides)


def _add_noise_to_user_message(content: str, seed_key: str) -> str:
    """Add irrelevant noise to a user message.

    Args:
        content: The actual user message content
        seed_key: A string to ensure deterministic noise generation

    Returns:
        The message with added noise
    """
    if not content or content.strip() == "":
        return content

    # Create a local Random instance seeded based on the message content
    seed_value = int(hashlib.md5(seed_key.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed_value)

    # Select which noise types to add (deterministically based on seed)
    noise_generators = [
        _generate_weather_comment,
        _generate_personal_tangent,
        _generate_filler_phrase,
        _generate_gratitude_noise,
        _generate_time_comment,
        _generate_tech_complaint,
        _generate_life_update,
        _generate_random_aside,
    ]

    # Shuffle generators deterministically
    rng.shuffle(noise_generators)

    # Decide how many noise elements to add (1-3)
    num_noise_elements = rng.randint(1, 3)

    # Build the noisy message
    noise_parts = []

    # Sometimes add noise at the beginning
    if rng.random() < 0.4:
        noise_parts.append(noise_generators[0](rng))

    # Add the actual content
    noise_parts.append(content)

    # Add noise elements after the main content
    for i in range(1, min(num_noise_elements + 1, len(noise_generators))):
        if rng.random() < 0.5:
            noise_parts.append(noise_generators[i](rng))

    return " ".join(noise_parts)


class NoisyUserSimulator(UserSimulator):
    """User simulator that adds irrelevant noise to user messages."""

    def __init__(
        self,
        tools: Optional[list[Tool]] = None,
        instructions: Optional[str] = None,
        llm: Optional[str] = None,
        llm_args: Optional[dict] = None,
    ):
        super().__init__(
            tools=tools,
            instructions=instructions,
            llm=llm,
            llm_args=llm_args,
        )
        self._message_counter = 0

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> UserState:
        """Get the initial state of the user simulator."""
        self._message_counter = 0
        return super().get_init_state(message_history)

    def generate_next_message(
        self, message: ValidUserInputMessage, state: UserState
    ) -> Tuple[UserMessage, UserState]:
        """Generate the next message with added noise."""
        # Get the base message from the parent class
        user_message, new_state = super().generate_next_message(message, state)

        # Add noise to the content if it exists and is not a tool call
        if user_message.content and not user_message.is_tool_call():
            # Use message counter and content hash for deterministic seed
            seed_key = f"user_msg_{self._message_counter}:{user_message.content[:50]}"
            user_message.content = _add_noise_to_user_message(
                user_message.content, seed_key
            )

        self._message_counter += 1
        return user_message, new_state
