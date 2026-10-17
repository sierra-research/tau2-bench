import re
from decimal import Decimal

from tau2.data_model.message import AssistantMessage, Message, Tick
from tau2.data_model.simulation import CommunicateCheck, RewardInfo
from tau2.data_model.tasks import RewardType, Task
from tau2.evaluator.evaluator_base import EvaluatorBase

_SMALL = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "zero ten twenty thirty forty fifty sixty seventy eighty ninety".split()


def _integer_words(value: int) -> str:
    """Spell a nonnegative integer using standard English cardinal numbers."""
    if value < 20:
        return _SMALL[value]
    if value < 100:
        return _TENS[value // 10] + (" " + _SMALL[value % 10] if value % 10 else "")
    for scale, name in (
        (10**12, "trillion"),
        (10**9, "billion"),
        (10**6, "million"),
        (1000, "thousand"),
        (100, "hundred"),
    ):
        if value >= scale:
            return (
                _integer_words(value // scale)
                + " "
                + name
                + (" " + _integer_words(value % scale) if value % scale else "")
            )
    raise ValueError(value)


def _communicated(info: str, content: str) -> bool:
    """Match integer values in digits or words; retain legacy matching for prose."""
    expected = info.lower().replace(",", "").strip()
    text = content.lower().replace(",", "")
    if not re.fullmatch(r"[0-9]+", expected):
        return info.lower() in text
    # Do not let a required 4 match 5244, 4.5, or an identifier containing 4.
    for match in re.finditer(
        r"(?<![\w.-])" + re.escape(expected) + r"(?!\w|\.[0-9]|-[0-9])", text
    ):
        if not re.search(r"(?:\bminus|\bnegative|-)\s*\$?\s*$", text[: match.start()]):
            return True
    value = int(expected)
    # Leading-zero strings are identifiers, not cardinal numbers.
    if str(value) != expected or value >= 10**15:
        return False
    text = re.sub(r"(?<!\w)-(?=[a-z])", "minus ", text)
    text = re.sub(r"\band\b", " ", text.replace("-", " "))
    text = re.sub(r"\s+", " ", text)
    forms = [_integer_words(value)]
    # Common spoken dollar amounts: "twelve eighty-six" for 1286.
    if 1000 <= value < 10000 and value % 100 >= 10:
        forms.append(_integer_words(value // 100) + " " + _integer_words(value % 100))
    for form in forms:
        for match in re.finditer(r"\b" + re.escape(form) + r"\b", text):
            # Reject a proper subphrase of a larger spoken number.
            before = text[: match.start()].split()
            after = text[match.end() :].split()
            number_words = set(
                _SMALL
                + _TENS
                + ["hundred", "thousand", "million", "billion", "trillion", "point"]
            )
            if (before and before[-1] in number_words | {"minus", "negative"}) or (
                after and after[0] in number_words
            ):
                continue
            return True
    return False


class CommunicateEvaluator(EvaluatorBase[Message]):
    """
    Evaluates whether or not the agent communicated the required information.
    """

    @classmethod
    def calculate_reward(
        cls,
        task: Task,
        full_trajectory: list[Message],
    ) -> RewardInfo:
        """
        Calculate the reward based on whether the agent communicated the required information.
        """
        if task.evaluation_criteria is None:
            return RewardInfo(
                reward=1.0,
                info={"notes": "No evaluation criteria"},
                reward_breakdown={RewardType.COMMUNICATE: 1.0},
            )
        communicate_info = task.evaluation_criteria.communicate_info
        if not communicate_info:
            return RewardInfo(
                reward=1.0,
                info={"note": "No communicate_info to evaluate"},
                reward_breakdown={RewardType.COMMUNICATE: 1.0},
            )

        communicate_info_checks = cls.evaluate_communicate_info(
            full_trajectory, communicate_info
        )

        # Calculate reward: 1 if all expectations are met, 0 otherwise
        all_expectations_met = all(result.met for result in communicate_info_checks)
        reward = 1.0 if all_expectations_met else 0.0

        return RewardInfo(
            reward=reward,
            communicate_checks=communicate_info_checks,
            reward_breakdown={RewardType.COMMUNICATE: reward},
        )

    @classmethod
    def evaluate_communicate_info(
        cls,
        full_trajectory: list[Message],
        communicate_info: list[str],
    ) -> list[CommunicateCheck]:
        """
        Evaluate whether the agent communicates the information correctly.
        """
        if len(communicate_info) == 0:
            return []

        outputs = []
        for info_str in communicate_info:
            found = False
            for message in full_trajectory:
                if not isinstance(message, AssistantMessage):
                    continue
                if not message.has_text_content():
                    continue
                if _communicated(info_str, message.content):
                    found = True
                    break
            if found:
                met = True
                justification = f"Information '{info_str}' communicated in the message:\n '{message.content}'"
            else:
                met = False
                justification = f"Information '{info_str}' not communicated."
            outputs.append(
                CommunicateCheck(
                    info=info_str,
                    met=met,
                    justification=justification,
                )
            )
        return outputs


class FullDuplexCommunicateEvaluator(EvaluatorBase[Tick]):
    @classmethod
    def ticks_to_message_history(cls, ticks: list[Tick]) -> list[AssistantMessage]:
        """
        Convert a list of Ticks to a list of AssistantMessages by extracting and merging agent chunks.

        Chunks with overlapping utterance_ids are merged into single messages.
        Missing IDs fall back to contiguous text, separated by tools or a one-second pause.
        This groups consecutive chunks that belong to the same utterance(s).

        Args:
            ticks: List of Tick objects from full-duplex simulation.

        Returns:
            List of AssistantMessages, where chunks with overlapping utterance_ids
            have been merged together.
        """
        messages: list[AssistantMessage] = []
        group: list[AssistantMessage] = []
        utterance_ids: set[str] = set()
        silence = Decimal(0)

        def flush() -> None:
            if group:
                # Evaluation needs only text; avoid concatenating audio buffers.
                messages.append(
                    group[0].model_copy(
                        update={
                            "content": "".join(chunk.content or "" for chunk in group)
                        }
                    )
                )
                group.clear()
                utterance_ids.clear()

        for tick in ticks:
            chunk = tick.agent_chunk
            if (
                tick.agent_tool_calls
                or tick.agent_tool_results
                or (chunk and chunk.is_tool_call())
            ):
                flush()
                silence = Decimal(0)
                # Speech and tool events may share a full-duplex tick. Preserve
                # that speech as a separate message without joining across tools.
                if chunk and chunk.content:
                    messages.append(chunk.model_copy(update={"tool_calls": None}))
                continue
            if chunk is None or not chunk.content:
                if group and not utterance_ids:
                    silence += Decimal(str(tick.tick_duration_seconds or 0.2))
                    if silence >= 1.0:
                        flush()
                continue
            ids = set(chunk.utterance_ids or [])
            if group and (utterance_ids or ids) and not utterance_ids.intersection(ids):
                flush()
            # Providers without utterance IDs emit incremental text fragments.
            # Preserve their spacing and merge until a pause/tool/ID boundary.
            group.append(chunk)
            utterance_ids.update(ids)
            silence = Decimal(0)
        flush()

        return messages

    @classmethod
    def calculate_reward(
        cls,
        task: Task,
        full_trajectory: list[Tick],
    ) -> RewardInfo:
        """
        Calculate the reward based on whether the agent communicated the required information.
        """
        if task.evaluation_criteria is None:
            return RewardInfo(
                reward=1.0,
                info={"notes": "No evaluation criteria"},
                reward_breakdown={RewardType.COMMUNICATE: 1.0},
            )
        communicate_info = task.evaluation_criteria.communicate_info
        if not communicate_info:
            return RewardInfo(
                reward=1.0,
                info={"note": "No communicate_info to evaluate"},
                reward_breakdown={RewardType.COMMUNICATE: 1.0},
            )

        # Convert ticks to merged agent messages
        agent_messages = cls.ticks_to_message_history(full_trajectory)

        communicate_info_checks = cls.evaluate_communicate_info(
            agent_messages, communicate_info
        )

        # Calculate reward: 1 if all expectations are met, 0 otherwise
        all_expectations_met = all(result.met for result in communicate_info_checks)
        reward = 1.0 if all_expectations_met else 0.0

        return RewardInfo(
            reward=reward,
            communicate_checks=communicate_info_checks,
            reward_breakdown={RewardType.COMMUNICATE: reward},
        )

    @classmethod
    def evaluate_communicate_info(
        cls,
        agent_messages: list[AssistantMessage],
        communicate_info: list[str],
    ) -> list[CommunicateCheck]:
        """
        Evaluate whether the agent communicates the information correctly.
        """
        return CommunicateEvaluator.evaluate_communicate_info(
            full_trajectory=agent_messages,
            communicate_info=communicate_info,
        )
