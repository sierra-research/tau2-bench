import json
import textwrap
from copy import deepcopy
from typing import Callable, Optional, Protocol, cast

from tau2.data_model.message import ToolCall
from tau2.data_model.tasks import EnvAssertion, EnvFunctionCall, Task
from tau2.domains.healthcare.environment import HealthcareEnvironment, get_environment
from tau2.utils import DATA_DIR

from .const import PERSONAS
from .utils import BaseTask, ComposedTask, SelectionSet, compose_tasks


class GetEnvAssertionsCallable(Protocol):
    """Protocol for get_env_assertions function.

    Returns tuple of (env_assertions, nl_assertions, communicate_info).
    """

    def __call__(
        self, expected_success: bool
    ) -> tuple[list[EnvAssertion], list[str], list[str]]: ...


def prepare_base_task(base_task: dict, env: HealthcareEnvironment) -> dict:
    """Prepare task with patient-specific information."""
    base_task = deepcopy(base_task)
    patient_name = env.user_tools.surroundings.full_name
    date_of_birth = env.user_tools.surroundings.date_of_birth
    location = env.user_tools.surroundings.location

    user_info = {
        "name": patient_name,
        "date_of_birth": date_of_birth,
        "location": location,
    }

    known_info_template = base_task["user_scenario"]["instructions"]["known_info"]
    known_info = known_info_template.format(**user_info)
    base_task["user_scenario"]["instructions"]["known_info"] = known_info

    ticket_template = base_task["ticket"]
    ticket = ticket_template.format(**user_info)
    base_task["ticket"] = ticket

    return base_task


class TaskManager:
    def __init__(
        self,
        name: str,
        purpose: str,
        task_instructions: str,
        reason_for_call: str,
        known_info: str,
        ticket: str,
        selection_sets: list[SelectionSet],
        get_env_assertions: GetEnvAssertionsCallable,
        set_surrounding: Callable[[HealthcareEnvironment], list[EnvFunctionCall]],
        is_fixed: Callable[[HealthcareEnvironment], bool],
        task_validator: Optional[Callable[[list[Optional[BaseTask]]], bool]] = None,
        domain: str = "healthcare",
    ):
        self.domain = domain
        self.name = name
        self.base_task_template = {
            "id": f"[{name}]",
            "description": {
                "purpose": purpose,
            },
            "user_scenario": {
                "instructions": {
                    "task_instructions": task_instructions,
                    "domain": domain,
                    "reason_for_call": reason_for_call,
                    "known_info": known_info,
                },
                "persona": None,
            },
            "ticket": ticket,
            "initial_state": {},
            "evaluation_criteria": {"env_assertions": None},
        }

        self.selection_sets = selection_sets
        self.get_env_assertions = get_env_assertions
        self.set_surrounding = set_surrounding
        self.is_fixed = is_fixed
        self.task_validator = task_validator

    def create_task(self, composed_task: ComposedTask, persona: str = "None") -> Task:
        env = get_environment()

        init_actions = self.set_surrounding(env)
        env.run_env_function_calls(init_actions)
        for func in composed_task.init_funcs:
            func_calls = func(env)
            env.run_env_function_calls(func_calls)
            init_actions.extend(
                [fc for fc in func_calls if not isinstance(fc, EnvAssertion)]
            )

        fix_tool_calls: list[ToolCall] = []
        expected_failure = False
        for func in composed_task.fix_funcs:
            if func is None:
                expected_failure = True
                break
            tool_calls = func(env)
            fix_tool_calls.extend(tool_calls)
            # Check if any fix function contains a transfer action
            if any(
                tc.name in {"transfer_to_nurse", "transfer_to_human_agent"}
                for tc in tool_calls
            ):
                expected_failure = True
                break

        deduplicated_tool_calls: list[ToolCall] = []
        seen_identity_verification = False

        for tc in fix_tool_calls:
            if tc.name == "get_patient_details" and tc.requestor == "assistant":
                if seen_identity_verification:
                    continue
                else:
                    seen_identity_verification = True

            deduplicated_tool_calls.append(tc)

        fix_tool_calls = deduplicated_tool_calls

        if expected_failure:
            fix_actions = [
                {
                    "action_id": "transfer_to_nurse",
                    "name": "transfer_to_nurse",
                    "requestor": "assistant",
                    "arguments": {},
                }
            ]
        else:
            fix_actions = []
            for i, tc in enumerate(fix_tool_calls):
                action = {
                    "action_id": f"{tc.name}_{i}",
                    "name": tc.name,
                    "requestor": tc.requestor,
                    "arguments": tc.arguments,
                }
                compare_args = getattr(tc, "compare_args", None)
                if compare_args is not None:
                    action["compare_args"] = compare_args
                fix_actions.append(action)

        env_assertions, nl_assertions, communicate_info = self.get_env_assertions(
            expected_success=not expected_failure
        )
        if not expected_failure:
            for func in composed_task.extra_env_assertions:
                extra_assertions = func(env)
                env_assertions.extend(extra_assertions)

        outcome_focused_intents = {
            "appointment_scheduling",
            "test_results_access",
            "chronic_monitoring",
            "telehealth_setup",
            "urgent_triage",
        }

        intent_name = self.name

        if expected_failure:
            reward_eval_mode = ["ACTION"]
        elif intent_name in outcome_focused_intents:
            # For outcome-focused tasks, use BOTH ENV_ASSERTION and ACTION
            # to enforce correct outcomes AND safe procedures
            if len(fix_actions) > 0 and len(env_assertions) > 0:
                reward_eval_mode = ["ENV_ASSERTION", "ACTION"]
            elif len(env_assertions) > 0:
                reward_eval_mode = ["ENV_ASSERTION"]
            else:
                reward_eval_mode = ["ACTION"]
        elif len(fix_actions) > 0 and len(env_assertions) > 0:
            reward_eval_mode = ["ACTION", "ENV_ASSERTION"]
        elif len(env_assertions) > 0:
            reward_eval_mode = ["ENV_ASSERTION"]
        else:
            reward_eval_mode = ["ACTION"]

        final_task = prepare_base_task(self.base_task_template, env)
        final_task["initial_state"]["initialization_actions"] = init_actions
        final_task["evaluation_criteria"]["actions"] = fix_actions
        final_task["evaluation_criteria"]["env_assertions"] = env_assertions
        final_task["evaluation_criteria"]["nl_assertions"] = (
            nl_assertions if nl_assertions else None
        )
        final_task["evaluation_criteria"]["communicate_info"] = (
            communicate_info if communicate_info else None
        )
        final_task["evaluation_criteria"]["reward_basis"] = reward_eval_mode
        final_task["user_scenario"]["persona"] = PERSONAS[persona]
        final_task["id"] += f"{composed_task.name}[PERSONA:{persona}]"
        final_task["description"]["info"] = composed_task.description
        task = Task(**final_task)
        return task

    def create_tasks(
        self,
        save_tasks: bool = False,
        custom_composed_tasks: Optional[list[ComposedTask]] = None,
    ) -> list[Task]:
        if custom_composed_tasks is not None:
            composed_tasks = custom_composed_tasks
        else:
            composed_tasks = compose_tasks(self.selection_sets, self.task_validator)
        composed_tasks = sorted(composed_tasks, key=lambda x: len(x.composed_from))
        print(f"Number of composed tasks: {len(composed_tasks)}")
        persona_options = list(PERSONAS.keys())
        personas = [
            persona_options[i % len(persona_options)]
            for i in range(len(composed_tasks))
        ]
        tasks = []
        for i, composed_task in enumerate(composed_tasks):
            print(f"Task {i + 1}")
            print(composed_task.name)
            task = self.create_task(composed_task, personas[i])
            print(task)
            print("-" * 100)
            self.verify_task(task)
            print("-" * 100)
            tasks.append(task)
        if save_tasks:
            file = (
                DATA_DIR / "tau2" / "domains" / self.domain / f"{self.name}_tasks.json"
            )
            with open(file, "w") as f:
                json.dump([t.model_dump() for t in tasks], f, indent=2)
        return tasks

    def run_assertions(
        self,
        env: HealthcareEnvironment,
        task: Task,
        verbose: bool = False,
        skip_behavioral: bool = False,
    ):
        if task.evaluation_criteria is None:
            return True
        assertions = task.evaluation_criteria.env_assertions or []
        if len(assertions) == 0:
            return True
        success = True
        for i, assertion in enumerate(assertions):
            # Skip behavioral assertions (tool call history checks) during task verification
            # These require full conversation trajectories and will be verified during evaluation
            if skip_behavioral and assertion.func_name in [
                "assert_tool_was_called",
                "assert_tool_was_not_called",
            ]:
                if verbose:
                    print(
                        f"Skipping behavioral assertion {i + 1} of {len(assertions)} (will be verified during evaluation)"
                    )
                    print(textwrap.indent(str(assertion), "  "))
                continue

            if verbose:
                print(f"Verifying env assertion {i + 1} of {len(assertions)}")
                print(textwrap.indent(str(assertion), "  "))
            assertion_success = env.run_env_assertion(
                assertion,
                raise_assertion_error=False,
            )
            if verbose:
                print("Success: ", assertion_success)
            success = success and assertion_success
        return success

    def _is_fixable(self, task: Task) -> bool:
        transfer_action_names = {"transfer_to_human_agent", "transfer_to_nurse"}
        if task.evaluation_criteria is None:
            return True
        action_names = {a.name for a in task.evaluation_criteria.actions or []}
        if action_names & transfer_action_names:
            return False
        return True

    def verify_task(self, task: Task):
        from tau2.registry import registry

        print("Verifying task: ", task.id)

        healthcare_env = cast(
            HealthcareEnvironment, registry.get_env_constructor("healthcare")()
        )
        assert self.is_fixed(healthcare_env), "Healthcare env starts in broken state"

        initialization_data = None
        initialization_actions = None
        if task.initial_state is not None:
            initialization_data = task.initial_state.initialization_data
            initialization_actions = task.initial_state.initialization_actions

        healthcare_env.set_state(
            initialization_data=initialization_data,
            initialization_actions=initialization_actions,
            message_history=[],
        )

        fix_actions = []
        if task.evaluation_criteria is not None:
            fix_actions = task.evaluation_criteria.actions or []

        fixable = self._is_fixable(task)
        for i, action in enumerate(fix_actions):
            assert not self.is_fixed(healthcare_env), (
                f"Task {task.id} is already fixed after {i} actions. {task}"
            )
            healthcare_env.make_tool_call(
                tool_name=action.name, requestor=action.requestor, **action.arguments
            )
            healthcare_env.sync_tools()
        if fixable:
            assert self.is_fixed(healthcare_env), (
                f"Task {task.id} is not fixed after all actions. {task}"
            )
        else:
            assert not self.is_fixed(healthcare_env), (
                f"Task {task.id} is fixed but should not be. {task}"
            )
        assert self.run_assertions(
            healthcare_env, task, verbose=True, skip_behavioral=True
        )
