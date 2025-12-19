from typing import List, Dict, Any, Optional

import fastworkflow
from pydantic import BaseModel, Field, ConfigDict
from fastworkflow.workflow import Workflow
from fastworkflow import CommandOutput, CommandResponse

from tau2.domains.airline.tools import AirlineTools
from tau2.domains.airline.data_model import FlightDB
from tau2.domains.airline.utils import AIRLINE_DB_PATH

class Signature:
    """Calculate a mathematical expression"""

    class Input(BaseModel):
        expression: str = Field(
            default="NOT_FOUND",
            description=(
                "The mathematical expression to calculate. Allowed characters: digits, "
                "+, -, *, /, parentheses, spaces."
            ),
            pattern=r"^(NOT_FOUND|[0-9+\-*/().\s]+)$",
            examples=["(2 + 3) * 4", "10 / 2 + 3", "(8-3)/5"],
        )

        model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    class Output(BaseModel):
        result: str = Field(
            description=(
                "Calculated result as a string. Returns an error message for invalid input."
            )
        )

    plain_utterances: List[str] = [
        "What is 2 + 2?",
        "Calculate (2 + 3) * 4",
        "Can you compute 10 / 2 + 3?",
        "Evaluate (8-3)/5",
        "Please calculate 7 * (6 - 2)",
        "What's the result of this mathematical expression?",
        "Can you do some math for me?",
        "Help me with this calculation",
        "Solve this math problem",
        "Calculate this equation"
    ]

    template_utterances: List[str] = []

    @staticmethod
    def generate_utterances(workflow: fastworkflow.Workflow, command_name: str) -> List[str]:
        utterance_definition = fastworkflow.RoutingRegistry.get_definition(workflow.folderpath)
        utterances_obj = utterance_definition.get_command_utterances(command_name)
        from fastworkflow.train.generate_synthetic import generate_diverse_utterances
        return generate_diverse_utterances(utterances_obj.plain_utterances, command_name)
    
class ResponseGenerator:
    def __call__(self, workflow: Workflow, command: str, command_parameters: Signature.Input) -> CommandOutput:
        output = self._process_command(workflow, command_parameters)
        response = f"Result: {output.result}"
        return CommandOutput(
            workflow_id=workflow.id,
            command_responses=[CommandResponse(response=response)],
        )
    
    def _process_command(self, workflow: Workflow, input: Signature.Input) -> Signature.Output:
        """
        Process the calculate command using tau2-bench airline tools.
        """
        try:
            db = FlightDB.load(AIRLINE_DB_PATH)
            tools = AirlineTools(db)
            result = tools.calculate(expression=input.expression)
            return Signature.Output(result=result)
        except ValueError as e:
            return Signature.Output(result=f"Error: {str(e)}")
        except Exception as e:
            return Signature.Output(result=f"Unexpected error: {str(e)}")