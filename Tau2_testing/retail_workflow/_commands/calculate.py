from typing import List

import fastworkflow
from pydantic import BaseModel, Field, ConfigDict
from fastworkflow.workflow import Workflow
from fastworkflow import CommandOutput, CommandResponse

# Tau2-bench imports
from tau2.domains.retail.tools import RetailTools
from tau2.domains.retail.data_model import RetailDB
from tau2.domains.retail.utils import RETAIL_DB_PATH


class Signature:
    """Calculate mathematical expressions"""
    class Input(BaseModel):
        expression: str = Field(
            default="NOT_FOUND",
            description="Mathematical expression to calculate (e.g., '2 + 2', '10 * 5 - 3')",
            examples=["2 + 2", "10 * 5", "(100 - 20) / 2"],
        )

        model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    class Output(BaseModel):
        result: str = Field(
            description="The calculated result of the mathematical expression"
        )

    plain_utterances: List[str] = [
        "What is 2 plus 2?",
        "Calculate 10 times 5 for me.",
        "Can you compute 100 minus 25?",
        "What's 50 divided by 2?",
        "Help me calculate (100 - 20) * 3.",
    ]

    template_utterances: List[str] = []

    @staticmethod
    def generate_utterances(workflow: fastworkflow.Workflow, command_name: str) -> List[str]:
        utterance_definition = fastworkflow.RoutingRegistry.get_definition(workflow.folderpath)
        utterances_obj = utterance_definition.get_command_utterances(command_name)

        import os
        command_name = os.path.splitext(os.path.basename(__file__))[0]
        from fastworkflow.train.generate_synthetic import generate_diverse_utterances
        return generate_diverse_utterances(
            utterances_obj.plain_utterances, command_name
        )


class ResponseGenerator:
    def __call__(
        self,
        workflow: Workflow,
        command: str,
        command_parameters: Signature.Input
    ) -> CommandOutput:
        output = self._process_command(workflow, command_parameters)
        return CommandOutput(
            workflow_id=workflow.id,
            command_responses=[
                CommandResponse(response=f"The result is: {output.result}")
            ]
        )

    def _process_command(self,
        workflow: Workflow, input: Signature.Input
    ) -> Signature.Output:
        """
        Process the calculate command using tau2-bench tools.
        """
        try:
            # Load database and create tools instance
            db = RetailDB.load(RETAIL_DB_PATH)
            tools = RetailTools(db)
            
            # Call the tau2-bench tool method
            result = tools.calculate(expression=input.expression)
            
            return Signature.Output(result=result)
            
        except ValueError as e:
            return Signature.Output(result=f"Error: {str(e)}")
        except Exception as e:
            return Signature.Output(result=f"Unexpected error: {str(e)}")
