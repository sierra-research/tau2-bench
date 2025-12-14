"""
Transfer to Human Agents Command for FastWorkflow.

This command wraps tau2-bench's transfer_to_human_agents tool.
"""

from typing import List

import fastworkflow
from pydantic import BaseModel, Field, ConfigDict
from fastworkflow.workflow import Workflow
from fastworkflow import CommandOutput, CommandResponse

from tau2.domains.retail.tools import RetailTools
from tau2.domains.retail.data_model import RetailDB
from tau2.domains.retail.utils import RETAIL_DB_PATH


class Signature:
    """Transfer the conversation to a human agent."""
    
    class Input(BaseModel):
        # No parameters needed for this tool
        trigger: str = Field(
            default="transfer",
            description="Trigger to transfer to human agent",
        )

        model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    class Output(BaseModel):
        result: str = Field(
            description="Confirmation message that transfer is initiated"
        )

    plain_utterances: List[str] = [
        "Transfer me to a human agent",
        "I want to speak with a real person",
        "Connect me to customer service",
        "Can I talk to a human?",
        "I need help from a human agent",
        "Transfer to human support",
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
                CommandResponse(response=output.result)
            ]
        )

    def _process_command(self,
        workflow: Workflow, input: Signature.Input
    ) -> Signature.Output:
        """Transfer to human agent using tau2-bench tool."""
        try:
            db = RetailDB.load(RETAIL_DB_PATH)
            tools = RetailTools(db)
            
            result = tools.transfer_to_human_agents()
            
            return Signature.Output(result=result)
            
        except Exception as e:
            return Signature.Output(result=f"Error transferring to human agent: {str(e)}")
