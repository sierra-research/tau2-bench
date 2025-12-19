from typing import List

import fastworkflow
from pydantic import BaseModel, Field, ConfigDict
from fastworkflow.workflow import Workflow
from fastworkflow import CommandOutput, CommandResponse

from tau2.domains.telecom.tools import TelecomTools
from tau2.domains.telecom.data_model import TelecomDB
from tau2.domains.telecom.utils import TELECOM_DB_PATH


class Signature:
    """Transfer to a human agent as the last resort"""
    class Input(BaseModel):
        summary: str = Field(
            default="NOT_FOUND",
            description="A summary of the user's issue",
            examples=["Customer needs help with complex return process"],
        )

        model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    class Output(BaseModel):
        status: str = Field(description="Whether transfer succeeded.")

    plain_utterances: List[str] = [
        "This isn't working, I need to speak with a real person",
        "Can I talk to a human agent please?",
        "I want to speak with someone from customer service",
        "This is too complicated, connect me to a live agent",
        "I need human help, this bot can't solve my issue",
        "Transfer me to a customer service representative",
        "I'd like to speak with a support agent directly",
        "Can you get me a real person? This isn't helping",
        "I need to talk to someone who can actually help me",
        "Please escalate this to a human agent",
        "Connect me with customer support please",
        "I want to speak to your supervisor",
        "This is frustrating, I need a real person to help",
        "Transfer me to someone who can handle this properly",
        "I prefer speaking with a human agent about this",
    ]
    template_utterances: List[str] = []

    @staticmethod
    def generate_utterances(workflow: fastworkflow.Workflow, command_name: str) -> List[str]:
        utterance_definition = fastworkflow.RoutingRegistry.get_definition(workflow.folderpath)
        utterances_obj = utterance_definition.get_command_utterances(command_name)
        from fastworkflow.train.generate_synthetic import generate_diverse_utterances
        return generate_diverse_utterances(utterances_obj.plain_utterances, command_name)


class ResponseGenerator:
    def __call__(
        self,
        workflow: Workflow,
        command: str,
        command_parameters: Signature.Input
    ) -> CommandOutput:
        output = self._process_command(workflow, command_parameters)
        response = f"Transfer status: {output.status}"
        return CommandOutput(
            workflow_id=workflow.id,
            command_responses=[
                CommandResponse(response=response)
            ]
        )

    def _process_command(self,
        workflow: Workflow, input: Signature.Input
    ) -> Signature.Output:
        """Transfer to human agent using tau2-bench tool."""
        try:
            db = TelecomDB.load(TELECOM_DB_PATH)
            tools = TelecomTools(db)
            
            # Use the summary from input
            result = tools.transfer_to_human_agents(summary=input.summary)
            
            return Signature.Output(status=result)
            
        except Exception as e:
            return Signature.Output(status=f"Error transferring to human agent: {str(e)}")
