from typing import List, Dict, Any, Optional

import fastworkflow
from pydantic import BaseModel, Field, ConfigDict
from fastworkflow.workflow import Workflow
from fastworkflow import CommandOutput, CommandResponse

from tau2.domains.airline.tools import AirlineTools
from tau2.domains.airline.data_model import FlightDB
from tau2.domains.airline.utils import AIRLINE_DB_PATH

class Signature:
    """Transfer to a human agent ONLY if the user demands it explicitly."""
    class Input(BaseModel):
        summary: str = Field(
            default="NOT_FOUND",
            description="A summary of the user's issue",
            examples=["Customer needs assistance with flight reservation modification that requires human agent approval"],
        )

        model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    class Output(BaseModel):
        status: str = Field(description="Whether transfer succeeded.")

    plain_utterances: List[str] = [
        "This is really frustrating — I've already tried everything and nothing's working.",
        "Can I please talk to someone who can actually help me with this?",
        "I've explained my issue multiple times now, and I just want to speak to a real person.",
        "This feels too complicated for a bot. Is there anyone I can call or chat with directly?",
        "Ugh, I've been going in circles. Just connect me to customer support.",
        "Look, I don't think this automated system understands what I'm trying to say.",
        "Your system isn't solving my issue. I need to talk to someone immediately.",
        "This is beyond frustrating — I need a human to step in and resolve this.",
        "Nothing you're suggesting is helping. Is there someone else I can speak with?",
        "I'm done with the chatbot, get me a real person now.",
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
        response = (
            f'Response: transfer status: {output.status}'
        )
        return CommandOutput(
            workflow_id=workflow.id,
            command_responses=[CommandResponse(response=response)],
        )
    
    def _process_command(self, workflow: Workflow, input: Signature.Input) -> Signature.Output:
        """
        Process the transfer_to_human_agents command using tau2-bench airline tools.
        """
        try:
            db = FlightDB.load(AIRLINE_DB_PATH)
            tools = AirlineTools(db)
            result = tools.transfer_to_human_agents(summary=input.summary)
            return Signature.Output(status=result)
        except Exception as e:
            return Signature.Output(status=f"Unexpected error: {str(e)}")