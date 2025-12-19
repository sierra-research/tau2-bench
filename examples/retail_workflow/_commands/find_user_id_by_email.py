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
    """Find user id by email"""
    class Input(BaseModel):
        """Parameters taken from user utterance."""

        email: str = Field(
            default="NOT_FOUND",
            description=(
                "The email address to search for. If email is not available, "
                "use `find_user_id_by_name_zip` instead."
            ),
            pattern=r"^(NOT_FOUND|[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})$",
            examples=["user@example.com"],
        )

        model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    class Output(BaseModel):
        user_id: str = Field(
            description="User identifier returned by lookup.",
            json_schema_extra={
                "used_by": ["get_user_details"]
            }
        )

    plain_utterances: List[str] = [
        "Can you tell me the ID linked to john.doe@example.com?",
        "I only have the email address — how do I find the user ID?",
        "What's the account ID for someone with the email jane_smith@domain.com?",
        "I need the unique user ID associated with this email: support@acme.io",
        "Do we have an ID on file for mark.brown@company.org?",
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
        return CommandOutput(
            workflow_id=workflow.id,
            command_responses=[
                CommandResponse(response=f"User ID: {output.user_id}")
            ]
        )

    def _process_command(self,
        workflow: Workflow, input: Signature.Input
    ) -> Signature.Output:
        """
        Process the find_user_id_by_email command using tau2-bench tools.
        """
        try:
            db = RetailDB.load(RETAIL_DB_PATH)
            tools = RetailTools(db)
            
            user_id = tools.find_user_id_by_email(email=input.email)
            
            return Signature.Output(user_id=user_id)
            
        except ValueError as e:
            return Signature.Output(user_id=f"Error: {str(e)}")
        except Exception as e:
            return Signature.Output(user_id=f"Unexpected error: {str(e)}")
