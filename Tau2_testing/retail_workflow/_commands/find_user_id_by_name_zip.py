"""
Find User ID by Name and Zip Command for FastWorkflow.

This command wraps tau2-bench's find_user_id_by_name_zip tool.
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
    """Find a user's ID by their first name, last name, and zip code."""
    
    class Input(BaseModel):
        first_name: str = Field(
            default="NOT_FOUND",
            description="User's first name",
            examples=["John", "Jane"],
        )
        
        last_name: str = Field(
            default="NOT_FOUND",
            description="User's last name",
            examples=["Doe", "Smith"],
        )
        
        zip_code: str = Field(
            default="NOT_FOUND",
            description="User's zip code",
            examples=["12345", "90210"],
        )

        model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    class Output(BaseModel):
        user_id: str = Field(
            description="The user's ID if found, or error message",
            json_schema_extra={
                "used_by": ["get_user_details", "modify_user_address"]
            }
        )

    plain_utterances: List[str] = [
        "Find user ID for John Doe in zip code 12345",
        "Look up user ID by name Jane Smith and zip 90210",
        "Search for user ID using name and zip code",
        "Get user ID for first name John, last name Doe, zip 12345",
        "Find the user ID for someone named Jane Smith in zip code 90210",
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
                CommandResponse(response=f"User ID: {output.user_id}")
            ]
        )

    def _process_command(self,
        workflow: Workflow, input: Signature.Input
    ) -> Signature.Output:
        """Find user ID using tau2-bench's find_user_id_by_name_zip tool."""
        try:
            db = RetailDB.load(RETAIL_DB_PATH)
            tools = RetailTools(db)
            
            user_id = tools.find_user_id_by_name_zip(
                first_name=input.first_name,
                last_name=input.last_name,
                zip_code=input.zip_code
            )
            
            return Signature.Output(user_id=user_id)
            
        except ValueError as e:
            return Signature.Output(user_id=f"Error: {str(e)}")
        except Exception as e:
            return Signature.Output(user_id=f"Unexpected error: {str(e)}")
