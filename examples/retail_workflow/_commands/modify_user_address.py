from typing import List

import fastworkflow
from pydantic import BaseModel, Field, ConfigDict
from fastworkflow.workflow import Workflow
from fastworkflow import CommandOutput, CommandResponse

from tau2.domains.retail.tools import RetailTools
from tau2.domains.retail.data_model import RetailDB
from tau2.domains.retail.utils import RETAIL_DB_PATH


class Signature:
    """Modify user address"""
    class Input(BaseModel):
        user_id: str = Field(
            default="NOT_FOUND",
            description="The user ID to modify",
            pattern=r"^([a-z]+_[a-z]+_\d+|NOT_FOUND)$",
            examples=["sara_doe_496"],
            json_schema_extra={
                "available_from": ["find_user_id_by_email", "find_user_id_by_name_zip"]
            }
        )
        address1: str = Field(default="NOT_FOUND", description="First line of address", examples=["123 Main St"])
        address2: str = Field(default="NOT_FOUND", description="Second line of address", examples=["Apt 1"])
        city: str = Field(default="NOT_FOUND", description="City name", examples=["San Francisco"])
        state: str = Field(
            default="NOT_FOUND",
            description="State code",
            pattern=r"^([A-Z]{2}|NOT_FOUND)$",
            examples=["CA"],
        )
        country: str = Field(default="NOT_FOUND", description="Country name", examples=["USA"])
        zip: str = Field(
            default="NOT_FOUND",
            description="ZIP code",
            pattern=r"^(\d{5}|NOT_FOUND)$",
            examples=["12345"],
        )

        model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    class Output(BaseModel):
        status: str = Field(description="Whether address modification succeeded.")

    plain_utterances: List[str] = [
        "I need to update my shipping address.",
        "Please change my default address to a new one.",
        "Can you modify my address to 123 Main St, Apt 1, San Francisco, CA, USA, 94105?",
        "I want to update my profile address to a new location.",
        "Change my address to 456 Elm Street, Suite 300, New York, NY, USA, 10001.",
        "I recently moved and need to change my delivery address.",
        "Update my account with my new address in Los Angeles.",
        "I̵'d like to edit my saved address details.",
        "Can you update my contact info with a new address?",
        "Switch my default shipping address to 789 Oak Ave, Dallas, TX, USA, 75201.",
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
        response = f"Response: Modified details: {output.status}"
        return CommandOutput(
            workflow_id=workflow.id,
            command_responses=[
                CommandResponse(response=response)
            ]
        )

    def _process_command(self,
        workflow: Workflow, input: Signature.Input
    ) -> Signature.Output:
        """Modify user address using tau2-bench tool."""
        try:
            db = RetailDB.load(RETAIL_DB_PATH)
            tools = RetailTools(db)
            
            result = tools.modify_user_address(
                user_id=input.user_id,
                address1=input.address1,
                address2=input.address2,
                city=input.city,
                state=input.state,
                country=input.country,
                zip=input.zip
            )
            
            return Signature.Output(status=result)
            
        except ValueError as e:
            return Signature.Output(status=f"Error: {str(e)}")
        except Exception as e:
            return Signature.Output(status=f"Unexpected error: {str(e)}")
