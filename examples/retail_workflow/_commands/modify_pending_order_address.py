from typing import List

import fastworkflow
from pydantic import BaseModel, Field, ConfigDict
from fastworkflow.workflow import Workflow
from fastworkflow import CommandOutput, CommandResponse

from tau2.domains.retail.tools import RetailTools
from tau2.domains.retail.data_model import RetailDB
from tau2.domains.retail.utils import RETAIL_DB_PATH


class Signature:
    """Modify pending order address"""
    class Input(BaseModel):
        order_id: str = Field(
            default="NOT_FOUND",
            description="The order ID to modify (must start with #)",
            pattern=r"^(#W\d+|NOT_FOUND)$",
            examples=["#W0000000"],
            json_schema_extra={
                "available_from": ["get_user_details"]
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
        status: str = Field(description="Whether modification succeeded.")

    plain_utterances: List[str] = [
        "Can I update the shipping address for my order?",
        "I need to change the delivery address on an order I just placed.",
        "Please modify the address for my pending order.",
        "I made a mistake in the shipping details — can I fix it?",
        "I'd like to change where my order is being delivered.",
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
        """Modify pending order address using tau2-bench tool."""
        try:
            db = RetailDB.load(RETAIL_DB_PATH)
            tools = RetailTools(db)
            
            # Ensure order_id has # prefix
            order_id = input.order_id if input.order_id.startswith('#') else f'#{input.order_id}'
            
            result = tools.modify_pending_order_address(
                order_id=order_id,
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
