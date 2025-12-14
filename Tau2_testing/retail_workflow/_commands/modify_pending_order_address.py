"""
Modify Pending Order Address Command for FastWorkflow.

This command wraps tau2-bench's modify_pending_order_address tool.
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
    """Modify the shipping address of a pending order."""
    
    class Input(BaseModel):
        order_id: str = Field(
            default="NOT_FOUND",
            description="ID of the pending order",
            examples=["ORD123", "ORD456"],
        )
        
        street: str = Field(
            default="NOT_FOUND",
            description="New street address",
            examples=["123 Main St", "456 Oak Ave"],
        )
        
        city: str = Field(
            default="NOT_FOUND",
            description="New city",
            examples=["New York", "Los Angeles"],
        )
        
        state: str = Field(
            default="NOT_FOUND",
            description="New state (2-letter code)",
            examples=["NY", "CA"],
        )
        
        zip_code: str = Field(
            default="NOT_FOUND",
            description="New zip code",
            examples=["10001", "90001"],
        )

        model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    class Output(BaseModel):
        result: str = Field(
            description="Confirmation message or error"
        )

    plain_utterances: List[str] = [
        "Change the shipping address for order ORD123",
        "Update delivery address to 123 Main St, New York, NY 10001",
        "Modify the address on my pending order",
        "Change shipping address for order ORD456 to 456 Oak Ave, Los Angeles, CA 90001",
        "Update the delivery address on order ORD123",
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
        """Modify pending order address using tau2-bench tool."""
        try:
            db = RetailDB.load(RETAIL_DB_PATH)
            tools = RetailTools(db)
            
            result = tools.modify_pending_order_address(
                order_id=input.order_id,
                street=input.street,
                city=input.city,
                state=input.state,
                zip_code=input.zip_code
            )
            
            return Signature.Output(result=result)
            
        except ValueError as e:
            return Signature.Output(result=f"Error: {str(e)}")
        except Exception as e:
            return Signature.Output(result=f"Unexpected error: {str(e)}")
