"""
Modify Pending Order Items Command for FastWorkflow.

This command wraps tau2-bench's modify_pending_order_items tool.
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
    """Modify items in a pending order (add, remove, or change quantity)."""
    
    class Input(BaseModel):
        order_id: str = Field(
            default="NOT_FOUND",
            description="ID of the pending order",
            examples=["ORD123", "ORD456"],
        )
        
        product_id: str = Field(
            default="NOT_FOUND",
            description="Product ID to modify",
            examples=["PROD123", "PROD456"],
        )
        
        quantity: int = Field(
            default=0,
            description="New quantity (0 to remove item)",
            examples=[2, 5, 0],
        )

        model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    class Output(BaseModel):
        result: str = Field(
            description="Confirmation message or error"
        )

    plain_utterances: List[str] = [
        "Change quantity of product PROD123 to 2 in order ORD123",
        "Add 5 units of product PROD456 to order ORD456",
        "Remove product PROD123 from order ORD123",
        "Update item quantity in my pending order",
        "Modify order ORD123 to have 3 units of product PROD456",
        "Change the quantity of an item in my order",
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
        """Modify pending order items using tau2-bench tool."""
        try:
            db = RetailDB.load(RETAIL_DB_PATH)
            tools = RetailTools(db)
            
            result = tools.modify_pending_order_items(
                order_id=input.order_id,
                product_id=input.product_id,
                quantity=input.quantity
            )
            
            return Signature.Output(result=result)
            
        except ValueError as e:
            return Signature.Output(result=f"Error: {str(e)}")
        except Exception as e:
            return Signature.Output(result=f"Unexpected error: {str(e)}")
