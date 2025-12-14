"""
Exchange Delivered Order Items Command for FastWorkflow.

This command wraps tau2-bench's exchange_delivered_order_items tool.
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
    """Exchange items from a delivered order for different products."""
    
    class Input(BaseModel):
        order_id: str = Field(
            default="NOT_FOUND",
            description="ID of the delivered order",
            examples=["ORD123", "ORD456"],
        )
        
        old_product_id: str = Field(
            default="NOT_FOUND",
            description="Product ID to exchange from",
            examples=["PROD123", "PROD456"],
        )
        
        new_product_id: str = Field(
            default="NOT_FOUND",
            description="Product ID to exchange to",
            examples=["PROD789", "PROD012"],
        )
        
        quantity: int = Field(
            default=1,
            description="Quantity to exchange",
            examples=[1, 2, 3],
        )

        model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    class Output(BaseModel):
        result: str = Field(
            description="Confirmation message or error"
        )

    plain_utterances: List[str] = [
        "Exchange product PROD123 for PROD789 in order ORD123",
        "I want to exchange an item from my delivered order",
        "Swap product PROD456 with PROD012 from order ORD456",
        "Exchange 2 units of PROD123 for PROD789",
        "Process exchange for order ORD123",
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
        """Process exchange using tau2-bench tool."""
        try:
            db = RetailDB.load(RETAIL_DB_PATH)
            tools = RetailTools(db)
            
            result = tools.exchange_delivered_order_items(
                order_id=input.order_id,
                old_product_id=input.old_product_id,
                new_product_id=input.new_product_id,
                quantity=input.quantity
            )
            
            return Signature.Output(result=result)
            
        except ValueError as e:
            return Signature.Output(result=f"Error: {str(e)}")
        except Exception as e:
            return Signature.Output(result=f"Unexpected error: {str(e)}")
