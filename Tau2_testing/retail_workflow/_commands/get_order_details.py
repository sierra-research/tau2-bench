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
    """Get order details"""
    class Input(BaseModel):
        order_id: str = Field(
            default="NOT_FOUND",
            description=(
                "The order ID to get details for (must start with #)"
            ),
            pattern=r"^(#[\w\d]+|NOT_FOUND)$",
            examples=["#W0000000"],
            json_schema_extra={
                "available_from": ["get_user_details"]
            }
        )

        model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    class Output(BaseModel):
        order_details: str = Field(
            description=(
                "Detailed information about the order such as "
                "shipping address, "
                "items ordered with details including name, product and item id, price and options, "
                "fulfillments with details including tracking id and item ids, "
                "order status, and"
                "payment history with details including transaction type, amount and payment method id"
            ),
            json_schema_extra={
                "used_by": [
                    "exchange_delivered_order_items",
                    "get_product_details",
                    "modify_pending_order_items",
                    "return_delivered_order_items",
                ]
            },
        )

    plain_utterances: List[str] = [
        "Can you look up order #W0000000 for me?",
        "I need details on my order number #W1234567.",
        "What's the status of order #W9876543?",
        "Tell me more about my order #W0000001.",
        "Show me all the information for order #W5555555.",
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
                CommandResponse(response=output.order_details)
            ]
        )

    def _process_command(self,
        workflow: Workflow, input: Signature.Input
    ) -> Signature.Output:
        """
        Process the get_order_details command using tau2-bench tools.
        """
        try:
            # Load database and create tools instance
            db = RetailDB.load(RETAIL_DB_PATH)
            tools = RetailTools(db)
            
            # Call the tau2-bench tool method
            order = tools.get_order_details(order_id=input.order_id)
            
            # Convert Pydantic model to JSON string
            import json
            order_json = order.model_dump_json(indent=2)
            
            return Signature.Output(
                order_details=order_json
            )
            
        except ValueError as e:
            return Signature.Output(
                order_details=f"Error: {str(e)}"
            )
        except Exception as e:
            return Signature.Output(
                order_details=f"Unexpected error: {str(e)}"
            )
