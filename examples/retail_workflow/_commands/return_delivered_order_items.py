from typing import List

import fastworkflow
from pydantic import BaseModel, Field, ConfigDict
from fastworkflow.workflow import Workflow
from fastworkflow import CommandOutput, CommandResponse

from tau2.domains.retail.tools import RetailTools
from tau2.domains.retail.data_model import RetailDB
from tau2.domains.retail.utils import RETAIL_DB_PATH


class Signature:
    """Return delivered order items"""
    class Input(BaseModel):
        order_id: str = Field(
            default="NOT_FOUND",
            description="The order ID for return (must start with #)",
            pattern=r"^(#W\d+|NOT_FOUND)$",
            examples=["#W3456890"],
            json_schema_extra={
                "available_from": ["get_user_details"]
            }
        )
        item_ids: List[str] = Field(
            default_factory=list,
            description="List of item IDs to be returned",
            examples=["1008292230"],
            json_schema_extra={
                "available_from": ["get_order_details"]
            }
        )
        payment_method_id: str = Field(
            default="NOT_FOUND",
            description="Payment method ID for refund",
            pattern=r"^((gift_card|credit_card|paypal)_\d+|NOT_FOUND)$",
            examples=["gift_card_0000000", "credit_card_0000000", "paypal_0000000"],
            json_schema_extra={
                "available_from": ["get_order_details"]
            }
        )

        model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    class Output(BaseModel):
        status: str = Field(description="Whether return succeeded.")

    plain_utterances: List[str] = [
        "I want to return some items from my order #W0001234.",
        "Please help me return item 1008292230 and 1008292231 from order #W0005678.",
        "Can I return item 1008292230 from my delivered order?",
        "I need to return these products from my last order #W0009876.",
        "Return item 1008292230 from order #W0001357 and refund to my gift card.",
        "I want to request a return for multiple items from order #W0004567.",
        "How can I return items 1008292230 and 1008292240 from order #W0002468?",
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
        response = f"Response: return status is: {output.status}"
        return CommandOutput(
            workflow_id=workflow.id,
            command_responses=[
                CommandResponse(response=response)
            ]
        )

    def _process_command(self,
        workflow: Workflow, input: Signature.Input
    ) -> Signature.Output:
        """Process return using tau2-bench tool."""
        try:
            db = RetailDB.load(RETAIL_DB_PATH)
            tools = RetailTools(db)
            
            # Ensure order_id has # prefix
            order_id = input.order_id if input.order_id.startswith('#') else f'#{input.order_id}'
            
            result = tools.return_delivered_order_items(
                order_id=order_id,
                item_ids=input.item_ids,
                payment_method_id=input.payment_method_id
            )
            
            return Signature.Output(status=str(result))
            
        except ValueError as e:
            return Signature.Output(status=f"Error: {str(e)}")
        except Exception as e:
            return Signature.Output(status=f"Unexpected error: {str(e)}")
