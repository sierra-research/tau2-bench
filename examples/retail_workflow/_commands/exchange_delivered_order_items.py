from typing import List

import fastworkflow
from pydantic import BaseModel, Field, ConfigDict
from fastworkflow.workflow import Workflow
from fastworkflow import CommandOutput, CommandResponse

from tau2.domains.retail.tools import RetailTools
from tau2.domains.retail.data_model import RetailDB
from tau2.domains.retail.utils import RETAIL_DB_PATH


class Signature:
    """Exchange delivered order items"""
    class Input(BaseModel):
        order_id: str = Field(
            default="NOT_FOUND",
            description="The order ID to exchange (must start with #)",
            pattern=r"^(#[\w\d]+|NOT_FOUND)$",
            examples=["#W0000000"],
            json_schema_extra={
                "available_from": ["get_user_details"]
            }
        )
        item_ids: List[str] = Field(
            default_factory=list,
            description="The item IDs to be exchanged",
            examples=["1008292230"],
            json_schema_extra={
                "available_from": ["get_order_details"]
            }
        )
        new_item_ids: List[str] = Field(
            default_factory=list,
            description="The new item IDs to exchange for",
            examples=["1008292230"],
            json_schema_extra={
                "available_from": ["get_product_details"]
            }
        )
        payment_method_id: str = Field(
            default="NOT_FOUND",
            description="Payment method ID for price difference.",
            examples=["gift_card_0000000", "credit_card_0000000"],
            json_schema_extra={
                "available_from": ["get_order_details", "get_user_details"]
            }
        )

        model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    class Output(BaseModel):
        status: str = Field(description="Whether exchange succeeded.")

    plain_utterances: List[str] = [
        "I received the wrong size in my order. Can I get it exchanged for a different one?",
        "The item I got doesn't fit well. I'd like to swap it with another of the same kind.",
        "Can you help me replace one of the items I received with a different variant?",
        "I'd like to exchange some things from my last delivery — they weren't quite right.",
        "One of the products I got isn't what I expected. Can I trade it in for a different one?",
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
        response = f"Response: {output.status}"
        return CommandOutput(
            workflow_id=workflow.id,
            command_responses=[
                CommandResponse(response=response)
            ]
        )

    def _process_command(self,
        workflow: Workflow, input: Signature.Input
    ) -> Signature.Output:
        """Process exchange using tau2-bench tool."""
        try:
            db = RetailDB.load(RETAIL_DB_PATH)
            tools = RetailTools(db)
            
            # Ensure order_id has # prefix
            order_id = input.order_id if input.order_id.startswith('#') else f'#{input.order_id}'
            
            result = tools.exchange_delivered_order_items(
                order_id=order_id,
                item_ids=input.item_ids,
                new_item_ids=input.new_item_ids,
                payment_method_id=input.payment_method_id
            )
            
            return Signature.Output(status=str(result))
            
        except ValueError as e:
            return Signature.Output(status=f"Error: {str(e)}")
        except Exception as e:
            return Signature.Output(status=f"Unexpected error: {str(e)}")
