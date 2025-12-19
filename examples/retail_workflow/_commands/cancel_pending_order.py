from pydantic import BaseModel
from fastworkflow.workflow import Workflow

import os
from typing import Annotated
from pydantic import Field, ConfigDict
import fastworkflow
from fastworkflow.train.generate_synthetic import generate_diverse_utterances
from fastworkflow import CommandOutput, CommandResponse

from tau2.domains.retail.tools import RetailTools
from tau2.domains.retail.data_model import RetailDB
from tau2.domains.retail.utils import RETAIL_DB_PATH


class Signature:
    """Cancel pending orders"""
    class Input(BaseModel):
        order_id: Annotated[
            str,
            Field(
                default="NOT_FOUND",
                description="The order ID to cancel (must start with #)",
                pattern=r"^(#[\w\d]+|NOT_FOUND)$",
                examples=["#123", "#abc123", "#order456"],
                json_schema_extra={
                    "available_from": ["get_user_details"]
                }
            )
        ]

        reason: Annotated[
            str,
            Field(
                default="NOT_FOUND",
                description="Reason for cancellation",
                json_schema_extra={
                    "enum": ["no longer needed", "ordered by mistake", "NOT_FOUND"]
                },
                examples=["no longer needed", "ordered by mistake"]
            )
        ]

        model_config = ConfigDict(
            arbitrary_types_allowed=True,
            validate_assignment=True
        )

    class Output(BaseModel):
        status: str = Field(
            description="whether cancellation succeeded)",
        )

    plain_utterances = [
        "I want to cancel my order because I no longer need it.",
        "Please cancel order #W1234567 — I ordered it by mistake.",
        "Can you cancel my pending order?",
        "I made a mistake and need to cancel an order I just placed.",
        "Cancel my order, I don't need it anymore.",
        "I accidentally placed an order — can you help me cancel it?",
        "Please stop processing order #W0000001, I no longer need the items.",
        "I'd like to cancel my order before it's shipped.",
        "I want to cancel a pending order — reason: ordered by mistake.",
        "Can I cancel my order? I changed my mind and don't need it."
    ]

    template_utterances = []

    @staticmethod
    def generate_utterances(workflow: fastworkflow.Workflow, command_name: str) -> list[str]:
        utterance_definition = fastworkflow.RoutingRegistry.get_definition(workflow.folderpath)
        utterances_obj = utterance_definition.get_command_utterances(command_name)

        command_name = os.path.splitext(os.path.basename(__file__))[0]
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
                CommandResponse(response=f"current status is: {output.status}")
            ]
        )

    def _process_command(self,
        workflow: Workflow, input: Signature.Input
    ) -> Signature.Output:
        """
        Process the cancel_pending_order command using tau2-bench tools.
        """
        try:
            db = RetailDB.load(RETAIL_DB_PATH)
            tools = RetailTools(db)
            
            order_id = input.order_id if input.order_id.startswith('#') else f'#{input.order_id}'
            
            order = tools.cancel_pending_order(
                order_id=order_id,
                reason=input.reason
            )
            
            return Signature.Output(
                status=f"Order {order_id} cancelled successfully. Status: {order.status}"
            )
            
        except ValueError as e:
            # Handle tool errors
            return Signature.Output(
                status=f"Error: {str(e)}"
            )
        except Exception as e:
            # Handle unexpected errors
            return Signature.Output(
                status=f"Unexpected error: {str(e)}"
            )
