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
    """Get user details"""
    class Input(BaseModel):
        user_id: str = Field(
            default="NOT_FOUND",
            description="The user ID to get details for",
            pattern=r"^([a-z]+_[a-z]+_\d+|NOT_FOUND)$",
            examples=["sara_doe_496"],
            json_schema_extra={
                "available_from": ["find_user_id_by_email", "find_user_id_by_name_zip"]
            }
        )

        model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    class Output(BaseModel):
        user_details: str = Field(
            description=(
                "Detailed user information such as "
                "first and last name, address, email, payment methods and "
                "the list of order id's"
            ),
            json_schema_extra={
                "used_by": [
                    "cancel_pending_order",
                    "exchange_delivered_order_items",
                    "get_order_details",
                    "modify_pending_order_address",
                    "modify_pending_order_items",
                    "modify_pending_order_payment",
                    "return_delivered_order_items",
                ]
            },
        )

    plain_utterances: List[str] = [
        "Can you pull up my account info?",
        "I want to see all the orders linked to my profile.",
        "What details do you have on my user account?",
        "Can you show me everything tied to my profile?",
        "I'd like to review my account and recent activity.",
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
        return CommandOutput(
            workflow_id=workflow.id,
            command_responses=[
                CommandResponse(response=output.user_details)
            ]
        )

    def _process_command(self,
        workflow: Workflow, input: Signature.Input
    ) -> Signature.Output:
        """
        Process the get_user_details command using tau2-bench tools.
        """
        try:
            db = RetailDB.load(RETAIL_DB_PATH)
            tools = RetailTools(db)
            
            user = tools.get_user_details(user_id=input.user_id)
            
            # Convert Pydantic model to JSON string
            user_json = user.model_dump_json(indent=2)
            
            return Signature.Output(user_details=user_json)
            
        except ValueError as e:
            return Signature.Output(user_details=f"Error: {str(e)}")
        except Exception as e:
            return Signature.Output(user_details=f"Unexpected error: {str(e)}")
