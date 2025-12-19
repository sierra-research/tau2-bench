from typing import List

import fastworkflow
from pydantic import BaseModel, Field, ConfigDict
from fastworkflow.workflow import Workflow
from fastworkflow import CommandOutput, CommandResponse

from tau2.domains.retail.tools import RetailTools
from tau2.domains.retail.data_model import RetailDB
from tau2.domains.retail.utils import RETAIL_DB_PATH


class Signature:
    """Find user id by name and zip"""
    class Input(BaseModel):
        first_name: str = Field(
            default="NOT_FOUND",
            description="The first name of the customer",
            pattern=r"^(NOT_FOUND|[A-Za-z]+)$",
            examples=["John"],
        )
        last_name: str = Field(
            default="NOT_FOUND",
            description="The last name of the customer",
            pattern=r"^(NOT_FOUND|[A-Za-z]+)$",
            examples=["Doe"],
        )
        zip: str = Field(
            default="NOT_FOUND",
            description="The zip code of the customer",
            pattern=r"^(NOT_FOUND|\d{5})$",
            examples=["12345"],
        )

        model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    class Output(BaseModel):
        user_id: str = Field(
            description="User identifier returned by lookup.",
            json_schema_extra={
                "used_by": ["get_user_details"]
            }
        )

    # ------------------------------------------------------------------
    # Utterances
    # ------------------------------------------------------------------

    plain_utterances: List[str] = [
        "I can't remember the email I used, but my name is John Doe and I live in 12345.",
        "I forgot my email address. Can you look me up with my name and zip code?",
        "My name is Sarah Parker and my zip is 90210 — can you help me find my account?",
        "I'm not sure which email I signed up with, but my name is Michael Lee and I live in 77001.",
        "Can you check if you have me under Amanda White in zip code 10001? I don't recall the email.",
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
                CommandResponse(response=f"User ID: {output.user_id}")
            ]
        )

    def _process_command(self,
        workflow: Workflow, input: Signature.Input
    ) -> Signature.Output:
        """Find user ID using tau2-bench's find_user_id_by_name_zip tool."""
        try:
            db = RetailDB.load(RETAIL_DB_PATH)
            tools = RetailTools(db)
            
            user_id = tools.find_user_id_by_name_zip(
                first_name=input.first_name,
                last_name=input.last_name,
                zip=input.zip
            )
            
            return Signature.Output(user_id=user_id)
            
        except ValueError as e:
            return Signature.Output(user_id=f"Error: {str(e)}")
        except Exception as e:
            return Signature.Output(user_id=f"Unexpected error: {str(e)}")
