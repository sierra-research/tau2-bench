from typing import List, Dict, Any, Optional

import fastworkflow
from pydantic import BaseModel, Field, ConfigDict
from fastworkflow.workflow import Workflow
from fastworkflow import CommandOutput, CommandResponse

from tau2.domains.airline.tools import AirlineTools
from tau2.domains.airline.data_model import FlightDB
from tau2.domains.airline.utils import AIRLINE_DB_PATH

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
                "the list of reservation id's"
            ),
            json_schema_extra={
                "used_by": [
                    "cancel_reservation",
                    "get_reservation_details",
                    "search_direct_flight",
                    "search_onestop_flight",
                    "send_certificate",
                    "update_reservation_baggages",
                    "update_reservation_flights",
                    "update_reservation_passengers",
                ]
            },
        )

    plain_utterances: List[str] = [
        "Can you pull up my account info?",
        "I want to see all the reservations linked to my profile.",
        "What details do you have on my user account?",
        "Can you show me everything tied to my profile?",
        "I'd like to review my account and recent activity.",
        "retrieve user details"
    ]
    template_utterances: List[str] = []

    @staticmethod
    def generate_utterances(workflow: fastworkflow.Workflow, command_name: str) -> List[str]:
        utterance_definition = fastworkflow.RoutingRegistry.get_definition(workflow.folderpath)
        utterances_obj = utterance_definition.get_command_utterances(command_name)
        from fastworkflow.train.generate_synthetic import generate_diverse_utterances
        return generate_diverse_utterances(utterances_obj.plain_utterances, command_name)
    

class ResponseGenerator:
    def __call__(self, workflow: Workflow, command: str, command_parameters: Signature.Input) -> CommandOutput:
        output = self._process_command(workflow, command_parameters)
        response = (
            f'Response: {output.user_details}'
        )
        return CommandOutput(
            workflow_id=workflow.id,
            command_responses=[CommandResponse(response=response)],
        )
    
    def _process_command(self, workflow: Workflow, input: Signature.Input) -> Signature.Output:
        """
        Process the get_user_details command using tau2-bench airline tools.
        """
        try:
            db = FlightDB.load(AIRLINE_DB_PATH)
            tools = AirlineTools(db)
            user = tools.get_user_details(user_id=input.user_id)
            import json
            user_json = user.model_dump_json(indent=2)
            return Signature.Output(user_details=user_json)
        except ValueError as e:
            return Signature.Output(user_details=f"Error: {str(e)}")
        except Exception as e:
            return Signature.Output(user_details=f"Unexpected error: {str(e)}")