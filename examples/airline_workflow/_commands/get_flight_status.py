from typing import List, Dict, Any, Optional

import fastworkflow
from pydantic import BaseModel, Field, ConfigDict
from fastworkflow.workflow import Workflow
from fastworkflow import CommandOutput, CommandResponse

from tau2.domains.airline.tools import AirlineTools
from tau2.domains.airline.data_model import FlightDB
from tau2.domains.airline.utils import AIRLINE_DB_PATH


class Signature:
    """Get the status of a flight"""
    class Input(BaseModel):
        flight_number: str = Field(
            default="NOT_FOUND",
            description="The flight number",
            pattern=r"^([A-Z]{3}\d{3}|NOT_FOUND)$",
            examples=["HAT001", "HAT170", "HAT022"],
            json_schema_extra={
                "available_from": ["search_direct_flight", "search_onestop_flight", "get_reservation_details"]
            }
        )
        date: str = Field(
            default="NOT_FOUND",
            description="The date of the flight in YYYY-MM-DD format",
            pattern=r"^(NOT_FOUND|\d{4}-\d{2}-\d{2})$",
            examples=["2024-05-16", "2024-05-20", "2024-05-25"],
        )

        model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    class Output(BaseModel):
        flight_status: str = Field(
            description="The status of the flight"
        )

    plain_utterances: List[str] = [
        "What's the status of flight HAT001?",
        "Is flight HAT170 on time?",
        "Can you check the status of my flight?",
        "What's the current status of flight HAT022 on May 16th?",
        "Is my flight delayed?",
        "Check flight status for HAT001 on 2024-05-20",
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
        response = f"Flight status: {output.flight_status}"
        return CommandOutput(
            workflow_id=workflow.id,
            command_responses=[CommandResponse(response=response)],
        )
    
    def _process_command(self, workflow: Workflow, input: Signature.Input) -> Signature.Output:
        """
        Process the get_flight_status command using tau2-bench airline tools.
        """
        try:
            db = FlightDB.load(AIRLINE_DB_PATH)
            tools = AirlineTools(db)
            status = tools.get_flight_status(
                flight_number=input.flight_number,
                date=input.date
            )
            return Signature.Output(flight_status=status)
        except ValueError as e:
            return Signature.Output(flight_status=f"Error: {str(e)}")
        except Exception as e:
            return Signature.Output(flight_status=f"Unexpected error: {str(e)}")