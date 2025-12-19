from typing import List, Dict, Any, Optional

import fastworkflow
import os
from typing import Annotated
from pydantic import BaseModel, Field, ConfigDict
from fastworkflow.workflow import Workflow
from fastworkflow import CommandOutput, CommandResponse

from tau2.domains.airline.tools import AirlineTools
from tau2.domains.airline.data_model import FlightDB
from tau2.domains.airline.utils import AIRLINE_DB_PATH

class Signature:
    """Cancel the whole reservation"""
    class Input(BaseModel):
        reservation_id: Annotated[
            str,
            Field(
                default="NOT_FOUND",
                description="The reservation ID to cancel",
                pattern=r"^([A-Z0-9]{6}|NOT_FOUND)$",
                examples=["4WQ150", "VAAOXJ", "PGAGLM"],
                json_schema_extra={
                    "available_from": ["get_user_details", "get_reservation_details"]
                }
            )
        ]

        model_config = ConfigDict(
            arbitrary_types_allowed=True,
            validate_assignment=True
        )

    class Output(BaseModel):
        cancellation_status: str = Field(
            description="Status of the reservation cancellation including refund details",
        )

    plain_utterances = [
        "I want to cancel my reservation.",
        "Please cancel my flight booking.",
        "Can you cancel my entire reservation?",
        "I need to cancel my flight.",
        "Cancel my booking please.",
        "I'd like to cancel my reservation.",
        "Please cancel reservation 4WQ150.",
        "Can you help me cancel my flight reservation?",
        "I want to cancel my whole trip.",
        "Cancel my airline reservation."
    ]

    template_utterances = []

    @staticmethod
    def generate_utterances(workflow: fastworkflow.Workflow, command_name: str) -> list[str]:
        utterance_definition = fastworkflow.RoutingRegistry.get_definition(workflow.folderpath)
        utterances_obj = utterance_definition.get_command_utterances(command_name)

        command_name = os.path.splitext(os.path.basename(__file__))[0]
        
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
                CommandResponse(response=f"Cancellation result: {output.cancellation_status}")
            ]
        )
    
    def _process_command(self, workflow: Workflow, input: Signature.Input) -> Signature.Output:
        """
        Process the cancel_reservation command using tau2-bench airline tools.
        """
        try:
            db = FlightDB.load(AIRLINE_DB_PATH)
            tools = AirlineTools(db)
            reservation = tools.cancel_reservation(reservation_id=input.reservation_id)
            return Signature.Output(
                cancellation_status=f"Reservation {input.reservation_id} cancelled. Status: {reservation.status}"
            )
        except ValueError as e:
            return Signature.Output(cancellation_status=f"Error: {str(e)}")
        except Exception as e:
            return Signature.Output(cancellation_status=f"Unexpected error: {str(e)}")