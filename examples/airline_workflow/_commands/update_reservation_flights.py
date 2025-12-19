from typing import List, Dict, Any, Optional

import fastworkflow
from pydantic import BaseModel, Field, ConfigDict
from fastworkflow.workflow import Workflow
from fastworkflow import CommandOutput, CommandResponse

from tau2.domains.airline.tools import AirlineTools
from tau2.domains.airline.data_model import FlightDB
from tau2.domains.airline.utils import AIRLINE_DB_PATH

class PaymentMethodInfo(BaseModel):
    payment_id: str = Field(
        description="Payment ID from user profile",
        examples=["credit_card_7815826", "gift_card_4421486", "certificate_7504069"]
    )
    amount: float = Field(
        description="Amount to be charged to this payment method",
        examples=[250.50, 100.0, 500.25]
    )


class FlightInfo(BaseModel):
    flight_number: str = Field(
        description="Flight number like 'HAT001'",
        examples=["HAT001", "HAT002", "HAT170"]
    )
    date: str = Field(
        description="Flight date in YYYY-MM-DD format",
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        examples=["2024-05-16", "2024-05-20", "2024-05-25"]
    )


class Signature:
    """Update the flight information of a reservation"""
    class Input(BaseModel):
        reservation_id: str = Field(
            default="NOT_FOUND",
            description="The reservation ID",
            pattern=r"^([A-Z0-9]{6}|NOT_FOUND)$",
            examples=["ZFA04Y", "4WQ150", "VAAOXJ"],
            json_schema_extra={
                "available_from": ["get_reservation_details"]
            }
        )
        cabin: str = Field(
            default="economy",
            description="The cabin class for the reservation",
            pattern=r"^(basic_economy|economy|business)$",
            examples=["basic_economy", "economy", "business"],
            json_schema_extra={
                "available_from": ["get_reservation_details"]
            }
        )
        flights: List[FlightInfo] = Field(
            default=[],
            description="An array of objects containing details about each flight in the ENTIRE new reservation. Even if a flight segment is not changed, it should still be included in the array.",
            examples=[[{"flight_number": "HAT170", "date": "2024-05-22"}, {"flight_number": "HAT022", "date": "2024-05-26"}]]
        )
        payment_id: str = Field(
            default="NOT_FOUND",
            # alias="payment_method_id",  # Accept both payment_id and payment_method_id
            description="The payment method ID for price differences",
            pattern=r"^((credit_card|gift_card|certificate)_\d+|NOT_FOUND)$",
            examples=["credit_card_4421486", "gift_card_1234567", "certificate_7504069"],
            json_schema_extra={
                "available_from": ["get_user_details", "get_reservation_details"]
            }
        )

        model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    class Output(BaseModel):
        status: str = Field(description="Whether flight update succeeded or error message.")
        
        model_config = ConfigDict(json_schema_extra={
            "example": {
                "status": "Reservation updated successfully"
            }
        })

    plain_utterances: List[str] = [
        "I need to change my flight reservation to different flights.",
        "Can I update the flights on my booking to new ones?",
        "Please change my flight details to new flight numbers and dates.",
        "I want to modify my flight reservation with different flights.",
        "How can I update my airline reservation to use different flights?",
        "I need to change my flight times and upgrade my cabin class.",
        "Can I switch my flights to different dates and flight numbers?",
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
            f'Response: Update result: {output.status}'
        )
        return CommandOutput(
            workflow_id=workflow.id,
            command_responses=[CommandResponse(response=response)],
        )
    
    def _process_command(self, workflow: Workflow, input: Signature.Input) -> Signature.Output:
        """
        Process the update_reservation_flights command using tau2-bench airline tools.
        """
        try:
            db = FlightDB.load(AIRLINE_DB_PATH)
            tools = AirlineTools(db)
            
            flights = [f.model_dump() if hasattr(f, 'model_dump') else f for f in input.flights]
            
            reservation = tools.update_reservation_flights(
                reservation_id=input.reservation_id,
                cabin=input.cabin,
                flights=flights,
                payment_id=input.payment_id
            )
            return Signature.Output(
                status=f"Flights updated successfully for reservation {input.reservation_id}"
            )
        except ValueError as e:
            return Signature.Output(status=f"Error: {str(e)}")
        except Exception as e:
            return Signature.Output(status=f"Unexpected error: {str(e)}")