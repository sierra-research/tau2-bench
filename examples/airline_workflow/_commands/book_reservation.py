from typing import List, Dict, Any, Optional

import fastworkflow
from pydantic import BaseModel, Field, ConfigDict
from fastworkflow.workflow import Workflow
from fastworkflow import CommandOutput, CommandResponse

from tau2.domains.airline.tools import AirlineTools
from tau2.domains.airline.data_model import FlightDB
from tau2.domains.airline.utils import AIRLINE_DB_PATH

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


class PassengerInfo(BaseModel):
    first_name: str = Field(
        description="First name of the passenger",
        examples=["John", "Sarah", "Michael"]
    )
    last_name: str = Field(
        description="Last name of the passenger", 
        examples=["Doe", "Smith", "Johnson"]
    )
    dob: Optional[str] = Field(
        default="NOT_FOUND",
        description="Date of birth in YYYY-MM-DD format or NOT_FOUND if unknown",
        pattern=r"^(\d{4}-\d{2}-\d{2}|NOT_FOUND)$",
        examples=["1990-01-01", "1985-05-15", "NOT_FOUND"]
    )


class PaymentMethodInfo(BaseModel):
    payment_id: str = Field(
        description="Payment ID from user profile",
        examples=["credit_card_7815826", "gift_card_4421486", "certificate_7504069"]
    )
    amount: float = Field(
        description="Amount to be charged to this payment method",
        examples=[250.50, 100.0, 500.25]
    )


class Signature:
    """Book a flight reservation with passengers, flights, and payment details"""
    class Input(BaseModel):
        user_id: str = Field(
            default="NOT_FOUND",
            description="The ID of the user booking the reservation",
            examples=["mia_li_3668", "chen_jackson_3290", "sara_doe_496"]
        )
        origin: str = Field(
            default="NOT_FOUND",
            description="Origin airport IATA code",
            pattern=r"^(NOT_FOUND|[A-Z]{3})$",
            examples=["JFK", "LAX", "ATL"]
        )
        destination: str = Field(
            default="NOT_FOUND",
            description="Destination airport IATA code",
            pattern=r"^(NOT_FOUND|[A-Z]{3})$",
            examples=["LAX", "MIA", "DFW"]
        )
        flight_type: str = Field(
            default="NOT_FOUND",
            description="Type of flight booking",
            pattern=r"^(NOT_FOUND|one_way|round_trip)$",
            examples=["one_way", "round_trip"]
        )
        cabin: str = Field(
            default="NOT_FOUND",
            description="Cabin class for the reservation",
            pattern=r"^(NOT_FOUND|basic_economy|economy|business)$",
            examples=["basic_economy", "economy", "business"]
        )
        flights: List[FlightInfo] = Field(
            default=[],
            description="List of flights for the reservation",
            examples=[[{"flight_number": "HAT001", "date": "2024-05-16"}]]
        )
        passengers: List[PassengerInfo] = Field(
            default=[],
            description="List of passengers for the reservation", 
            examples=[[{"first_name": "John", "last_name": "Doe", "dob": "1990-01-01"}]]
        )
        payment_methods: List[PaymentMethodInfo] = Field(
            default=[],
            description="List of payment methods and amounts",
            examples=[[{"payment_id": "credit_card_7815826", "amount": 250.0}]]
        )
        total_baggages: int = Field(
            default=0,
            description="Total number of baggage items",
            examples=[2, 3, 5],
            alias="total_baggage"
        )
        nonfree_baggages: int = Field(
            default=0,
            description="Number of non-free baggage items",
            examples=[0, 1, 2],
            alias="nonfree_baggage"
        )
        insurance: str = Field(
            default="NOT_FOUND",
            description="Whether to include travel insurance",
            pattern=r"^(NOT_FOUND|yes|no)$",
            examples=["yes", "no"]
        )

        model_config = ConfigDict(
            arbitrary_types_allowed=True, 
            validate_assignment=True, 
            populate_by_name=True
        )

    class Output(BaseModel):
        reservation_details: str = Field(
            description="JSON string containing the complete reservation details.",
            json_schema_extra={
                "used_by": ["get_reservation", "cancel_reservation", "modify_reservation"]
            }
        )

    # ------------------------------------------------------------------
    # Utterances
    # ------------------------------------------------------------------

    plain_utterances: List[str] = [
        "I want to book a round trip flight from JFK to LAX on May 16th for myself.",
        "Can you help me book a one-way business class flight from ATL to MIA on 2024-05-20?",
        "I need to book a reservation for two passengers from BOS to DFW on May 25th with economy class.",
        "Book me a direct flight from LAS to PHX on 2024-05-18 in basic economy, no insurance needed.",
        "I'd like to reserve seats on flight HAT001 for May 22nd, business class with travel insurance.",
        "Please book a round trip reservation from PHL to SEA, departing May 19th, returning May 26th.",
        "I want to book economy class seats for my family of 3 from DEN to LGA on 2024-05-21.",
        "Can you book me on flight HAT170 from MIA to BOS on May 17th? I'll need 2 checked bags.",
        "Book a business class reservation from LAX to ATL on May 23rd with my credit card.",
        "I need to make a reservation for May 24th from SEA to DFW, one passenger, economy class.",
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
        command_parameters: Signature.Input,
    ) -> CommandOutput:
        output = self._process_command(workflow, command_parameters)
        return CommandOutput(
            workflow_id=workflow.id,
            command_responses=[
                CommandResponse(response=f"Reservation booked successfully: {output.reservation_details}")
            ],
        )
    
    def _process_command(self, workflow: Workflow, input: Signature.Input) -> Signature.Output:
        """
        Process the book_reservation command using tau2-bench airline tools.
        """
        try:
            db = FlightDB.load(AIRLINE_DB_PATH)
            tools = AirlineTools(db)
            
            # Convert FlightInfo and PassengerInfo to dict if needed
            flights = [f.model_dump() if hasattr(f, 'model_dump') else f for f in input.flights]
            passengers = [p.model_dump() if hasattr(p, 'model_dump') else p for p in input.passengers]
            payment_methods = [pm.model_dump() if hasattr(pm, 'model_dump') else pm for pm in input.payment_methods]
            
            reservation = tools.book_reservation(
                user_id=input.user_id,
                origin=input.origin,
                destination=input.destination,
                flight_type=input.flight_type,
                cabin=input.cabin,
                flights=flights,
                passengers=passengers,
                payment_methods=payment_methods,
                total_baggages=input.total_baggages,
                nonfree_baggages=input.nonfree_baggages,
                insurance=input.insurance
            )
            
            import json
            reservation_json = reservation.model_dump_json(indent=2)
            return Signature.Output(reservation_details=reservation_json)
        except ValueError as e:
            return Signature.Output(reservation_details=f"Error: {str(e)}")
        except Exception as e:
            return Signature.Output(reservation_details=f"Unexpected error: {str(e)}")