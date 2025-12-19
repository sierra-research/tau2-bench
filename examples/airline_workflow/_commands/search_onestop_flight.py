from typing import List, Dict, Any, Optional

import fastworkflow
from pydantic import BaseModel, Field, ConfigDict
from fastworkflow.workflow import Workflow
from fastworkflow import CommandOutput, CommandResponse

from tau2.domains.airline.tools import AirlineTools
from tau2.domains.airline.data_model import FlightDB
from tau2.domains.airline.utils import AIRLINE_DB_PATH

class Signature:
    """Search for one-stop flights between two cities on a specific date"""
    class Input(BaseModel):
        origin: str = Field(
            default="NOT_FOUND",
            description="The origin city airport code in three letters",
            pattern=r"^(NOT_FOUND|[A-Z]{3})$",
            examples=["JFK", "LAX", "ATL"],
        )
        destination: str = Field(
            default="NOT_FOUND",
            description="The destination city airport code in three letters",
            pattern=r"^(NOT_FOUND|[A-Z]{3})$",
            examples=["LAX", "JFK", "MIA"],
        )
        date: str = Field(
            default="NOT_FOUND",
            description="The date of the flight in YYYY-MM-DD format",
            pattern=r"^(NOT_FOUND|\d{4}-\d{2}-\d{2})$",
            examples=["2024-05-16", "2024-05-20", "2024-05-25"],
        )

        model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    class Output(BaseModel):
        flight_results: str = Field(
            description="JSON string containing one-stop flight search results with connecting flights.",
            json_schema_extra={
                "used_by": ["book_flight", "check_flight_availability"]
            }
        )

    # ------------------------------------------------------------------
    # Utterances
    # ------------------------------------------------------------------

    plain_utterances: List[str] = [
        "I need to find one-stop flights from JFK to LAX on May 16th, 2024.",
        "Can you search for connecting flights from ATL to MIA on 2024-05-20?",
        "I'm looking for one-stop flights from BOS to DFW on May 25th.",
        "Find me flights with one connection from LAS to PHX on 2024-05-18.",
        "I need connecting flights from PHL to SEA on May 22nd, 2024.",
        "Search for one-stop flights from DEN to LGA on 2024-05-19.",
        "Can you find flights with a layover from MIA to BOS on May 17th?",
        "I want to book a connecting flight from LAX to ATL on 2024-05-21.",
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
                CommandResponse(response=f"One-stop flight search results: {output.flight_results}")
            ],
        )
    
    def _process_command(self, workflow: Workflow, input: Signature.Input) -> Signature.Output:
        """
        Process the search_onestop_flight command using tau2-bench airline tools.
        """
        try:
            db = FlightDB.load(AIRLINE_DB_PATH)
            tools = AirlineTools(db)
            flights = tools.search_onestop_flight(
                origin=input.origin,
                destination=input.destination,
                date=input.date
            )
            import json
            # Convert list of tuples to serializable format
            flights_json = json.dumps(
                [[f1.model_dump(), f2.model_dump()] for f1, f2 in flights],
                indent=2
            )
            return Signature.Output(flight_results=flights_json)
        except ValueError as e:
            return Signature.Output(flight_results=f"Error: {str(e)}")
        except Exception as e:
            return Signature.Output(flight_results=f"Unexpected error: {str(e)}")