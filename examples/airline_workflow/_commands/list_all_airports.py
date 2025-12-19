from typing import List, Dict, Any, Optional

import fastworkflow
from pydantic import BaseModel, Field, ConfigDict
from fastworkflow.workflow import Workflow
from fastworkflow import CommandOutput, CommandResponse

from tau2.domains.airline.tools import AirlineTools
from tau2.domains.airline.data_model import FlightDB
from tau2.domains.airline.utils import AIRLINE_DB_PATH

class Signature:
    """List all airports"""
    """Metadata and parameter definitions for `list_all_airports`."""

    class Input(BaseModel):
        """No parameters expected for this command."""

    class Output(BaseModel):
        status: str = Field(
            description="List of airports and their cities as a JSON string representation.",
            json_schema_extra={
                "used_by": ["search_direct_flight", "search_onestop_flight"]
            }
        )

    # ---------------------------------------------------------------------
    # Utterances
    # ---------------------------------------------------------------------

    plain_utterances: List[str] = [
        "What airports do you fly to?",
        "Can you show me all airports you serve?",
        "I want to see all available airports.",
        "What are all the airports in your network?",
        "Which airports can I fly from or to?",
        "Show me all destinations you offer.",
        "What cities do you have flights to?",
        "List all airports and cities you serve.",
        "I need to see all available airports.",
        "What are your airport options?",
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
        command_parameters: Signature.Input | None = None,
    ) -> CommandOutput:
        output = self._process_command(workflow)
        return CommandOutput(
            workflow_id=workflow.id,
            command_responses=[
                CommandResponse(response=f"Available airports: {output.status}")
            ],
        )
    
    def _process_command(self, workflow: Workflow) -> Signature.Output:
        """
        Process the list_all_airports command using tau2-bench airline tools.
        """
        try:
            db = FlightDB.load(AIRLINE_DB_PATH)
            tools = AirlineTools(db)
            airports = tools.list_all_airports()
            import json
            airports_json = json.dumps([a.model_dump() for a in airports], indent=2)
            return Signature.Output(status=airports_json)
        except Exception as e:
            return Signature.Output(status=f"Unexpected error: {str(e)}")