from typing import List

import fastworkflow
from pydantic import BaseModel, Field, ConfigDict
from fastworkflow.workflow import Workflow
from fastworkflow import CommandOutput, CommandResponse
from fastworkflow.train.generate_synthetic import generate_diverse_utterances

from tau2.domains.telecom.tools import TelecomTools
from tau2.domains.telecom.data_model import TelecomDB
from tau2.domains.telecom.utils import TELECOM_DB_PATH


class Signature:
    """Get details for any ID (Customer, Line, Device, Bill, or Plan)"""
    
    class Input(BaseModel):
        id: str = Field(
            default="NOT_FOUND",
            description="The ID to retrieve details for. Must start with: L (Line), D (Device), B (Bill), C (Customer), or P (Plan)",
            pattern=r"^([LDBCP][A-Za-z0-9]+|NOT_FOUND)$",
            examples=["L123456", "D789012", "B345678", "C901234", "P567890"],
            json_schema_extra={
                "available_from": [
                    "get_customer_by_phone",
                    "get_customer_by_id",
                    "get_customer_by_name",
                    "get_bills_for_customer",
                    "get_data_usage"
                ]
            }
        )
        
        model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)
    
    class Output(BaseModel):
        details: str = Field(
            description="Detailed information for the specified ID type",
        )
    
    plain_utterances: List[str] = [
        "What are the details for ID L12345?",
        "Can you show me information about device D98765?",
        "Get me the details for bill B54321",
        "I need information on line ID Labc123",
        "Show details for plan P99999",
        "What's in customer record C11111?",
        "Look up the info for ID D22222",
        "Can you pull up details for B33333?",
        "Get information about line Lxyz789",
        "Show me what's in plan ID P44444",
    ]
    
    template_utterances: List[str] = []
    
    @staticmethod
    def generate_utterances(workflow: fastworkflow.Workflow, command_name: str) -> List[str]:
        utterance_definition = fastworkflow.RoutingRegistry.get_definition(workflow.folderpath)
        utterances_obj = utterance_definition.get_command_utterances(command_name)
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
                CommandResponse(response=f"Details for {command_parameters.id}:\n{output.details}")
            ]
        )
    
    def _process_command(
        self,
        workflow: Workflow,
        input: Signature.Input
    ) -> Signature.Output:
        """
        Process the get_details_by_id command using tau2-bench tools.
        """
        try:
            db = TelecomDB.load(TELECOM_DB_PATH)
            tools = TelecomTools(db)
            
            details_dict = tools.get_details_by_id(id=input.id)
            
            import json
            details_json = json.dumps(details_dict, indent=2, default=str)
            
            return Signature.Output(details=details_json)
            
        except ValueError as e:
            return Signature.Output(details=f"Error: {str(e)}")
        except Exception as e:
            return Signature.Output(details=f"Unexpected error: {str(e)}")