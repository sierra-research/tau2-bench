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
    """Disable international roaming on a line"""
    
    class Input(BaseModel):
        customer_id: str = Field(
            default="NOT_FOUND",
            description="The unique customer identifier (starts with C)",
            pattern=r"^(C[A-Za-z0-9]+|NOT_FOUND)$",
            examples=["C1001", "C1002", "C1003"],
            json_schema_extra={
                "available_from": ["get_customer_by_phone", "get_customer_by_id", "get_customer_by_name"]
            }
        )
        
        line_id: str = Field(
            default="NOT_FOUND",
            description="The line identifier to disable roaming for (starts with L)",
            pattern=r"^(L[A-Za-z0-9]+|NOT_FOUND)$",
            examples=["L1001", "L1002", "L1003"],
            json_schema_extra={
                "available_from": ["get_customer_by_id", "get_customer_by_phone", "get_customer_by_name"]
            }
        )
        
        model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)
    
    class Output(BaseModel):
        status: str = Field(
            description="Status message indicating whether roaming was disabled successfully",
        )
    
    plain_utterances: List[str] = [
        "I'm back home, please turn off roaming",
        "Can you disable international roaming on my line?",
        "Turn off roaming for line L12345",
        "Please deactivate roaming, I don't need it anymore",
        "Disable roaming on my account - I'm back from traveling",
        "Can you switch off international roaming for my phone?",
        "Turn roaming off please, I don't want unexpected charges",
        "Deactivate roaming for customer C12345 line L67890",
        "I want to disable roaming to avoid extra fees",
        "Please turn off roaming on my line, I'm done traveling",
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
                CommandResponse(response=output.status)
            ]
        )
    
    def _process_command(
        self,
        workflow: Workflow,
        input: Signature.Input
    ) -> Signature.Output:
        """
        Process the disable_roaming command using tau2-bench tools.
        """
        try:
            db = TelecomDB.load(TELECOM_DB_PATH)
            tools = TelecomTools(db)
            
            result = tools.disable_roaming(
                customer_id=input.customer_id,
                line_id=input.line_id
            )
            
            return Signature.Output(status=str(result))
            
        except ValueError as e:
            return Signature.Output(status=f"Error: {str(e)}")
        except Exception as e:
            return Signature.Output(status=f"Unexpected error: {str(e)}")