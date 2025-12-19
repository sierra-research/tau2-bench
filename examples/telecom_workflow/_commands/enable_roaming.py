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
    """Enable international roaming on a line"""
    
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
            description="The line identifier to enable roaming for (starts with L)",
            pattern=r"^(L[A-Za-z0-9]+|NOT_FOUND)$",
            examples=["L1001", "L1002", "L1003"],
            json_schema_extra={
                "available_from": ["get_customer_by_id", "get_customer_by_phone", "get_customer_by_name"]
            }
        )
        
        model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)
    
    class Output(BaseModel):
        status: str = Field(
            description="Status message indicating whether roaming was enabled successfully",
        )
    
    plain_utterances: List[str] = [
        "I'm traveling internationally, can you enable roaming?",
        "Please turn on international roaming for my line",
        "I need roaming enabled for line L12345",
        "Can you activate roaming on my phone? I'm going abroad",
        "Enable international roaming please, I'm traveling to Europe",
        "Turn on roaming for my line - I'll be out of the country",
        "I need to use my phone overseas, enable roaming",
        "Can you switch on roaming for customer C12345 line L67890?",
        "Activate international roaming, I'm going on a trip",
        "Please enable roaming on my account for international travel",
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
        Process the enable_roaming command using tau2-bench tools.
        """
        try:
            db = TelecomDB.load(TELECOM_DB_PATH)
            tools = TelecomTools(db)
            
            result = tools.enable_roaming(
                customer_id=input.customer_id,
                line_id=input.line_id
            )
            
            if isinstance(result, dict):
                status_message = result.get('message', 'Roaming enabled successfully')
            else:
                status_message = str(result)
            
            return Signature.Output(status=status_message)
            
        except ValueError as e:
            return Signature.Output(status=f"Error: {str(e)}")
        except Exception as e:
            return Signature.Output(status=f"Unexpected error: {str(e)}")