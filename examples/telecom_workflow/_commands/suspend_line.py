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
    """Suspend a line (max 6 months with $5/month holding fee)"""
    
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
            description="The line identifier to suspend (starts with L)",
            pattern=r"^(L[A-Za-z0-9]+|NOT_FOUND)$",
            examples=["L1001", "L1002", "L1003"],
            json_schema_extra={
                "available_from": ["get_customer_by_id", "get_customer_by_phone", "get_customer_by_name"]
            }
        )
        
        reason: str = Field(
            default="NOT_FOUND",
            description="Reason for suspending the line",
            examples=["Going on vacation", "Traveling abroad", "Temporary loss", "Not needed temporarily"],
        )
        
        model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)
    
    class Output(BaseModel):
        status: str = Field(
            description="Status message indicating whether suspension succeeded, including details about holding fee",
        )
    
    plain_utterances: List[str] = [
        "I need to suspend my line temporarily while I'm traveling",
        "Can you put my phone service on hold for a few months?",
        "I want to suspend line L12345 because I'm going abroad",
        "Please pause my service - I won't need it for the next 3 months",
        "Can I temporarily suspend my line? I'm not using it right now",
        "I'd like to put my phone line on hold while I'm on vacation",
        "Suspend my line please, I lost my phone and need time to replace it",
        "Can you freeze my account temporarily? I don't need service right now",
        "I want to suspend my line for customer C12345 - going on extended leave",
        "Please put my phone service on pause, I'll be out of the country",
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
        Process the suspend_line command using tau2-bench tools.
        """
        try:
            db = TelecomDB.load(TELECOM_DB_PATH)
            tools = TelecomTools(db)
            
            result = tools.suspend_line(
                customer_id=input.customer_id,
                line_id=input.line_id,
                reason=input.reason
            )
            
            status_message = result.get('message', 'Line suspended successfully')
            
            return Signature.Output(status=status_message)
            
        except ValueError as e:
            return Signature.Output(status=f"Error: {str(e)}")
        except Exception as e:
            return Signature.Output(status=f"Unexpected error: {str(e)}")