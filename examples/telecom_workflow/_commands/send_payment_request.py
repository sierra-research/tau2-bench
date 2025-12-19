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
    """Send payment request to customer for a specific bill"""
    
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
        
        bill_id: str = Field(
            default="NOT_FOUND",
            description="The bill identifier to request payment for (starts with B)",
            pattern=r"^(B[A-Za-z0-9]+|NOT_FOUND)$",
            examples=["B1001", "B1002", "B1003"],
            json_schema_extra={
                "available_from": ["get_bills_for_customer"]
            }
        )
        
        model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)
    
    class Output(BaseModel):
        status: str = Field(
            description="Status message indicating whether payment request was sent successfully",
        )
    
    plain_utterances: List[str] = [
        "I'm ready to pay my bill, can you send me the payment request?",
        "Please send a payment request for bill B12345",
        "I want to pay my outstanding bill, send me the payment link",
        "Can you request payment for the bill from last month?",
        "Send me a payment request for my latest invoice",
        "I'd like to pay bill B67890, please send the request",
        "Request payment for customer C12345 bill B54321",
        "Can you send the payment request for my overdue bill?",
        "I need to pay my bill, send me the payment details",
        "Please initiate a payment request for bill ID B99999",
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
        Process the send_payment_request command using tau2-bench tools.
        """
        try:
            db = TelecomDB.load(TELECOM_DB_PATH)
            tools = TelecomTools(db)
            
            result = tools.send_payment_request(
                customer_id=input.customer_id,
                bill_id=input.bill_id
            )
            
            return Signature.Output(status=str(result))
            
        except ValueError as e:
            return Signature.Output(status=f"Error: {str(e)}")
        except Exception as e:
            return Signature.Output(status=f"Unexpected error: {str(e)}")