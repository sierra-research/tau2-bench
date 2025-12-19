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
    """Retrieve customer's billing history"""
    
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
        
        limit: int = Field(
            default=12,
            description="Maximum number of bills to return (default: 12)",
            ge=1,
            le=100,
            examples=[12, 6, 24],
        )
        
        model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)
    
    class Output(BaseModel):
        bills: str = Field(
            description="List of customer bills ordered by issue date (newest first), including bill_id, amount, status, dates, and line items",
            json_schema_extra={
                "used_by": ["send_payment_request", "get_details_by_id"]
            }
        )
    
    plain_utterances: List[str] = [
        "Can you show me my recent bills?",
        "I need to see my billing history",
        "What are my last few invoices?",
        "Pull up my bills for customer C12345",
        "Show me the payment history for my account",
        "I want to check my past bills",
        "Can you get me the last 6 months of bills?",
        "What invoices do I have on file?",
        "Show my billing statements please",
        "I'd like to review my recent charges and bills",
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
                CommandResponse(response=f"Bills for customer:\n{output.bills}")
            ]
        )
    
    def _process_command(
        self,
        workflow: Workflow,
        input: Signature.Input
    ) -> Signature.Output:
        """
        Process the get_bills_for_customer command using tau2-bench tools.
        """
        try:
            db = TelecomDB.load(TELECOM_DB_PATH)
            tools = TelecomTools(db)
            
            bills = tools.get_bills_for_customer(
                customer_id=input.customer_id,
                limit=input.limit
            )
            
            import json
            bills_json = json.dumps(
                [bill.model_dump() for bill in bills],
                indent=2,
                default=str
            )
            
            return Signature.Output(bills=bills_json)
            
        except ValueError as e:
            return Signature.Output(bills=f"Error: {str(e)}")
        except Exception as e:
            return Signature.Output(bills=f"Unexpected error: {str(e)}")