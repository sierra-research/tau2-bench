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
    """Find customer by phone number"""
    
    class Input(BaseModel):
        phone_number: str = Field(
            default="NOT_FOUND",
            description="The phone number to search for (format: 555-123-2001)",
            pattern=r"^(NOT_FOUND|\d{3}-\d{3}-\d{4})$",
            examples=["555-123-2001", "555-987-6543", "555-444-3333"],
        )
        
        model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)
    
    class Output(BaseModel):
        customer_details: str = Field(
            description="Customer information including customer_id, name, contact info, and associated line_ids",
            json_schema_extra={
                "used_by": [
                    "get_bills_for_customer",
                    "get_data_usage",
                    "suspend_line",
                    "resume_line",
                    "enable_roaming",
                    "disable_roaming",
                    "refuel_data",
                    "send_payment_request"
                ]
            }
        )
    
    plain_utterances: List[str] = [
        "Can you look up the account for phone number 555-1234?",
        "I need to find a customer by their phone number +1-555-123-4567",
        "What customer has the phone number 5551234567?",
        "Look up account info for this phone: 555-987-6543",
        "Find the customer associated with phone +15559876543",
        "Can you pull up the customer with phone number 555.123.4567?",
        "I want to check who owns the phone line 5551112222",
        "Search for customer by phone 555-444-3333",
        "Find account using phone number +1 555 789 0123",
        "What's the customer info for phone 5556667777?",
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
                CommandResponse(response=f"Customer details:\n{output.customer_details}")
            ]
        )
    
    def _process_command(
        self,
        workflow: Workflow,
        input: Signature.Input
    ) -> Signature.Output:
        """
        Process the get_customer_by_phone command using tau2-bench tools.
        """
        try:
            db = TelecomDB.load(TELECOM_DB_PATH)
            tools = TelecomTools(db)
            
            customer = tools.get_customer_by_phone(phone_number=input.phone_number)
            
            import json
            customer_json = customer.model_dump_json(indent=2)
            
            return Signature.Output(customer_details=customer_json)
            
        except ValueError as e:
            return Signature.Output(customer_details=f"Error: {str(e)}")
        except Exception as e:
            return Signature.Output(customer_details=f"Unexpected error: {str(e)}")