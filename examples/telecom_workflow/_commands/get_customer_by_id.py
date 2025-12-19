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
    """Get customer details by customer ID"""
    
    class Input(BaseModel):
        customer_id: str = Field(
            default="NOT_FOUND",
            description="The unique customer identifier (starts with C)",
            pattern=r"^(C[A-Za-z0-9]+|NOT_FOUND)$",
            examples=["C1001", "C1002", "C1003"],
            json_schema_extra={
                "available_from": ["get_customer_by_phone", "get_customer_by_name"]
            }
        )
        
        model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)
    
    class Output(BaseModel):
        customer_details: str = Field(
            description="Full customer information including customer_id, name, contact info, line_ids, and bill_ids",
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
        "Can you pull up the account details for customer ID C12345?",
        "I need information about customer C98765",
        "Show me everything for customer ID Cabc123",
        "What are the details for customer C555444?",
        "Look up customer Cxyz789 for me",
        "Get me the account info for customer ID C111222",
        "I want to see the full profile for customer C333444",
        "Retrieve customer details for ID C666777",
        "Can you find customer Cabc999 in the system?",
        "Show customer information for C888999",
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
        Process the get_customer_by_id command using tau2-bench tools.
        """
        try:
            db = TelecomDB.load(TELECOM_DB_PATH)
            tools = TelecomTools(db)
            
            customer = tools.get_customer_by_id(customer_id=input.customer_id)
            
            import json
            customer_json = customer.model_dump_json(indent=2)
            
            return Signature.Output(customer_details=customer_json)
            
        except ValueError as e:
            return Signature.Output(customer_details=f"Error: {str(e)}")
        except Exception as e:
            return Signature.Output(customer_details=f"Unexpected error: {str(e)}")