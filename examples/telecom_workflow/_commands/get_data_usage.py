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
    """Get current billing cycle data usage for a line"""
    
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
            description="The line identifier (starts with L)",
            pattern=r"^(L[A-Za-z0-9]+|NOT_FOUND)$",
            examples=["L1001", "L1002", "L1003"],
            json_schema_extra={
                "available_from": ["get_customer_by_id", "get_customer_by_phone", "get_customer_by_name"]
            }
        )
        
        model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)
    
    class Output(BaseModel):
        usage_info: str = Field(
            description="Current billing cycle data usage including: data used (GB), data refueling amount (GB), plan data limit (GB), and cycle end date",
        )
    
    plain_utterances: List[str] = [
        "How much data have I used this month?",
        "Can you check my data usage for line L12345?",
        "What's my current data consumption?",
        "I need to see how much data I've used on my plan",
        "Show me the data usage for my phone line",
        "How much of my data allowance have I used?",
        "Can you tell me my remaining data for this billing cycle?",
        "What's the data usage on line Labc123?",
        "I want to check how much data I have left",
        "Show me my data usage details for customer C12345 line L67890",
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
                CommandResponse(response=f"Data usage information:\n{output.usage_info}")
            ]
        )
    
    def _process_command(
        self,
        workflow: Workflow,
        input: Signature.Input
    ) -> Signature.Output:
        """
        Process the get_data_usage command using tau2-bench tools.
        """
        try:
            db = TelecomDB.load(TELECOM_DB_PATH)
            tools = TelecomTools(db)
            
            usage_dict = tools.get_data_usage(
                customer_id=input.customer_id,
                line_id=input.line_id
            )
            
            import json
            usage_json = json.dumps(usage_dict, indent=2, default=str)
            
            return Signature.Output(usage_info=usage_json)
            
        except ValueError as e:
            return Signature.Output(usage_info=f"Error: {str(e)}")
        except Exception as e:
            return Signature.Output(usage_info=f"Unexpected error: {str(e)}")