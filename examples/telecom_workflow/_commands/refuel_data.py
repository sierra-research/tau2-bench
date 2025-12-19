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
    """Refuel/add data to a line and charge customer"""
    
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
            description="The line identifier to add data to (starts with L)",
            pattern=r"^(L[A-Za-z0-9]+|NOT_FOUND)$",
            examples=["L1001", "L1002", "L1003"],
            json_schema_extra={
                "available_from": ["get_customer_by_id", "get_customer_by_phone", "get_customer_by_name"]
            }
        )
        
        gb_amount: float = Field(
            default=0.0,
            description="Amount of data to add in gigabytes (must be positive)",
            gt=0,
            examples=[1.0, 5.0, 10.0, 2.5],
        )
        
        model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)
    
    class Output(BaseModel):
        status: str = Field(
            description="Status message with details about data added, new total, and charge amount",
        )
    
    plain_utterances: List[str] = [
        "I'm running out of data, can I buy more?",
        "Please add 5GB of data to my line",
        "I need to refuel data on line L12345",
        "Can you add more data to my plan? I'm almost at my limit",
        "Purchase 10GB of additional data for my account",
        "I need extra data this month, can you add 3GB?",
        "Buy more data for line L67890 please",
        "Add 2 gigabytes to my data allowance",
        "I want to refuel 7GB of data for customer C12345",
        "Can I get an additional 4GB added to my line?",
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
        Process the refuel_data command using tau2-bench tools.
        """
        try:
            db = TelecomDB.load(TELECOM_DB_PATH)
            tools = TelecomTools(db)
            
            result = tools.refuel_data(
                customer_id=input.customer_id,
                line_id=input.line_id,
                gb_amount=input.gb_amount
            )
            
            message = result.get('message', 'Data refueled successfully')
            new_refuel_amount = result.get('new_data_refueling_gb', 'N/A')
            charge = result.get('charge', 'N/A')
            
            status_message = f"{message}\nNew data refueling total: {new_refuel_amount} GB\nCharge: ${charge}"
            
            return Signature.Output(status=status_message)
            
        except ValueError as e:
            return Signature.Output(status=f"Error: {str(e)}")
        except Exception as e:
            return Signature.Output(status=f"Unexpected error: {str(e)}")